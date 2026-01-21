//! Acceptance tests for the any-llm Rust client.

use std::collections::HashMap;
use std::env;

use any_llm_rs::{AnyLLM, CompletionOptions, Message, Provider, StreamOptions};
use futures::StreamExt;
use serde::Deserialize;

fn get_base_url() -> String {
    env::var("TEST_SERVER_URL").unwrap_or_else(|_| "http://localhost:8080/v1".to_string())
}

const DUMMY_API_KEY: &str = "test-key";

#[derive(Debug, Deserialize)]
struct TestScenario {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    options: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TestDataResponse {
    scenarios: HashMap<String, TestScenario>,
}

async fn load_scenarios(
    server_base: &str,
) -> Result<HashMap<String, TestScenario>, reqwest::Error> {
    let url = format!("{}/v1/test-data", server_base);
    let response: TestDataResponse = reqwest::get(&url).await?.json().await?;
    Ok(response.scenarios)
}

async fn create_test_run(server_base: &str, test_run_id: &str) -> Result<(), reqwest::Error> {
    let url = format!(
        "{}/v1/test-runs?test_run_id={}&description=Rust%20acceptance%20tests",
        server_base, test_run_id
    );
    reqwest::Client::new().post(&url).send().await?;
    Ok(())
}

fn generate_test_run_id() -> String {
    format!(
        "rs-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    )
}

#[tokio::test]
async fn test_basic_completion() {
    let base_url = get_base_url();
    let server_base = base_url.replace("/v1", "");
    let test_run_id = generate_test_run_id();

    let _ = create_test_run(&server_base, &test_run_id).await;

    let scenarios = match load_scenarios(&server_base).await {
        Ok(s) => s,
        Err(_) => return, // Skip if server not available
    };

    let mut headers = HashMap::new();
    headers.insert("X-Test-Run-Id".to_string(), test_run_id);

    let client = AnyLLM::create(
        Provider::OpenAI,
        Some(DUMMY_API_KEY.to_string()),
        Some(base_url),
        Some(headers),
    )
    .expect("Failed to create client");

    let scenario = scenarios
        .get("basic_completion")
        .expect("Scenario not found");

    let response = client
        .completion(&scenario.model, scenario.messages.clone(), None)
        .await
        .expect("Completion failed");

    assert!(!response.choices.is_empty());
}

#[tokio::test]
async fn test_streaming() {
    let base_url = get_base_url();
    let server_base = base_url.replace("/v1", "");
    let test_run_id = generate_test_run_id();

    let _ = create_test_run(&server_base, &test_run_id).await;

    let scenarios = match load_scenarios(&server_base).await {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut headers = HashMap::new();
    headers.insert("X-Test-Run-Id".to_string(), test_run_id);

    let client = AnyLLM::create(
        Provider::OpenAI,
        Some(DUMMY_API_KEY.to_string()),
        Some(base_url),
        Some(headers),
    )
    .expect("Failed to create client");

    let scenario = scenarios.get("streaming").expect("Scenario not found");

    let options = CompletionOptions {
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
        }),
        ..Default::default()
    };

    let mut stream = client
        .completion_stream(&scenario.model, scenario.messages.clone(), Some(options))
        .await
        .expect("Stream creation failed");

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
        chunks.push(result.expect("Chunk error"));
    }

    assert!(!chunks.is_empty());
}

#[tokio::test]
async fn test_all_scenarios() {
    let base_url = get_base_url();
    let server_base = base_url.replace("/v1", "");
    let test_run_id = generate_test_run_id();

    let _ = create_test_run(&server_base, &test_run_id).await;

    let scenarios = match load_scenarios(&server_base).await {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut headers = HashMap::new();
    headers.insert("X-Test-Run-Id".to_string(), test_run_id);

    let client = AnyLLM::create(
        Provider::OpenAI,
        Some(DUMMY_API_KEY.to_string()),
        Some(base_url),
        Some(headers),
    )
    .expect("Failed to create client");

    for (name, scenario) in &scenarios {
        println!("Running scenario: {}", name);

        if scenario.stream {
            let mut stream = client
                .completion_stream(&scenario.model, scenario.messages.clone(), None)
                .await
                .expect("Stream failed");

            let mut count = 0;
            while let Some(Ok(_)) = stream.next().await {
                count += 1;
            }
            assert!(count > 0, "Scenario {} should receive chunks", name);
        } else {
            let response = client
                .completion(&scenario.model, scenario.messages.clone(), None)
                .await
                .expect("Completion failed");

            assert!(
                !response.choices.is_empty(),
                "Scenario {} should have choices",
                name
            );
        }
    }
}
