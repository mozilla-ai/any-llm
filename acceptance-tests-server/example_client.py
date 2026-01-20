"""Example client tests demonstrating how to use the acceptance test server."""

import asyncio

import httpx

TEST_RUN_ID = "example-client-run"


async def setup_test_run():
    """Create a test run before running tests."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/test-runs",
            params={"test_run_id": TEST_RUN_ID, "description": "Example client test run"},
        )
        if response.status_code == 409:
            print(f"Test run '{TEST_RUN_ID}' already exists, reusing...")
        else:
            print(f"Created test run: {TEST_RUN_ID}")


async def test_basic_completion():
    """Test basic completion scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-basic",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        data = response.json()
        print("Basic completion:")
        print(f"  Passed: {data['_validation']['passed']}")
        print(f"  Response: {data['choices'][0]['message']['content']}")
        print()


async def test_tool_calls():
    """Test tool calls scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-tools",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"},
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
            },
        )
        data = response.json()
        print("Tool calls:")
        print(f"  Passed: {data['_validation']['passed']}")
        if data["choices"][0]["message"].get("tool_calls"):
            print(f"  Tool calls: {data['choices'][0]['message']['tool_calls']}")
        print()


async def test_tool_response():
    """Test tool response scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-tool-response",
                "messages": [
                    {"role": "user", "content": "What's the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": '{"temperature": "15C", "condition": "sunny"}',
                        "tool_call_id": "call_123",
                    },
                ],
            },
        )
        data = response.json()
        print("Tool response:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def test_streaming():
    """Test streaming scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-stream",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            print("Streaming:")
            print(f"  Validation passed: {response.headers.get('x-validation-passed')}")
            print(f"  Test run ID: {response.headers.get('x-test-run-id')}")
            print("  Chunks: ", end="")
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    import json

                    chunk = json.loads(line[6:])
                    if chunk["choices"][0]["delta"].get("content"):
                        print(chunk["choices"][0]["delta"]["content"], end="")
            print("\n")


async def test_structured_output():
    """Test structured output scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-structured",
                "messages": [{"role": "user", "content": "Give me a person"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        )
        data = response.json()
        print("Structured output:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def test_multi_turn():
    """Test multi-turn conversation scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-multi-turn",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        data = response.json()
        print("Multi-turn:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def test_system_message():
    """Test system message scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-system",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        data = response.json()
        print("System message:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def test_image_content():
    """Test image content scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-image",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.jpg"},
                            },
                        ],
                    }
                ],
            },
        )
        data = response.json()
        print("Image content:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def test_temperature_params():
    """Test temperature parameters scenario."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Test-Run-Id": TEST_RUN_ID},
            json={
                "model": "test-params",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
            },
        )
        data = response.json()
        print("Temperature params:")
        print(f"  Passed: {data['_validation']['passed']}")
        if not data["_validation"]["passed"]:
            print(f"  Errors: {data['_validation']['errors']}")
        print()


async def get_results():
    """Get all validation results for this test run."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        response = await client.get(f"/v1/test-runs/{TEST_RUN_ID}/summary")
        data = response.json()
        print("=" * 50)
        print(f"Summary for test run: {TEST_RUN_ID}")
        print(f"  Total: {data['total']}")
        print(f"  Passed: {data['passed']}")
        print(f"  Failed: {data['failed']}")
        print("  By scenario:")
        for scenario, counts in data.get("by_scenario", {}).items():
            print(f"    {scenario}: {counts['passed']} passed, {counts['failed']} failed")


async def main():
    """Run all example tests."""
    print("Running acceptance tests...")
    print("=" * 50)

    await setup_test_run()
    print()

    await test_basic_completion()
    await test_tool_calls()
    await test_tool_response()
    await test_streaming()
    await test_structured_output()
    await test_multi_turn()
    await test_system_message()
    await test_image_content()
    await test_temperature_params()

    await get_results()


if __name__ == "__main__":
    asyncio.run(main())
