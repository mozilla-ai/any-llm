import json
import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

from any_llm.provider import ApiConfig
from any_llm.providers.aws.aws import AwsProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_aws_provider(region: str):  # type: ignore[no-untyped-def]
    with (
        patch.dict(os.environ, {"AWS_REGION": region}),
        patch("any_llm.providers.aws.aws.AwsProvider._check_aws_credentials"),
        patch("any_llm.providers.aws.aws._convert_messages", return_value=("", [])),
        patch("any_llm.providers.aws.aws._convert_kwargs", return_value={}),
        patch("any_llm.providers.aws.aws._convert_response"),
        patch("boto3.client") as mock_boto3_client,
    ):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}
        yield mock_boto3_client


def test_boto3_client_created_with_api_base() -> None:
    """Test that boto3.client is created with api_base as endpoint_url when provided."""
    custom_endpoint = "https://custom-bedrock-endpoint.amazonaws.com"
    region = "us-east-1"

    with mock_aws_provider(region) as mock_boto3_client:
        provider = AwsProvider(ApiConfig(api_base=custom_endpoint, api_key="test_key"))
        provider.completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=custom_endpoint, region_name=region)


def test_boto3_client_created_without_api_base() -> None:
    """Test that boto3.client is created with None endpoint_url when api_base is not provided."""
    region = "us-west-2"

    with mock_aws_provider(region) as mock_boto3_client:
        provider = AwsProvider(ApiConfig(api_key="test_key"))
        provider.completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None, region_name=region)


@contextmanager
def mock_aws_embedding_provider(region: str):  # type: ignore[no-untyped-def]
    """Mock AWS provider specifically for embedding tests."""
    with (
        patch.dict(os.environ, {"AWS_REGION": region}),
        patch("any_llm.providers.aws.aws.AwsProvider._check_aws_credentials"),
        patch("boto3.client") as mock_boto3_client,
    ):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        yield mock_boto3_client, mock_client


def test_embedding_single_string() -> None:
    """Test embedding with a single string input."""
    region = "us-east-1"
    model_id = "amazon.titan-embed-text-v1"
    input_text = "Hello world"

    mock_response_body = {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}

    with mock_aws_embedding_provider(region) as (mock_boto3_client, mock_client):
        mock_client.invoke_model.return_value = {"body": Mock(read=Mock(return_value=json.dumps(mock_response_body)))}

        provider = AwsProvider(ApiConfig(api_key="test_key"))
        response = provider.embedding(model_id, input_text)

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None, region_name=region)

        expected_request_body = {"inputText": input_text}
        mock_client.invoke_model.assert_called_once_with(modelId=model_id, body=json.dumps(expected_request_body))

        assert response.model == model_id
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[0].index == 0
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5


def test_embedding_list_of_strings() -> None:
    """Test embedding with a list of strings."""
    region = "us-east-1"
    model_id = "amazon.titan-embed-text-v1"
    input_texts = ["Hello world", "Goodbye world"]

    mock_response_bodies = [
        {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5},
        {"embedding": [0.4, 0.5, 0.6], "inputTextTokenCount": 6},
    ]

    with mock_aws_embedding_provider(region) as (mock_boto3_client, mock_client):
        mock_client.invoke_model.side_effect = [
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[0])))},
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[1])))},
        ]

        provider = AwsProvider(ApiConfig(api_key="test_key"))
        response = provider.embedding(model_id, input_texts)

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None, region_name=region)

        assert mock_client.invoke_model.call_count == 2
        expected_calls = [({"inputText": "Hello world"}, model_id), ({"inputText": "Goodbye world"}, model_id)]
        for i, (expected_body, expected_model) in enumerate(expected_calls):
            actual_call = mock_client.invoke_model.call_args_list[i]
            assert actual_call[1]["modelId"] == expected_model
            assert json.loads(actual_call[1]["body"]) == expected_body

        assert response.model == model_id
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[0].index == 0
        assert response.data[1].embedding == [0.4, 0.5, 0.6]
        assert response.data[1].index == 1
        assert response.usage.prompt_tokens == 11
        assert response.usage.total_tokens == 11
