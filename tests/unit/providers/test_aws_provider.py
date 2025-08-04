import os
from contextlib import contextmanager
from unittest.mock import patch, Mock

from any_llm.provider import ApiConfig
from any_llm.providers.aws.aws import AwsProvider


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
        provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=custom_endpoint, region_name=region)


def test_boto3_client_created_without_api_base() -> None:
    """Test that boto3.client is created with None endpoint_url when api_base is not provided."""
    region = "us-west-2"

    with mock_aws_provider(region) as mock_boto3_client:
        provider = AwsProvider(ApiConfig(api_key="test_key"))
        provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None, region_name=region)
