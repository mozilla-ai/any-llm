import os
from unittest.mock import patch, Mock

from any_llm.provider import ApiConfig
from any_llm.providers.aws.aws import AwsProvider


def test_boto3_client_created_with_api_base() -> None:
    """Test that boto3.client is created with api_base as endpoint_url when provided."""
    custom_endpoint = "https://custom-bedrock-endpoint.amazonaws.com"
    region = "us-east-1"

    with patch.dict(os.environ, {"AWS_REGION": region}):
        with patch("any_llm.providers.aws.aws.AwsProvider._check_aws_credentials"):
            with patch("any_llm.providers.aws.aws._convert_messages", return_value=("", [])):
                with patch("any_llm.providers.aws.aws._convert_kwargs", return_value={}):
                    with patch("any_llm.providers.aws.aws._convert_response"):
                        with patch("boto3.client") as mock_boto3_client:
                            mock_client = Mock()
                            mock_boto3_client.return_value = mock_client
                            mock_client.converse.return_value = {
                                "output": {"message": {"content": [{"text": "response"}]}}
                            }

                            provider = AwsProvider(ApiConfig(api_base=custom_endpoint))
                            provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

                            mock_boto3_client.assert_called_once_with(
                                "bedrock-runtime", endpoint_url=custom_endpoint, region_name=region
                            )


def test_boto3_client_created_without_api_base() -> None:
    """Test that boto3.client is created with None endpoint_url when api_base is not provided."""
    region = "us-west-2"

    with patch.dict(os.environ, {"AWS_REGION": region}):
        with patch("any_llm.providers.aws.aws.AwsProvider._check_aws_credentials"):
            with patch("any_llm.providers.aws.aws._convert_messages", return_value=("", [])):
                with patch("any_llm.providers.aws.aws._convert_kwargs", return_value={}):
                    with patch("any_llm.providers.aws.aws._convert_response"):
                        with patch("boto3.client") as mock_boto3_client:
                            mock_client = Mock()
                            mock_boto3_client.return_value = mock_client
                            mock_client.converse.return_value = {
                                "output": {"message": {"content": [{"text": "response"}]}}
                            }

                            provider = AwsProvider(ApiConfig())
                            provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

                            mock_boto3_client.assert_called_once_with(
                                "bedrock-runtime", endpoint_url=None, region_name=region
                            )
