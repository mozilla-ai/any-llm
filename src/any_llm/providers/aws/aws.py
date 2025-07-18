import os
from typing import Any

try:
    import boto3
    import instructor
except ImportError:
    msg = "boto3 or instructor is not installed. Please install it with `pip install any-llm-sdk[aws]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.providers.aws.utils import _convert_response, _convert_kwargs, _convert_messages


class AwsProvider(Provider):
    """AWS Bedrock Provider using boto3 and instructor for structured output."""

    PROVIDER_NAME = "AWS"
    ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/bedrock/"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self.config = config

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
        credentials = session.get_credentials()  # type: ignore[no-untyped-call]

        bedrock_api_key = os.getenv(self.ENV_API_KEY_NAME)

        if credentials is None and bedrock_api_key is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the AWS Bedrock provider."""
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using AWS Bedrock with instructor support."""
        self._check_aws_credentials()
        client = boto3.client("bedrock-runtime", region_name=self.region_name)  # type: ignore[no-untyped-call]

        if "response_format" in kwargs:
            instructor_client = instructor.from_bedrock(client)
            response_format = kwargs.pop("response_format")

            # Use instructor for structured output
            instructor_response = instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "aws")

        # Regular completion flow
        system_message, formatted_messages = _convert_messages(messages)
        request_config = _convert_kwargs(kwargs)

        response = client.converse(
            modelId=model,
            messages=formatted_messages,
            system=system_message,
            **request_config,
        )

        # Convert to OpenAI format
        return _convert_response(response)
