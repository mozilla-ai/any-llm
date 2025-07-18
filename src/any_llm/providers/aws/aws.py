import os
from typing import Any, Optional

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

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self.config = config
        self.client: Optional[Any] = None
        self.instructor_client: Optional[Any] = None

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        try:
            # Create a session to check if credentials are available
            session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
            credentials = session.get_credentials()  # type: ignore[no-untyped-call]

            bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

            if credentials is None and bedrock_api_key is None:
                raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

        except Exception as e:
            if isinstance(e, MissingApiKeyError):
                raise
            # If any other error o  ccurs while checking credentials, treat as missing
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME) from e

    def _initialize_clients(self) -> None:
        """Initialize both regular and instructor clients."""
        if self.client is None:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)  # type: ignore[no-untyped-call]

        if self.instructor_client is None:
            self.instructor_client = instructor.from_bedrock(self.client)

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
        # Initialize clients
        self._initialize_clients()

        # Check credentials before creating client
        self._check_aws_credentials()

        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")

            # Use instructor for structured output
            assert self.instructor_client is not None  # For mypy
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_format,
                **kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "aws")

        # Regular completion flow
        system_message, formatted_messages = _convert_messages(messages)
        request_config = _convert_kwargs(kwargs)

        assert self.client is not None  # For mypy
        response = self.client.converse(
            modelId=model,
            messages=formatted_messages,
            system=system_message,
            **request_config,
        )

        # Convert to OpenAI format
        return _convert_response(response)
