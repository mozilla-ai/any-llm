import os
from typing import Any, Optional

try:
    import boto3
    import instructor
except ImportError:
    msg = "boto3 or instructor is not installed. Please install it with `pip install any-llm-sdk[aws]`"
    raise ImportError(msg)

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.types import ChatCompletion, ChatCompletionChunk, Stream
from any_llm.providers.aws.utils import _convert_response, _convert_kwargs, _convert_messages


class AwsProvider(Provider):
    """AWS Bedrock Provider using boto3 with instructor support."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        # AWS uses region from environment variables or default
        self.region_name = os.getenv("AWS_REGION", "us-east-1")

        # Store config for later use
        self.config = config

        # Don't create client during init to avoid test failures
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
                raise MissingApiKeyError(
                    provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )

        except Exception as e:
            if isinstance(e, MissingApiKeyError):
                raise
            # If any other error o  ccurs while checking credentials, treat as missing
            raise MissingApiKeyError(
                provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            ) from e

    def _initialize_clients(self) -> None:
        """Initialize both regular and instructor clients."""
        if self.client is None:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)  # type: ignore[no-untyped-call]

        if self.instructor_client is None:
            self.instructor_client = instructor.from_bedrock(self.client)

    def completion(
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

        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", "AWS Bedrock")

        # Handle response_format for structured output
        if "response_format" in kwargs:
            if kwargs.get("stream", False):
                raise UnsupportedParameterError("response_format with streaming", "AWS Bedrock")

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
            instructor_result: ChatCompletion = convert_instructor_response(instructor_response, model, "aws")
            return instructor_result

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
        result: ChatCompletion = _convert_response(response)
        return result
