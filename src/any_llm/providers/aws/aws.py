import os
import json
from typing import Any, Iterator

try:
    import boto3
    import instructor

except ImportError:
    msg = "boto3 or instructor is not installed. Please install it with `pip install any-llm-sdk[aws]`"
    raise ImportError(msg)

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.aws.utils import (
    _convert_response,
    _convert_kwargs,
    _convert_messages,
    _create_openai_chunk_from_aws_chunk,
    _create_openai_embedding_response_from_aws,
)


class AwsProvider(Provider):
    """AWS Bedrock Provider using boto3 and instructor for structured output."""

    PROVIDER_NAME = "AWS"
    ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/bedrock/"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = True

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        # This intentionally does not call super().__init__(config) because AWS has a different way of handling credentials
        self.config = config
        self.region_name = os.getenv("AWS_REGION", "us-east-1")

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
        credentials = session.get_credentials()  # type: ignore[no-untyped-call]

        bedrock_api_key = os.getenv(self.ENV_API_KEY_NAME)

        if credentials is None and bedrock_api_key is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the AWS Bedrock provider."""
        pass

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using AWS Bedrock with instructor support."""
        self._check_aws_credentials()

        client = boto3.client("bedrock-runtime", endpoint_url=self.config.api_base, region_name=self.region_name)  # type: ignore[no-untyped-call]

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

            return convert_instructor_response(instructor_response, model, "aws")

        stream = kwargs.pop("stream", False)

        request_config = _convert_kwargs(kwargs)

        system_message, formatted_messages = _convert_messages(messages)

        if stream:
            response_stream = client.converse_stream(
                modelId=model,
                messages=formatted_messages,
                system=system_message,
                **request_config,
            )
            stream_generator = response_stream["stream"]
            return (
                chunk
                for chunk in (_create_openai_chunk_from_aws_chunk(item, model=model) for item in stream_generator)
                if chunk is not None
            )
        else:
            response = client.converse(
                modelId=model,
                messages=formatted_messages,
                system=system_message,
                **request_config,
            )

            return _convert_response(response)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using AWS Bedrock."""
        self._check_aws_credentials()

        client = boto3.client("bedrock-runtime", endpoint_url=self.config.api_base, region_name=self.region_name)  # type: ignore[no-untyped-call]

        input_texts = [inputs] if isinstance(inputs, str) else inputs

        embedding_data = []
        total_tokens = 0

        for index, text in enumerate(input_texts):
            request_body = {"inputText": text}

            if "dimensions" in kwargs:
                request_body["dimensions"] = kwargs["dimensions"]
            if "normalize" in kwargs:
                request_body["normalize"] = kwargs["normalize"]

            # Make the API call
            response = client.invoke_model(modelId=model, body=json.dumps(request_body))

            # Parse the response
            response_body = json.loads(response["body"].read())

            embedding_data.append({"embedding": response_body["embedding"], "index": index})

            total_tokens += response_body.get("inputTextTokenCount", 0)

        return _create_openai_embedding_response_from_aws(embedding_data, model, total_tokens)
