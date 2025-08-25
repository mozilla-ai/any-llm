import asyncio
import functools
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError
from any_llm.logging import logger
from any_llm.provider import ApiConfig, Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.utils.instructor import _convert_instructor_response

MISSING_PACKAGES_ERROR = None
try:
    import boto3
    import instructor

    from .utils import (
        _convert_kwargs,
        _convert_messages,
        _convert_response,
        _create_openai_chunk_from_aws_chunk,
        _create_openai_embedding_response_from_aws,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class AwsProvider(Provider):
    """AWS Bedrock Provider using boto3 and instructor for structured output."""

    PROVIDER_NAME = "aws"
    ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/bedrock/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        # This intentionally does not call super().__init__(config) because AWS has a different way of handling credentials
        self._verify_no_missing_packages()
        self.config = config
        self.region_name = os.getenv("AWS_REGION", "us-east-1")

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
        credentials = session.get_credentials()  # type: ignore[no-untyped-call]

        bedrock_api_key = os.getenv(self.ENV_API_KEY_NAME)

        if credentials is None and bedrock_api_key is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using AWS Bedrock with instructor support."""
        logger.warning("AWS Bedrock client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        # create partial function of sync call
        call_sync_partial: Callable[[], ChatCompletion | Iterator[ChatCompletionChunk]] = functools.partial(
            self.completion, params, **kwargs
        )

        result = await loop.run_in_executor(None, call_sync_partial)

        if isinstance(result, ChatCompletion):
            return result

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in result:
                yield chunk

        return _stream()

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using AWS Bedrock with instructor support."""
        self._check_aws_credentials()

        client = boto3.client("bedrock-runtime", endpoint_url=self.config.api_base, region_name=self.region_name)  # type: ignore[no-untyped-call]

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        completion_kwargs = params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "response_format", "stream", "parallel_tool_calls"},
        )
        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)

            instructor_client = instructor.from_bedrock(client)

            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)

            instructor_response = instructor_client.chat.completions.create(
                model=params.model_id,
                messages=params.messages,  # type: ignore[arg-type]
                response_model=params.response_format,
                **completion_kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, "aws")

        completion_kwargs = _convert_kwargs(completion_kwargs)

        system_message, formatted_messages = _convert_messages(params.messages)

        if params.stream:
            response_stream = client.converse_stream(
                modelId=params.model_id,
                messages=formatted_messages,
                system=system_message,
                **completion_kwargs,
            )
            stream_generator = response_stream["stream"]
            return (
                chunk
                for chunk in (
                    _create_openai_chunk_from_aws_chunk(item, model=params.model_id) for item in stream_generator
                )
                if chunk is not None
            )
        response = client.converse(
            modelId=params.model_id,
            messages=formatted_messages,
            system=system_message,
            **completion_kwargs,
        )

        return _convert_response(response)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        logger.warning("AWS Bedrock client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        # create partial function of sync call
        call_sync_partial: Callable[[], CreateEmbeddingResponse] = functools.partial(
            self.embedding, model, inputs, **kwargs
        )

        return await loop.run_in_executor(None, call_sync_partial)

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

            response = client.invoke_model(modelId=model, body=json.dumps(request_body))

            response_body = json.loads(response["body"].read())

            embedding_data.append({"embedding": response_body["embedding"], "index": index})

            total_tokens += response_body.get("inputTextTokenCount", 0)

        return _create_openai_embedding_response_from_aws(embedding_data, model, total_tokens)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = boto3.client("bedrock", endpoint_url=self.config.api_base, region_name=self.region_name)  # type: ignore[no-untyped-call]
        models_list = client.list_foundation_models(**kwargs).get("modelSummaries", [])
        # AWS doesn't provide a creation date for models
        # AWS doesn't provide typing, but per https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
        # the modelId is a string and will not be None
        return [Model(id=model["modelId"], object="model", created=0, owned_by="aws") for model in models_list]
