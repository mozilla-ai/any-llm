import asyncio
import functools
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError
from any_llm.logging import logger
from any_llm.provider import ClientConfig, Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.utils.instructor import _convert_instructor_response

MISSING_PACKAGES_ERROR = None
try:
    import boto3

    from .utils import (
        _convert_params,
        _convert_response,
        _create_openai_chunk_from_sagemaker_chunk,
        _create_openai_embedding_response_from_sagemaker,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class SagemakerProvider(Provider):
    """AWS SageMaker Provider using boto3 for inference endpoints."""

    PROVIDER_NAME = "sagemaker"
    ENV_API_KEY_NAME = "None"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/sagemaker/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, config: ClientConfig) -> None:
        """Initialize AWS SageMaker provider."""
        logger.warning(
            "AWS Sagemaker Support is experimental and may have issues. Please file an ticket at https://github.com/mozilla-ai/any-llm/issues if you encounter any issues."
        )
        # This intentionally does not call super().__init__(config) because AWS has a different way of handling credentials
        self._verify_no_missing_packages()
        self.config = config
        self.region_name = os.getenv("AWS_REGION", "us-east-1")

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
        credentials = session.get_credentials()  # type: ignore[no-untyped-call]

        if credentials is None:
            raise MissingApiKeyError(
                provider_name=self.PROVIDER_NAME, env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using AWS SageMaker."""
        logger.warning("AWS SageMaker client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

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
        """Create a chat completion using AWS SageMaker with instructor support."""
        self._check_aws_credentials()

        client = boto3.client(  # type: ignore[no-untyped-call]
            "sagemaker-runtime",
            endpoint_url=self.config.api_base,
            region_name=self.region_name,
            **(self.config.client_args if self.config.client_args else {}),
        )

        completion_kwargs = _convert_params(params, kwargs)

        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)

            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)

            response = client.invoke_endpoint(
                EndpointName=params.model_id,
                Body=json.dumps(completion_kwargs),
                ContentType="application/json",
            )

            response_body = json.loads(response["Body"].read())

            try:
                structured_response = params.response_format.model_validate(response_body)
                return _convert_instructor_response(structured_response, params.model_id, "aws")
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse structured response: %s", e)
                return _convert_response(response_body, params.model_id)

        if params.stream:
            response = client.invoke_endpoint_with_response_stream(
                EndpointName=params.model_id,
                Body=json.dumps(completion_kwargs),
                ContentType="application/json",
            )

            event_stream = response["Body"]
            return (
                chunk
                for chunk in (
                    _create_openai_chunk_from_sagemaker_chunk(event, model=params.model_id) for event in event_stream
                )
                if chunk is not None
            )

        response = client.invoke_endpoint(
            EndpointName=params.model_id,
            Body=json.dumps(completion_kwargs),
            ContentType="application/json",
        )

        response_body = json.loads(response["Body"].read())
        return _convert_response(response_body, params.model_id)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        logger.warning("AWS SageMaker client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

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
        """Create embeddings using AWS SageMaker."""
        self._check_aws_credentials()

        client = boto3.client(
            "sagemaker-runtime",
            endpoint_url=self.config.api_base,
            region_name=self.region_name,
            **(self.config.client_args if self.config.client_args else {}),
        )  # type: ignore[no-untyped-call]

        input_texts = [inputs] if isinstance(inputs, str) else inputs

        embedding_data = []
        total_tokens = 0

        for index, text in enumerate(input_texts):
            request_body = {"inputs": text}

            if "dimensions" in kwargs:
                request_body["dimensions"] = kwargs["dimensions"]
            if "normalize" in kwargs:
                request_body["normalize"] = kwargs["normalize"]

            response = client.invoke_endpoint(
                EndpointName=model,
                Body=json.dumps(request_body),
                ContentType="application/json",
            )

            response_body = json.loads(response["Body"].read())

            if "embeddings" in response_body:
                embedding = (
                    response_body["embeddings"][0]
                    if isinstance(response_body["embeddings"], list)
                    else response_body["embeddings"]
                )
            elif "embedding" in response_body:
                embedding = response_body["embedding"]
            else:
                embedding = response_body

            embedding_data.append({"embedding": embedding, "index": index})
            total_tokens += response_body.get("usage", {}).get("prompt_tokens", len(text.split()))

        return _create_openai_embedding_response_from_sagemaker(embedding_data, model, total_tokens)
