import asyncio
import functools
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from any_llm.config import ClientConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.logging import logger
from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model
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
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for SageMaker API."""
        return _convert_params(params, kwargs)

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert SageMaker response to OpenAI format."""
        model = response.get("model", "")
        return _convert_response(response, model)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert SageMaker chunk response to OpenAI format."""
        model = kwargs.get("model", "")
        chunk = _create_openai_chunk_from_sagemaker_chunk(response, model)
        if chunk is None:
            msg = "Failed to convert SageMaker chunk to OpenAI format"
            raise ValueError(msg)
        return chunk

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for SageMaker."""
        return kwargs

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert SageMaker embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_sagemaker(
            response["embedding_data"], response["model"], response["total_tokens"]
        )

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert SageMaker list models response to OpenAI format."""
        return []

    def __init__(self, config: ClientConfig) -> None:
        """Initialize AWS SageMaker provider."""
        logger.warning(
            "AWS Sagemaker Support is experimental and may not work as expected. Please file an ticket at https://github.com/mozilla-ai/any-llm/issues if you encounter any issues."
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

        completion_kwargs = self._convert_completion_params(params, **kwargs)

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
                return self._convert_completion_response({"model": params.model_id, **response_body})

        if params.stream:
            response = client.invoke_endpoint_with_response_stream(
                EndpointName=params.model_id,
                Body=json.dumps(completion_kwargs),
                ContentType="application/json",
            )

            event_stream = response["Body"]
            return (
                self._convert_completion_chunk_response(event, model=params.model_id)
                for event in event_stream
                if _create_openai_chunk_from_sagemaker_chunk(event, model=params.model_id) is not None
            )

        response = client.invoke_endpoint(
            EndpointName=params.model_id,
            Body=json.dumps(completion_kwargs),
            ContentType="application/json",
        )

        response_body = json.loads(response["Body"].read())
        return self._convert_completion_response({"model": params.model_id, **response_body})

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

        response_data = {"embedding_data": embedding_data, "model": model, "total_tokens": total_tokens}
        return self._convert_embedding_response(response_data)
