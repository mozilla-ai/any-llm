# mypy: disable-error-code="no-untyped-call"
import asyncio
import functools
import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model

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


class SagemakerProvider(AnyLLM):
    """AWS SageMaker Provider using boto3 for inference endpoints."""

    PROVIDER_NAME = "sagemaker"
    ENV_API_KEY_NAME = "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/sagemaker/"

    SUPPORTS_BATCH = False
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

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        logger.warning(
            "AWS Sagemaker Support is experimental and may not work as expected. Please file an ticket at https://github.com/mozilla-ai/any-llm/issues if you encounter any issues."
        )
        self.client = boto3.client(
            "sagemaker-runtime",
            endpoint_url=api_base,
            **kwargs,
        )

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        session = boto3.Session()  # type: ignore[attr-defined]
        credentials = session.get_credentials()
        if credentials is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)
        return api_key

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using AWS SageMaker."""
        logger.warning("AWS SageMaker client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        call_sync_partial: Callable[[], ChatCompletion | Iterator[ChatCompletionChunk]] = functools.partial(
            self._completion, params, **kwargs
        )

        result = await loop.run_in_executor(None, call_sync_partial)

        if isinstance(result, ChatCompletion):
            return result

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in result:
                yield chunk

        return _stream()

    def _completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.response_format:
            param = "response_format"
            raise UnsupportedParameterError(param, "sagemaker")

        if params.stream:
            response = self.client.invoke_endpoint_with_response_stream(
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

        response = self.client.invoke_endpoint(
            EndpointName=params.model_id,
            Body=json.dumps(completion_kwargs),
            ContentType="application/json",
        )

        response_body = json.loads(response["Body"].read())
        return self._convert_completion_response({"model": params.model_id, **response_body})

    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        logger.warning("AWS SageMaker client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        call_sync_partial: Callable[[], CreateEmbeddingResponse] = functools.partial(
            self._embedding, model, inputs, **kwargs
        )

        return await loop.run_in_executor(None, call_sync_partial)

    def _embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using AWS SageMaker."""
        input_texts = [inputs] if isinstance(inputs, str) else inputs

        embedding_data = []
        total_tokens = 0

        for index, text in enumerate(input_texts):
            request_body = {"inputs": text}

            if "dimensions" in kwargs:
                request_body["dimensions"] = kwargs["dimensions"]
            if "normalize" in kwargs:
                request_body["normalize"] = kwargs["normalize"]

            response = self.client.invoke_endpoint(
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
