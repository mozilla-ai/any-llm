# mypy: disable-error-code="no-untyped-call"
import asyncio
import functools
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    import boto3

    from .utils import (
        _convert_params,
        _convert_response,
        _create_openai_chunk_from_aws_chunk,
        _create_openai_embedding_response_from_aws,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class BedrockProvider(AnyLLM):
    """AWS Bedrock Provider using boto3."""

    PROVIDER_NAME = "bedrock"
    ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/bedrock/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for AWS API."""
        return _convert_params(params, kwargs)

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert AWS Bedrock response to OpenAI format."""
        return _convert_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert AWS Bedrock chunk response to OpenAI format."""
        model = kwargs.get("model", "")
        chunk = _create_openai_chunk_from_aws_chunk(response, model)
        if chunk is None:
            msg = "Failed to convert AWS chunk to OpenAI format"
            raise ValueError(msg)
        return chunk

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for AWS Bedrock."""
        # For bedrock, we don't need to convert the params, just pass them through
        return kwargs

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert AWS Bedrock embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_aws(
            response["embedding_data"], response["model"], response["total_tokens"]
        )

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert AWS Bedrock list models response to OpenAI format."""
        models_list = response.get("modelSummaries", [])
        # AWS doesn't provide a creation date for models
        # AWS doesn't provide typing, but per https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
        # the modelId is a string and will not be None
        return [Model(id=model["modelId"], object="model", created=0, owned_by="aws") for model in models_list]

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_base = api_base
        self.kwargs = kwargs
        self.client = boto3.client("bedrock-runtime", endpoint_url=api_base, **kwargs)

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        session = boto3.Session()  # type: ignore[attr-defined]
        credentials = session.get_credentials()

        api_key = api_key or os.getenv(self.ENV_API_KEY_NAME)

        if credentials is None and api_key is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

        return api_key

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        logger.warning("AWS Bedrock client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        # create partial function of sync call
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

        if params.stream:
            response_stream = self.client.converse_stream(
                **completion_kwargs,
            )
            stream_generator = response_stream["stream"]
            return (
                self._convert_completion_chunk_response(item, model=params.model_id)
                for item in stream_generator
                if _create_openai_chunk_from_aws_chunk(item, model=params.model_id) is not None
            )
        response = self.client.converse(**completion_kwargs)

        return self._convert_completion_response(response)

    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        logger.warning("AWS Bedrock client does not support async. Calls made with this method will be blocking.")

        loop = asyncio.get_event_loop()

        # create partial function of sync call
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
        input_texts = [inputs] if isinstance(inputs, str) else inputs

        embedding_data = []
        total_tokens = 0

        for index, text in enumerate(input_texts):
            request_body = {"inputText": text}

            if "dimensions" in kwargs:
                request_body["dimensions"] = kwargs["dimensions"]
            if "normalize" in kwargs:
                request_body["normalize"] = kwargs["normalize"]

            response = self.client.invoke_model(modelId=model, body=json.dumps(request_body))

            response_body = json.loads(response["body"].read())

            embedding_data.append({"embedding": response_body["embedding"], "index": index})

            total_tokens += response_body.get("inputTextTokenCount", 0)

        response_data = {"embedding_data": embedding_data, "model": model, "total_tokens": total_tokens}
        return self._convert_embedding_response(response_data)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        client = boto3.client(
            "bedrock",
            endpoint_url=self.api_base,
            **self.kwargs,
        )
        response = client.list_foundation_models(**kwargs)
        return self._convert_list_models_response(response)
