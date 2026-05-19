# mypy: disable-error-code="no-untyped-call"
from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError, MissingApiKeyError
from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    import boto3

    from .utils import (
        _convert_bedrock_batch_output_to_result,
        _convert_bedrock_job_to_openai_batch,
        _convert_params,
        _convert_response,
        _create_openai_chunk_from_aws_chunk,
        _create_openai_embedding_response_from_aws,
        _parse_s3_uri,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from any_llm.types.batch import Batch, BatchResult


class BedrockProvider(AnyLLM):
    """AWS Bedrock Provider using boto3."""

    PROVIDER_NAME = "bedrock"
    ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
    ENV_API_BASE_NAME = "AWS_ENDPOINT_URL_BEDROCK_RUNTIME"
    PROVIDER_DOCUMENTATION_URL = "https://aws.amazon.com/bedrock/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    SUPPORTS_RERANK = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self._custom_client: Any = kwargs.pop("client", None)
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for AWS API."""
        return _convert_params(params, kwargs)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert AWS Bedrock response to OpenAI format."""
        return _convert_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert AWS Bedrock chunk response to OpenAI format."""
        model = kwargs.get("model", "")
        tool_index_map = kwargs.get("tool_index_map")
        chunk = _create_openai_chunk_from_aws_chunk(response, model, tool_index_map)
        if chunk is None:
            msg = "Failed to convert AWS chunk to OpenAI format"
            raise ValueError(msg)
        return chunk

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for AWS Bedrock."""
        # For bedrock, we don't need to convert the params, just pass them through
        return kwargs

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert AWS Bedrock embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_aws(
            response["embedding_data"], response["model"], response["total_tokens"]
        )

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert AWS Bedrock list models response to OpenAI format."""
        models_list = response.get("modelSummaries", [])
        # AWS doesn't provide a creation date for models
        # AWS doesn't provide typing, but per https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
        # the modelId is a string and will not be None
        return [Model(id=model["modelId"], object="model", created=0, owned_by="aws") for model in models_list]

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_base = api_base
        self.kwargs = kwargs
        if self._custom_client is not None:
            self.client = self._custom_client
        else:
            self.client = boto3.client("bedrock-runtime", endpoint_url=api_base, **kwargs)

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Skip credential verification when a pre-built client is provided
        if self._custom_client is not None:
            return api_key

        session = boto3.Session()  # type: ignore[attr-defined]
        credentials = session.get_credentials()

        api_key = api_key or os.getenv(self.ENV_API_KEY_NAME)

        if credentials is None and api_key is None:
            raise MissingApiKeyError(provider_name=self.PROVIDER_NAME, env_var_name=self.ENV_API_KEY_NAME)

        return api_key

    @override
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

            def _stream_with_state() -> Iterator[ChatCompletionChunk]:
                tool_index_map: dict[int, int] = {}
                for item in stream_generator:
                    chunk = _create_openai_chunk_from_aws_chunk(item, params.model_id, tool_index_map)
                    if chunk is not None:
                        yield chunk

            return _stream_with_state()
        response = self.client.converse(**completion_kwargs)

        return self._convert_completion_response(response)

    @override
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

    @override
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

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        client = self._get_bedrock_control_client()
        response = client.list_foundation_models(**kwargs)
        return self._convert_list_models_response(response)

    def _get_bedrock_control_client(self) -> Any:
        """Return a ``bedrock`` control-plane client for batch and model management operations."""
        return boto3.client("bedrock", endpoint_url=self.api_base, **self.kwargs)

    def _get_s3_client(self) -> Any:
        """Return an ``s3`` client for reading batch output files."""
        return boto3.client("s3", **self.kwargs)

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch inference job on AWS Bedrock.

        ``input_file_path`` must be an S3 URI (e.g. ``s3://bucket/input.jsonl``).
        The JSONL file should use the Bedrock Converse API request format.

        Required keyword arguments:
            role_arn: IAM role ARN that grants Bedrock permission to run the job.
            output_s3_uri: S3 URI where Bedrock will write the results.

        Optional keyword arguments:
            job_name: Human-readable name for the job (auto-generated if omitted).
            model_id: The Bedrock model ID to use. Required by the Bedrock API.
        """
        role_arn: str | None = kwargs.pop("role_arn", None)
        output_s3_uri: str | None = kwargs.pop("output_s3_uri", None)
        job_name: str | None = kwargs.pop("job_name", None)
        model_id: str | None = kwargs.pop("model_id", None)

        if not role_arn:
            msg = "Bedrock batch requires 'role_arn' to be passed as a keyword argument."
            raise InvalidRequestError(msg, provider_name=self.PROVIDER_NAME)
        if not output_s3_uri:
            msg = "Bedrock batch requires 'output_s3_uri' to be passed as a keyword argument."
            raise InvalidRequestError(msg, provider_name=self.PROVIDER_NAME)
        if not model_id:
            msg = "Bedrock batch requires 'model_id' to be passed as a keyword argument."
            raise InvalidRequestError(msg, provider_name=self.PROVIDER_NAME)

        if job_name is None:
            import uuid

            job_name = f"any-llm-batch-{uuid.uuid4().hex[:8]}"

        create_kwargs: dict[str, Any] = {
            "jobName": job_name,
            "roleArn": role_arn,
            "modelId": model_id,
            "inputDataConfig": {"s3InputDataConfig": {"s3Uri": input_file_path}},
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
            "modelInvocationType": "Converse",
        }

        if metadata:
            create_kwargs["tags"] = [{"key": k, "value": v} for k, v in metadata.items()]

        bedrock_control = self._get_bedrock_control_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, functools.partial(bedrock_control.create_model_invocation_job, **create_kwargs)
        )

        job_arn = response["jobArn"]
        job_response = await loop.run_in_executor(
            None, functools.partial(bedrock_control.get_model_invocation_job, jobIdentifier=job_arn)
        )
        return _convert_bedrock_job_to_openai_batch(job_response)

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch inference job from AWS Bedrock."""
        bedrock_control = self._get_bedrock_control_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, functools.partial(bedrock_control.get_model_invocation_job, jobIdentifier=batch_id)
        )
        return _convert_bedrock_job_to_openai_batch(response)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Stop a batch inference job on AWS Bedrock."""
        bedrock_control = self._get_bedrock_control_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, functools.partial(bedrock_control.stop_model_invocation_job, jobIdentifier=batch_id)
        )
        response = await loop.run_in_executor(
            None, functools.partial(bedrock_control.get_model_invocation_job, jobIdentifier=batch_id)
        )
        return _convert_bedrock_job_to_openai_batch(response)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch inference jobs on AWS Bedrock."""
        list_kwargs: dict[str, Any] = {}
        if after:
            list_kwargs["nextToken"] = after
        if limit:
            list_kwargs["maxResults"] = limit

        bedrock_control = self._get_bedrock_control_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, functools.partial(bedrock_control.list_model_invocation_jobs, **list_kwargs)
        )

        summaries = response.get("invocationJobSummaries", [])
        return [_convert_bedrock_job_to_openai_batch(s) for s in summaries]

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        """Retrieve the results of a completed batch inference job from AWS Bedrock.

        Reads the output JSONL file from the S3 location specified in the job
        configuration.
        """
        bedrock_control = self._get_bedrock_control_client()
        loop = asyncio.get_event_loop()
        job = await loop.run_in_executor(
            None, functools.partial(bedrock_control.get_model_invocation_job, jobIdentifier=batch_id)
        )

        status = job.get("status", "")
        if status not in ("Completed", "PartiallyCompleted"):
            openai_batch = _convert_bedrock_job_to_openai_batch(job)
            raise BatchNotCompleteError(
                batch_id=batch_id,
                status=openai_batch.status or "unknown",
                provider_name=self.PROVIDER_NAME,
            )

        output_s3_uri = job.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri", "")
        input_s3_uri = job.get("inputDataConfig", {}).get("s3InputDataConfig", {}).get("s3Uri", "")

        _, input_key = _parse_s3_uri(input_s3_uri)
        input_filename = input_key.rsplit("/", maxsplit=1)[-1]
        output_bucket, output_key_prefix = _parse_s3_uri(output_s3_uri)

        job_arn = job.get("jobArn", "")
        job_id = job_arn.rsplit("/", maxsplit=1)[-1] if "/" in job_arn else job_arn
        output_key = f"{output_key_prefix}{job_id}/{input_filename}.out"

        s3_client = self._get_s3_client()
        s3_response = await loop.run_in_executor(
            None, functools.partial(s3_client.get_object, Bucket=output_bucket, Key=output_key)
        )
        body_bytes: bytes = await loop.run_in_executor(None, s3_response["Body"].read)
        output_lines = body_bytes.decode("utf-8").strip().split("\n")
        return _convert_bedrock_batch_output_to_result(output_lines)
