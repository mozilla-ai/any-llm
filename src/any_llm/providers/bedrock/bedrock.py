# mypy: disable-error-code="no-untyped-call"
from __future__ import annotations

import asyncio
import functools
import json
import os
import threading
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
    from botocore.config import Config
    from botocore.tokens import ScopedEnvTokenProvider

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

    # boto3's Converse API has no per-request timeout, so it is translated into a per-timeout
    # boto3 client in _convert_completion_params via _client_for_timeout.
    TIMEOUT_SUPPORT = "mapped"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    # Bounds the per-timeout client cache in `_client_for_timeout`, so callers deriving
    # `timeout` from a remaining deadline (a different value on every call) can't grow it
    # (and its underlying connection pools) without limit.
    _MAX_TIMEOUT_CLIENTS = 8

    # Keyword arguments that `boto3.Session(...)` itself accepts. Used to build the session in
    # `_verify_and_set_api_key` with the same explicit credentials the real client is later
    # built with, instead of a bare `boto3.Session()` that only sees ambient credentials.
    _SESSION_CREDENTIAL_KWARGS = (
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "region_name",
        "profile_name",
        "botocore_session",
    )

    @override
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self._custom_client: Any = kwargs.pop("client", None)
        self._custom_control_client: Any = kwargs.pop("control_client", None)
        self._custom_s3_client: Any = kwargs.pop("s3_client", None)
        self._session_credential_kwargs = {k: v for k, v in kwargs.items() if k in self._SESSION_CREDENTIAL_KWARGS}
        # profile_name/botocore_session are boto3.Session-only parameters; Session.client() (used
        # to build the runtime/control-plane/S3 clients below) doesn't accept them, so they must
        # not remain in the kwargs forwarded to those calls. aws_access_key_id/
        # aws_secret_access_key/aws_session_token/region_name are valid on both Session() and
        # Session.client() and are left in kwargs unchanged.
        kwargs.pop("profile_name", None)
        kwargs.pop("botocore_session", None)
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
        self._api_key = api_key
        self._timeout_clients: dict[float, Any] = {}
        self._timeout_clients_lock = threading.Lock()
        if self._custom_client is not None:
            self.client = self._custom_client
            self._boto_session = None
            return
        self._boto_session = self._build_boto_session(api_key)
        self.client = self._boto_session.client(
            "bedrock-runtime", endpoint_url=api_base, **self._bedrock_client_kwargs(kwargs)
        )

    def _build_boto_session(self, api_key: str | None) -> Any:
        """Build a ``boto3.Session`` dedicated to this provider instance.

        Built with the same explicit credential kwargs (``aws_access_key_id``, ``profile_name``,
        etc) used for verification, so e.g. an explicit ``profile_name`` is actually honored by
        the real session and not just by the verification check in ``_verify_and_set_api_key``.

        When ``api_key`` (an AWS Bedrock bearer token) is provided, a token provider scoped to
        this session's own in-memory environ mapping is registered on the underlying botocore
        session. This resolves the token per instance instead of requiring the process-wide
        ``AWS_BEARER_TOKEN_BEDROCK`` env var, which is unsafe to mutate under concurrent,
        multi-tenant use.

        Note: passing the *same* ``botocore_session`` object to multiple ``BedrockProvider``
        instances with different ``api_key`` values is not isolated by this scoping, since
        ``register_component`` on a shared botocore session simply overwrites the previous
        registration (last-registration-wins is botocore's own documented behavior). Give each
        provider instance its own ``botocore_session`` if per-instance bearer tokens must not
        interfere with each other.
        """
        session = boto3.Session(**self._session_credential_kwargs)  # type: ignore[attr-defined]
        if api_key:
            session._session.register_component(
                "token_provider",
                ScopedEnvTokenProvider(session._session, environ={self.ENV_API_KEY_NAME: api_key}),
            )
        return session

    def _bedrock_client_kwargs(self, extra_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge caller-supplied kwargs with a bearer-token-aware ``Config``.

        Forces ``signature_version="bearer"`` when an api_key/bearer token was provided, since
        botocore's own auto-detection of bearer auth only looks at the real process environment,
        not the per-instance token provider set up in ``_build_boto_session``. Does not mutate
        ``extra_kwargs`` (typically ``self.kwargs``), so it can be reused for the control-plane
        client too.
        """
        call_kwargs = dict(extra_kwargs)
        user_config = call_kwargs.pop("config", None)
        merged_config = Config(signature_version="bearer") if self._api_key else None
        if user_config is not None:
            merged_config = user_config.merge(merged_config) if merged_config is not None else user_config
        if merged_config is not None:
            call_kwargs["config"] = merged_config
        return call_kwargs

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Skip credential verification when a pre-built client is provided
        if self._custom_client is not None:
            return api_key

        # Bedrock supports two independent auth mechanisms: a bearer-token API key, or standard
        # AWS credentials (aws_access_key_id/aws_secret_access_key/aws_session_token, IAM roles,
        # SSO, etc). Resolve the bearer token first and short-circuit on it, since it alone is
        # sufficient: this avoids an unnecessary (and potentially slow, or erroring on an invalid
        # explicit profile_name) credential-chain lookup when a bearer token is already provided.
        api_key = api_key or os.getenv(self.ENV_API_KEY_NAME)
        if api_key is not None:
            return api_key

        # A bare boto3.Session() only resolves the *ambient* default credential chain and ignores
        # explicit credential kwargs passed to AnyLLM.create(...), which made this check fail even
        # when valid credentials were explicitly provided (see #1183).
        session = boto3.Session(**self._session_credential_kwargs)  # type: ignore[attr-defined]
        credentials = session.get_credentials()

        if credentials is None:
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
        # boto3's Converse API has no `timeout` parameter; it must be pulled out here so it
        # never reaches `converse`/`converse_stream`, and applied via a client whose connection
        # is configured with that timeout instead (see `_client_for_timeout`).
        timeout = kwargs.pop("timeout", None)
        completion_kwargs = self._convert_completion_params(params, **kwargs)
        client = self._client_for_timeout(timeout)

        if params.stream:
            response_stream = client.converse_stream(
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
        response = client.converse(**completion_kwargs)

        return self._convert_completion_response(response)

    def _client_for_timeout(self, timeout: float | None) -> Any:
        """Return a boto3 client configured for ``timeout``, building/caching one if needed.

        boto3 has no per-request timeout; connect/read timeouts are only configurable at
        client-construction time via ``botocore.config.Config``. When a custom client was
        supplied, any-llm doesn't own its construction, so `timeout` is dropped with a warning
        instead of being silently ignored or crashing.

        ``_completion`` runs on executor threads (via ``_acompletion``), so the cache miss path
        is locked: concurrent calls with the same not-yet-cached timeout must not each build and
        leak their own client. The cache is also bounded, since a caller deriving `timeout` from
        a remaining deadline would otherwise produce a new distinct value (and client) per call.
        """
        if timeout is None or self._boto_session is None:
            if timeout is not None:
                logger.warning(
                    "Bedrock does not support a per-request 'timeout' when a custom client is provided; "
                    "ignoring it. Configure timeouts via botocore.config.Config when constructing your client."
                )
            return self.client

        with self._timeout_clients_lock:
            cached = self._timeout_clients.get(timeout)
            if cached is None:
                timeout_config = Config(connect_timeout=timeout, read_timeout=timeout)
                client_kwargs = self._bedrock_client_kwargs(self.kwargs)
                client_kwargs["config"] = (
                    client_kwargs["config"].merge(timeout_config) if "config" in client_kwargs else timeout_config
                )
                cached = self._boto_session.client("bedrock-runtime", endpoint_url=self.api_base, **client_kwargs)
                if len(self._timeout_clients) >= self._MAX_TIMEOUT_CLIENTS:
                    self._timeout_clients.pop(next(iter(self._timeout_clients)))
                self._timeout_clients[timeout] = cached
            return cached

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
        """Return a ``bedrock`` control-plane client for batch and model management operations.

        Built from the same session (and, when applicable, the same bearer-token credentials)
        as the runtime client, so overrides applied at construction time aren't silently
        bypassed for model listing and batch operations. An explicit ``control_client=``
        constructor kwarg always takes precedence.

        When a custom ``client=`` was supplied (so there's no shared ``_boto_session`` to reuse),
        a fresh session is still built via ``_build_boto_session``, from the caller's explicit
        credential kwargs and with the same bearer-token scoping as the runtime client would have
        had: a bare ``boto3.Session(...)`` here would silently drop an explicit
        ``profile_name``/``botocore_session``, and would leave the bearer ``Config`` forced by
        ``_bedrock_client_kwargs`` (whenever ``api_key`` was supplied) without a matching scoped
        token provider, so the request would fall through to an ambient (or missing) token.
        """
        if self._custom_control_client is not None:
            return self._custom_control_client
        if self._boto_session is not None:
            return self._boto_session.client("bedrock", **self._bedrock_client_kwargs(self.kwargs))
        return self._build_boto_session(self._api_key).client("bedrock", **self._bedrock_client_kwargs(self.kwargs))

    def _get_s3_client(self) -> Any:
        """Return an ``s3`` client for reading batch output files.

        Built from the same session as the runtime client. S3 doesn't support Bedrock's bearer
        token auth, so any ``signature_version="bearer"`` (forced by us elsewhere, or present in
        a caller-supplied ``config=``) is stripped before forwarding, otherwise every S3 call
        would fail to authenticate. An explicit ``s3_client=`` constructor kwarg always takes
        precedence.

        When a custom ``client=`` was supplied (so there's no shared ``_boto_session`` to reuse), a
        fresh session is still built via ``_build_boto_session``; see ``_get_bedrock_control_client``
        for why a bare ``boto3.client(...)``/``boto3.Session(...)`` isn't used instead. The scoped
        token provider registered by ``_build_boto_session`` is harmless here since
        ``_non_bearer_client_kwargs`` already strips ``signature_version="bearer"`` for S3.
        """
        if self._custom_s3_client is not None:
            return self._custom_s3_client
        if self._boto_session is not None:
            return self._boto_session.client("s3", **self._non_bearer_client_kwargs(self.kwargs))
        return self._build_boto_session(self._api_key).client("s3", **self._non_bearer_client_kwargs(self.kwargs))

    @staticmethod
    def _non_bearer_client_kwargs(extra_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Strip a ``signature_version="bearer"`` from a caller-supplied ``config=``, if present."""
        call_kwargs = dict(extra_kwargs)
        config = call_kwargs.get("config")
        if config is not None and getattr(config, "signature_version", None) == "bearer":
            call_kwargs["config"] = config.merge(Config(signature_version=None))
        return call_kwargs

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

        _parse_s3_uri(input_file_path)
        _parse_s3_uri(output_s3_uri)

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
        if not output_key_prefix.endswith("/"):
            output_key_prefix += "/"

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
