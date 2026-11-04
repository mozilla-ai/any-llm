"""DeepInfra: OpenAI-compatible chat and embeddings, plus video through the native inference API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from any_llm.exceptions import InvalidRequestError
from any_llm.providers.openai.base import BaseOpenAIProvider

from .video import VideoSchema, convert_inference_response, convert_video_params

if TYPE_CHECKING:
    from openai._types import RequestOptions

    from any_llm.types.video import VideoGenerationParams, VideoJob

# A Seedance render takes minutes and DeepInfra holds the request open for all of it, so the
# video call needs far more than the OpenAI client's default per-request timeout.
_VIDEO_REQUEST_TIMEOUT_SECONDS = 900.0
# Finished jobs live in memory until retrieve_video reads them, so a long-lived process keeps
# only the newest ones.
_VIDEO_JOBS_KEPT = 1000


class DeepinfraProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepinfra.com/v1/openai"
    ENV_API_KEY_NAME = "DEEPINFRA_API_KEY"
    ENV_API_BASE_NAME = "DEEPINFRA_API_BASE"
    PROVIDER_NAME = "deepinfra"
    PROVIDER_DOCUMENTATION_URL = "https://deepinfra.com/docs/openai_api"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_MODERATION = True
    SUPPORTS_VIDEO_GENERATION = True

    _video_jobs: ClassVar[dict[str, VideoJob]] = {}
    """The newest finished video jobs by id. DeepInfra renders inside the request and has no
    endpoint to look a request up afterwards, so this is the only place ``retrieve_video`` can read from."""

    _video_schemas: ClassVar[dict[tuple[str, str], VideoSchema]] = {}
    """Each video model's input schema by API root and model, fetched once per process from ``GET /models/{model}``."""

    def _api_root(self) -> str:
        """The host DeepInfra's native endpoints hang off, next to the OpenAI-compatible base."""
        return str(self.client.base_url).rstrip("/").removesuffix("/openai").removesuffix("/v1")

    def _inference_url(self, model: str) -> str:
        """The native endpoint that renders a video for ``model``."""
        return f"{self._api_root()}/v1/inference/{model}"

    def _model_url(self, model: str) -> str:
        """The public model record that holds ``model``'s input schema."""
        return f"{self._api_root()}/models/{model}"

    async def _video_schema(self, model: str, *, options: RequestOptions) -> VideoSchema:
        """The input schema of ``model``, from the cache or from one ``GET /models/{model}``."""
        key = (self._api_root(), model)
        if key not in self._video_schemas:
            record = await self.client.get(self._model_url(model), cast_to=dict[str, Any], options=options)
            self._video_schemas[key] = VideoSchema.from_model_record(model, record)
        return self._video_schemas[key]

    @override
    async def _avideo_generation(self, params: VideoGenerationParams, **kwargs: Any) -> VideoJob:
        # The OpenAI SDK has no inference resource, so post through its client to keep the
        # provider's auth and error mapping. No retries: a retried render is billed twice.
        timeout = kwargs.pop("timeout", _VIDEO_REQUEST_TIMEOUT_SECONDS)
        options: RequestOptions = {"max_retries": 0, "timeout": timeout}
        if (extra_headers := kwargs.pop("extra_headers", None)) is not None:
            options["headers"] = extra_headers
        extra_body = kwargs.pop("extra_body", None) or {}
        # The schema fetch shares the caller's deadline but keeps the client's retries: it is a cheap, idempotent GET.
        schema = await self._video_schema(params.model_id, options={"timeout": timeout})
        body = {**convert_video_params(params, schema, self.PROVIDER_NAME, **kwargs), **extra_body}
        if "webhook" in body:
            msg = (
                "DeepInfra video jobs cannot take a webhook: the reply would be a queued job that "
                "retrieve_video has no way to read back."
            )
            raise InvalidRequestError(msg)
        response = await self.client.post(
            self._inference_url(params.model_id),
            body=body,
            cast_to=dict[str, Any],
            options=options,
        )
        job = convert_inference_response(response, model=params.model_id, api_base=str(self.client.base_url))
        self._video_jobs[job.id] = job
        while len(self._video_jobs) > _VIDEO_JOBS_KEPT:
            del self._video_jobs[next(iter(self._video_jobs))]
        return job

    @override
    async def _aretrieve_video(self, video_id: str, **kwargs: Any) -> VideoJob:
        job = self._video_jobs.get(video_id)
        if job is None:
            msg = (
                f"DeepInfra has no endpoint to look up a video job, so only the newest {_VIDEO_JOBS_KEPT} jobs "
                f"started in this process can be retrieved, and {video_id!r} is not one of them."
            )
            raise InvalidRequestError(msg)
        return job
