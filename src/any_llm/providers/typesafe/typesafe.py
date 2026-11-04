from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError
from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import (
    AnyLLMError,
    InvalidRequestError,
    MissingApiKeyError,
    ProviderError,
    UnsupportedParameterError,
)
from any_llm.types.decide import DecideResponse, Question
from any_llm.types.model import Model
from any_llm.utils.decide import question_to_dict

MISSING_PACKAGES_ERROR: ImportError | None = None
try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        ListModelsResponse,
        RetryPolicy,
        SystemOneResponse,
        TypeSafeAPIResponseValidationError,
        TypeSafeBadRequestError,
        TypeSafeError,
        TypeSafeUnprocessableEntityError,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typesafe_sdk import QuestionModel as TypeSafeQuestionModel

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.decide import JSONContent


class TypesafeProvider(AnyLLM):
    """Provider for TypeSafe AI.

    TypeSafe exposes System One, which answers named questions (noul, choice, score) about a
    state. It has no chat completions, so use `decide` instead of `completion`.

    A base URL other than the hosted API (for example a local Ollama server) makes the API key
    optional. `list_models` only works against the hosted API, whose response shape the SDK expects.
    """

    PROVIDER_NAME = "typesafe"
    ENV_API_KEY_NAME = "TYPESAFE_API_KEY"
    ENV_API_BASE_NAME = "TYPESAFE_BASE_URL"
    API_BASE = "https://api.typesafe.ai"
    PROVIDER_DOCUMENTATION_URL = "https://docs.typesafe.ai/sdk/python"

    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_MESSAGES = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_DECIDE = True
    TIMEOUT_SUPPORT = "native"
    MAX_RETRIES_SUPPORT = "mapped"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncTypeSafeClient

    @override
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # AnyLLM.__init__ verifies the key before it resolves the base URL, and the key rule depends on it.
        resolved_api_base = self._resolve_api_base(api_base)
        self._self_hosted = resolved_api_base is not None and resolved_api_base.rstrip("/") != self.API_BASE
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    @override
    def _resolve_api_base(self, api_base: str | None = None) -> str | None:
        # A blank value (an empty TYPESAFE_BASE_URL= line) means "unset", as it does for the SDK itself.
        return (super()._resolve_api_base(api_base) or "").strip() or None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # The SDK strips the key, so a whitespace-only key counts as missing here too.
        api_key = (api_key or os.getenv(self.ENV_API_KEY_NAME) or "").strip()
        if api_key:
            return api_key
        if self._self_hosted:
            # A self-hosted server (for example Ollama) needs no key, but the SDK client still requires one.
            return "no-key-required"
        raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "TypeSafe does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "TypeSafe does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: ListModelsResponse) -> Sequence[Model]:
        models = []
        for model in response.models:
            # The SDK types release_date as a plain string, so a malformed date gets this far.
            try:
                released = datetime.fromisoformat(model.release_date)
            except ValueError as error:
                msg = f"TypeSafe returned an invalid release_date {model.release_date!r} for model {model.name!r}"
                raise ProviderError(msg, provider_name=TypesafeProvider.PROVIDER_NAME) from error
            models.append(
                Model(
                    id=model.name,
                    created=int(released.replace(tzinfo=UTC).timestamp()),
                    object="model",
                    owned_by=TypesafeProvider.PROVIDER_NAME,
                )
            )
        return models

    @staticmethod
    @override
    def _classify_exception(exception: Exception) -> type[AnyLLMError] | None:
        if isinstance(exception, TypeSafeAPIResponseValidationError):
            # A 200 the SDK cannot read with its own response models is a server fault.
            return ProviderError
        if isinstance(exception, (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError)):
            # The message lists the caller's question names, which must not pick the class.
            return InvalidRequestError
        if type(exception) is TypeSafeError:
            # The SDK raises the base class for caller mistakes it catches before sending anything.
            return InvalidRequestError
        return None

    @override
    def _apply_max_retries(self, max_retries: int, client_kwargs: dict[str, Any]) -> None:
        if client_kwargs.get("retry") is None:
            # The SDK's default 30 s retry budget would otherwise cut the retry count short.
            client_kwargs["retry"] = RetryPolicy(max_retries=max_retries, timeout=None)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncTypeSafeClient(api_key=api_key, base_url=api_base, **kwargs)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        return self._convert_list_models_response(await self.client.models.list(**kwargs))

    @staticmethod
    def _convert_question(question: Question) -> TypeSafeQuestionModel:
        """Return the SDK's dict form of a question, with unset optional fields left off.

        The SDK sends a dict as is, so nested values its encoder accepts (dates, UUIDs, Decimals)
        survive, where its question classes would reject or coerce them.
        """
        return cast("TypeSafeQuestionModel", question_to_dict(question))

    @staticmethod
    def _convert_decide_response(response: SystemOneResponse) -> DecideResponse:
        # The SDK drops fields and answer types it does not model, so read the body it was built from.
        body = response.raw_http_response.json()
        try:
            body["id"] = response.request_id
        except TypeSafeError:
            # The SDK raises when the server sent no x-typesafe-request-id header.
            pass
        try:
            return DecideResponse.model_validate(body)
        except ValidationError as error:
            # The validation error can quote the answers, so it stays on the cause.
            msg = "TypeSafe returned a System One response that is not valid"
            raise ProviderError(msg, provider_name=TypesafeProvider.PROVIDER_NAME) from error

    @override
    async def _adecide(
        self, model: str, state: JSONContent, questions: dict[str, Question], **kwargs: Any
    ) -> DecideResponse:
        if "response_model" in kwargs:
            # The SDK would return the caller's model instead of a SystemOneResponse, which the
            # conversion below cannot read.
            parameter_name = "response_model"
            additional_message = "decide always returns a DecideResponse."
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME, additional_message)
        response = await self.client.system_one(
            state,
            {name: self._convert_question(question) for name, question in questions.items()},
            model=model,
            **kwargs,
        )
        return self._convert_decide_response(response)
