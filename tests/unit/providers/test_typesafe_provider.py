import warnings
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    ListModelsResponse,
    ModelMetadata,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)
from typesafe_sdk import Noul as TypeSafeNoul

from any_llm import AnyLLM
from any_llm.exceptions import (
    InsufficientFundsError,
    InvalidRequestError,
    MissingApiKeyError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.typesafe import TypesafeProvider
from any_llm.types.decide import (
    Choice,
    ChoiceAnswer,
    DecideResponse,
    DecideUsage,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
    UnknownAnswer,
)

_MODEL = "jev-latest"
_QUESTIONS = {"billing": Noul(instructions="Is this about billing?")}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "ANY_LLM_UNIFIED_EXCEPTIONS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client_cls() -> Iterator[MagicMock]:
    with patch("any_llm.providers.typesafe.typesafe.AsyncTypeSafeClient") as mock:
        yield mock


_SYSTEM_ONE_BODY: dict[str, Any] = {
    "model": "jev-1",
    "usage": {"input_tokens": 12, "output_tokens": 3},
    "answers": {
        "billing": {"type": "noul", "noul": 0.9},
        "tone": {"type": "choice", "choice": "angry", "confidence": 0.8, "probabilities": {"angry": 0.8}},
        "risk": {
            "type": "score",
            "score": 0.4,
            "confidence": 0.6,
            "legend": {0: "low", 1: "high"},
            "probabilities": {0: 0.6, 1: 0.4},
        },
    },
}


@pytest.fixture
def provider() -> Iterator[TypesafeProvider]:
    """A provider whose client answers `_SYSTEM_ONE_BODY` unless a test overrides it."""
    provider = TypesafeProvider(api_key="test_key")
    with (
        patch.object(provider.client, "system_one", new_callable=AsyncMock) as system_one,
        patch.object(provider.client.models, "list", new_callable=AsyncMock),
    ):
        system_one.return_value = SystemOneResponse.from_http_response(httpx2.Response(200, json=_SYSTEM_ONE_BODY))
        yield provider


def _models_response() -> ListModelsResponse:
    return ListModelsResponse(models=(ModelMetadata(name="jev-latest", description="Jev", release_date="2026-01-01"),))


def test_provider_is_resolved_by_name() -> None:
    assert AnyLLM.get_provider_class("typesafe") is TypesafeProvider


def test_metadata_advertises_decide_but_not_messages() -> None:
    metadata = TypesafeProvider.get_provider_metadata()
    assert metadata.decide
    assert not metadata.messages
    assert metadata.list_models


@pytest.mark.parametrize("api_base", [None, "", "   ", "https://api.typesafe.ai", "https://api.typesafe.ai/"])
def test_missing_api_key_raises_for_the_hosted_api(api_base: str | None) -> None:
    with pytest.raises(MissingApiKeyError):
        TypesafeProvider(api_base=api_base)


def test_missing_api_key_raises_with_blank_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_BASE_URL", "")
    with pytest.raises(MissingApiKeyError):
        TypesafeProvider()


def test_whitespace_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "  ")
    with pytest.raises(MissingApiKeyError):
        TypesafeProvider()


def test_blank_base_url_env_falls_back_to_the_sdk_default(
    monkeypatch: pytest.MonkeyPatch, client_cls: MagicMock
) -> None:
    monkeypatch.setenv("TYPESAFE_BASE_URL", " ")
    TypesafeProvider(api_key="test_key")
    client_cls.assert_called_once_with(api_key="test_key", base_url=None)


def test_client_receives_api_base(client_cls: MagicMock) -> None:
    TypesafeProvider(api_key="test_key", api_base="http://localhost:1234")
    client_cls.assert_called_once_with(api_key="test_key", base_url="http://localhost:1234")


def test_api_key_optional_with_custom_api_base(client_cls: MagicMock) -> None:
    TypesafeProvider(api_base="http://localhost:11434")
    client_cls.assert_called_once_with(api_key="no-key-required", base_url="http://localhost:11434")


@pytest.mark.parametrize(("env_key", "expected"), [("env_key", "env_key"), ("  ", "no-key-required")])
def test_self_hosted_key_comes_from_env_when_set(
    monkeypatch: pytest.MonkeyPatch, client_cls: MagicMock, env_key: str, expected: str
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", env_key)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:11434")
    TypesafeProvider()
    client_cls.assert_called_once_with(api_key=expected, base_url="http://localhost:11434")


def test_max_retries_maps_to_retry_policy_without_time_budget(client_cls: MagicMock) -> None:
    TypesafeProvider(api_key="test_key", max_retries=3)
    assert client_cls.call_args.kwargs["retry"] == RetryPolicy(max_retries=3, timeout=None)


def test_explicit_retry_policy_wins_over_max_retries(client_cls: MagicMock) -> None:
    policy = RetryPolicy(max_retries=7)
    TypesafeProvider(api_key="test_key", max_retries=3, retry=policy)
    assert client_cls.call_args.kwargs["retry"] is policy


def test_explicit_retry_none_does_not_drop_max_retries(client_cls: MagicMock) -> None:
    TypesafeProvider(api_key="test_key", max_retries=0, retry=None)
    assert client_cls.call_args.kwargs["retry"] == RetryPolicy(max_retries=0, timeout=None)


@pytest.mark.asyncio
async def test_adecide_sends_question_dicts_and_converts_the_response(provider: TypesafeProvider) -> None:
    questions: dict[str, Question] = {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(criteria={"calm": None, "angry": "Hostile"}),
        "risk": Score(instructions="How risky?", criteria=["low", "high"]),
    }

    result = await provider.adecide(_MODEL, "I was charged twice.", questions, timeout=5)

    provider.client.system_one.assert_awaited_once_with(  # type: ignore[attr-defined]
        "I was charged twice.",
        {
            "billing": {"type": "noul", "instructions": "Is this about billing?"},
            "tone": {"type": "choice", "criteria": {"calm": None, "angry": "Hostile"}},
            "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "high"]},
        },
        model=_MODEL,
        timeout=5,
    )
    assert result == DecideResponse(
        id=None,
        model="jev-1",
        answers={
            "billing": NoulAnswer(noul=0.9),
            "tone": ChoiceAnswer(choice="angry", confidence=0.8, probabilities={"angry": 0.8}),
            "risk": ScoreAnswer(
                score=0.4, confidence=0.6, legend={0: "low", 1: "high"}, probabilities={0: 0.6, 1: 0.4}
            ),
        },
        usage=DecideUsage(input_tokens=12, output_tokens=3),
    )


@pytest.mark.parametrize(
    "criteria",
    [NoulCriteria(true="Advertising"), NoulCriteria(true="Advertising", false=None), {"true": "Advertising"}],
)
@pytest.mark.asyncio
async def test_adecide_leaves_undescribed_noul_outcomes_off(
    provider: TypesafeProvider, criteria: NoulCriteria | dict[str, str]
) -> None:
    await provider.adecide(_MODEL, "state", {"spam": {"type": "noul", "criteria": criteria}})

    sent = provider.client.system_one.await_args.args[1]  # type: ignore[attr-defined]
    assert sent == {"spam": {"type": "noul", "criteria": {"true": "Advertising"}}}


def test_noul_without_criteria_omits_them() -> None:
    assert TypesafeProvider._convert_question(Noul(criteria=None)) == {"type": "noul"}


@pytest.mark.asyncio
async def test_adecide_sends_nested_values_the_sdk_encodes(provider: TypesafeProvider) -> None:
    instructions = {"since": date(2026, 1, 1), "threshold": Decimal("0.5")}

    await provider.adecide(_MODEL, "state", {"recent": Noul(instructions=instructions)})

    sent = provider.client.system_one.await_args.args[1]  # type: ignore[attr-defined]
    assert sent == {"recent": {"type": "noul", "instructions": instructions}}


@pytest.mark.asyncio
async def test_adecide_accepts_sdk_question_objects(provider: TypesafeProvider) -> None:
    question = TypeSafeNoul(instructions="Is this about billing?", criteria={"true": "Billing"})

    await provider.adecide(_MODEL, "state", {"billing": question})  # type: ignore[dict-item]

    sent = provider.client.system_one.await_args.args[1]  # type: ignore[attr-defined]
    assert sent == {
        "billing": {"type": "noul", "instructions": "Is this about billing?", "criteria": {"true": "Billing"}}
    }


@pytest.mark.asyncio
async def test_adecide_refuses_images(provider: TypesafeProvider) -> None:
    with pytest.raises(UnsupportedParameterError, match="images"):
        await provider.adecide(_MODEL, "state", _QUESTIONS, images=["data:image/png;base64,"])

    provider.client.system_one.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_adecide_rejects_response_model(provider: TypesafeProvider) -> None:
    with pytest.raises(UnsupportedParameterError, match="response_model") as exc_info:
        await provider.adecide(_MODEL, "state", _QUESTIONS, response_model=SystemOneResponse)

    assert exc_info.value.parameter_name == "response_model"
    provider.client.system_one.assert_not_awaited()  # type: ignore[attr-defined]


def test_decide_runs_adecide_synchronously(provider: TypesafeProvider) -> None:
    result = provider.decide(_MODEL, "I was charged twice.", _QUESTIONS)

    assert result.nouls == {"billing": NoulAnswer(noul=0.9)}


@pytest.mark.asyncio
async def test_decide_inside_running_loop_raises_without_leaking_a_coroutine(provider: TypesafeProvider) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeError, match="Cannot use the `sync` API in an `async` context"):
            provider.decide(_MODEL, "state", _QUESTIONS)


@pytest.mark.asyncio
async def test_alist_models_converts_response(provider: TypesafeProvider) -> None:
    provider.client.models.list.return_value = _models_response()  # type: ignore[attr-defined]

    models = await provider.alist_models()

    assert [m.id for m in models] == ["jev-latest"]
    assert models[0].owned_by == "typesafe"
    assert models[0].created == 1767225600


@pytest.mark.parametrize(
    "call",
    [
        lambda: TypesafeProvider._convert_completion_params(None),  # type: ignore[arg-type]
        lambda: TypesafeProvider._convert_completion_response(None),
        lambda: TypesafeProvider._convert_completion_chunk_response(None),
        lambda: TypesafeProvider._convert_embedding_params(None),
        lambda: TypesafeProvider._convert_embedding_response(None),
    ],
)
def test_unsupported_conversions_raise(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(NotImplementedError):
        call()


@pytest.mark.asyncio
async def test_unified_rate_limit_carries_status_and_retry_after(provider: TypesafeProvider) -> None:
    provider._unified_exceptions = True
    error = TypeSafeRateLimitError(429, {"error": "slow down"}, httpx2.Headers({"retry-after": "7"}))
    provider.client.system_one.side_effect = error  # type: ignore[attr-defined]

    with pytest.raises(RateLimitError) as exc_info:
        await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == "7"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(402, InsufficientFundsError), (400, InvalidRequestError)],
)
@pytest.mark.asyncio
async def test_unified_errors_classify_by_status_not_hostname(
    provider: TypesafeProvider, status: int, expected: type[Exception]
) -> None:
    provider._unified_exceptions = True
    endpoint = "POST http://safety-api.acme.internal/v1/systemone"
    error = TypeSafeAPIError(status, None, httpx2.Headers(), endpoint=endpoint)
    provider.client.system_one.side_effect = error  # type: ignore[attr-defined]

    with pytest.raises(expected):
        await provider.adecide(_MODEL, "state", _QUESTIONS)


@pytest.mark.parametrize(
    "message",
    [
        "At least one question is required.",
        "The request body could not be encoded as JSON",
    ],
)
@pytest.mark.asyncio
async def test_unified_sdk_local_errors_are_invalid_requests(provider: TypesafeProvider, message: str) -> None:
    provider._unified_exceptions = True
    provider.client.system_one.side_effect = TypeSafeError(message)  # type: ignore[attr-defined]

    with pytest.raises(InvalidRequestError) as exc_info:
        await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert exc_info.value.status_code is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"api_key": "a b"}, "API key must contain only printable ASCII"),
        ({"api_key": "test_key", "timeout": -1}, "timeout must be a positive"),
    ],
)
def test_unified_client_construction_errors_are_invalid_requests(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidRequestError, match=message) as exc_info:
        AnyLLM.create("typesafe", unified_exceptions=True, **kwargs)

    assert type(exc_info.value.original_exception) is TypeSafeError


def _provider_answering(body: dict[str, Any], headers: dict[str, str] | None = None) -> TypesafeProvider:
    transport = httpx2.MockTransport(lambda _request: httpx2.Response(200, json=body, headers=headers))
    return TypesafeProvider(
        api_key="test_key", unified_exceptions=True, http_client=httpx2.AsyncClient(transport=transport)
    )


@pytest.mark.parametrize(
    ("headers", "expected_id"),
    [({"x-typesafe-request-id": "req_123"}, "req_123"), (None, None)],
)
@pytest.mark.asyncio
async def test_adecide_carries_the_request_id(headers: dict[str, str] | None, expected_id: str | None) -> None:
    provider = _provider_answering(_SYSTEM_ONE_BODY, headers)

    result = await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert result.id == expected_id


@pytest.mark.asyncio
async def test_adecide_keeps_fields_and_answer_types_the_sdk_drops() -> None:
    body = {
        **_SYSTEM_ONE_BODY,
        "provider": "typesafe",
        "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0.0001, "cached_tokens": 4},
        "answers": {
            **_SYSTEM_ONE_BODY["answers"],
            "logo": {"type": "box", "box": {"top": 1, "bottom": 2, "left": 3, "right": 4}},
        },
    }
    provider = _provider_answering(body, {"x-typesafe-request-id": "req_123"})

    result = await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert result.id == "req_123"
    assert result.model_extra == {"provider": "typesafe"}
    assert result.usage is not None
    assert result.usage.cost == 0.0001
    assert result.usage.model_extra == {"cached_tokens": 4}
    assert result.answers["logo"] == UnknownAnswer.model_validate(body["answers"]["logo"])
    assert result.nouls["billing"].noul == 0.9


@pytest.mark.asyncio
async def test_adecide_falls_back_to_the_body_id_without_a_request_id_header() -> None:
    provider = _provider_answering({**_SYSTEM_ONE_BODY, "id": "body_123"})

    result = await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert result.id == "body_123"


def test_a_body_the_sdk_reads_but_decide_cannot_is_a_provider_error() -> None:
    response = MagicMock()
    response.raw_http_response.json.return_value = {"answers": {"billing": {"type": 1}}}

    with pytest.raises(ProviderError, match="TypeSafe returned a System One response that is not valid") as exc_info:
        TypesafeProvider._convert_decide_response(response)

    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.asyncio
async def test_unified_unreadable_response_is_a_provider_error() -> None:
    provider = _provider_answering({"model": "jev", "usage": None, "answers": {}})

    with pytest.raises(ProviderError) as exc_info:
        await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert type(exc_info.value) is ProviderError
    assert exc_info.value.status_code == 200


@pytest.mark.parametrize("question", ["auth_check", "rate_limit_risk", "safety_concern"])
@pytest.mark.parametrize("error_class", [TypeSafeBadRequestError, TypeSafeUnprocessableEntityError])
@pytest.mark.asyncio
async def test_unified_rejected_requests_ignore_question_names(
    provider: TypesafeProvider, question: str, error_class: type[TypeSafeAPIError]
) -> None:
    provider._unified_exceptions = True
    body = {"detail": [{"loc": ["body", "questions", question, "criteria"], "msg": "Field required"}]}
    status = 400 if error_class is TypeSafeBadRequestError else 422
    provider.client.system_one.side_effect = error_class(status, body, httpx2.Headers())  # type: ignore[attr-defined]

    with pytest.raises(InvalidRequestError) as exc_info:
        await provider.adecide(_MODEL, "state", _QUESTIONS)

    assert type(exc_info.value) is InvalidRequestError
    assert exc_info.value.status_code == status


class _FutureSdkError(TypeSafeError):
    """A TypeSafeError subclass a later SDK release could add for a server or transport fault."""


@pytest.mark.parametrize(
    "error",
    [
        TypeSafeAPIError(400, None, httpx2.Headers()),
        TypeSafeAPITimeoutError(5.0),
        _FutureSdkError("stream could not be decoded"),
        RuntimeError("not from the SDK"),
    ],
)
def test_other_errors_use_the_shared_classification(error: Exception) -> None:
    assert TypesafeProvider._classify_exception(error) is None


@pytest.mark.asyncio
async def test_alist_models_rejects_a_malformed_release_date(provider: TypesafeProvider) -> None:
    provider.client.models.list.return_value = ListModelsResponse(  # type: ignore[attr-defined]
        models=(ModelMetadata(name="jev-latest", description="Jev", release_date="2026/01/01"),)
    )

    with pytest.raises(ProviderError, match="release_date") as exc_info:
        await provider.alist_models()

    assert type(exc_info.value) is ProviderError


@pytest.mark.asyncio
async def test_adecide_reraises_sdk_errors_by_default(provider: TypesafeProvider) -> None:
    error = TypeSafeRateLimitError(429, {"error": "slow down"}, httpx2.Headers())
    provider.client.system_one.side_effect = error  # type: ignore[attr-defined]

    with pytest.warns(DeprecationWarning, match="unified any-llm exceptions"), pytest.raises(TypeSafeRateLimitError):
        await provider.adecide(_MODEL, "state", _QUESTIONS)
