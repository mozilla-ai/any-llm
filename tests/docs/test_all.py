import pathlib
import sys
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mktestdocs import check_md_file

from any_llm.constants import LLMProvider


class FakeChatCompletion:
    def __init__(self, content: str = "Hello!") -> None:
        self.model = "gpt-4.1-mini"
        self.choices = [Mock(message=Mock(content=content), delta=Mock(content=content))]


def _build_fake_browser_use_modules() -> dict[str, ModuleType]:
    browser_use_module = ModuleType("browser_use")
    llm_module = ModuleType("browser_use.llm")
    messages_module = ModuleType("browser_use.llm.messages")
    views_module = ModuleType("browser_use.llm.views")
    browser_module = ModuleType("browser_use.browser")
    profile_module = ModuleType("browser_use.browser.profile")
    session_module = ModuleType("browser_use.browser.session")

    setattr(browser_use_module, "__path__", [])
    setattr(llm_module, "__path__", [])
    setattr(browser_module, "__path__", [])

    class BaseMessage:
        def __init__(self, content: object = None, tool_calls: list[object] | None = None) -> None:
            self.content = content
            self.tool_calls = tool_calls or []

    class SystemMessage(BaseMessage):
        pass

    class UserMessage(BaseMessage):
        pass

    class AssistantMessage(BaseMessage):
        pass

    class ChatInvokeUsage:
        def __init__(
            self,
            *,
            prompt_tokens: int,
            prompt_cached_tokens: int | None,
            prompt_cache_creation_tokens: int | None,
            prompt_image_tokens: int | None,
            completion_tokens: int,
            total_tokens: int,
        ) -> None:
            self.prompt_tokens = prompt_tokens
            self.prompt_cached_tokens = prompt_cached_tokens
            self.prompt_cache_creation_tokens = prompt_cache_creation_tokens
            self.prompt_image_tokens = prompt_image_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    class ChatInvokeCompletion:
        def __init__(self, *, completion: object, usage: ChatInvokeUsage | None) -> None:
            self.completion = completion
            self.usage = usage

        @classmethod
        def __class_getitem__(cls, item: object) -> type["ChatInvokeCompletion"]:
            return cls

    class BrowserProfile:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class BrowserSession:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def kill(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class Agent:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def run(self, max_steps: int) -> SimpleNamespace:
            return SimpleNamespace(final_result=lambda: "Browser-use demo result")

    setattr(messages_module, "BaseMessage", BaseMessage)
    setattr(messages_module, "SystemMessage", SystemMessage)
    setattr(messages_module, "UserMessage", UserMessage)
    setattr(messages_module, "AssistantMessage", AssistantMessage)
    setattr(views_module, "ChatInvokeUsage", ChatInvokeUsage)
    setattr(views_module, "ChatInvokeCompletion", ChatInvokeCompletion)
    setattr(profile_module, "BrowserProfile", BrowserProfile)
    setattr(session_module, "BrowserSession", BrowserSession)
    setattr(browser_use_module, "Agent", Agent)
    setattr(browser_use_module, "llm", llm_module)
    setattr(browser_use_module, "browser", browser_module)
    setattr(llm_module, "messages", messages_module)
    setattr(llm_module, "views", views_module)
    setattr(browser_module, "profile", profile_module)
    setattr(browser_module, "session", session_module)

    return {
        "browser_use": browser_use_module,
        "browser_use.llm": llm_module,
        "browser_use.llm.messages": messages_module,
        "browser_use.llm.views": views_module,
        "browser_use.browser": browser_module,
        "browser_use.browser.profile": profile_module,
        "browser_use.browser.session": session_module,
    }


def _build_docs_patches() -> ExitStack:
    stack = ExitStack()

    fake_chat_completion_class = type("ChatCompletion", (), {})
    fake_chat_completion = fake_chat_completion_class()
    fake_chat_completion.model = "gpt-4.1-mini"
    fake_chat_completion.choices = [Mock(message=Mock(content="Hello!"), delta=Mock(content="Hello!"))]

    fake_embedding = Mock()
    fake_embedding.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
    fake_embedding.usage = Mock(total_tokens=10)

    fake_rerank = Mock()
    fake_rerank.id = "rerank-123"
    fake_rerank.results = [Mock(index=0, relevance_score=0.95), Mock(index=2, relevance_score=0.80)]
    fake_rerank.meta = None
    fake_rerank.usage = Mock(total_tokens=50)

    fake_response = Mock(output_text="Paris", id="resp_123")
    fake_message_response = Mock(content=[Mock(text="Hello!")])
    fake_models = [
        Mock(id="gpt-4.1-mini", owned_by="openai", object="model", created=1),
        Mock(id="claude-sonnet-4-20250514", owned_by="anthropic", object="model", created=1),
    ]

    fake_metadata = Mock(
        name="openai",
        env_key="OPENAI_API_KEY",
        env_api_base=None,
        doc_url="https://example.com",
        streaming=True,
        reasoning=True,
        completion=True,
        embedding=True,
        responses=True,
        image=False,
        pdf=False,
        class_name="OpenAIProvider",
        list_models=True,
        messages=True,
        batch_completion=True,
    )

    fake_batch = Mock(
        id="batch_123",
        status="completed",
        output_file_id="file_123",
        request_counts=Mock(total=2, completed=2, failed=0),
    )

    provider_mock = Mock()
    provider_mock.completion.return_value = fake_chat_completion
    provider_mock.responses.return_value = fake_response
    provider_mock.messages.return_value = fake_message_response
    provider_mock.list_models.return_value = fake_models
    provider_mock.get_provider_metadata.return_value = fake_metadata

    def mock_completion_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("stream", False):
            return [Mock(choices=[Mock(delta=Mock(content="Hello!"))])]
        return fake_chat_completion

    def mock_responses_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("stream", False):
            return [Mock(type="response.created"), Mock(type="response.completed")]
        return fake_response

    def mock_messages_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("stream", False):
            return [Mock(type="message_start"), Mock(type="message_stop")]
        return fake_message_response

    stack.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": "fake", "ANTHROPIC_API_KEY": "fake"}, clear=False))
    stack.enter_context(patch("getpass.getpass", return_value=""))
    stack.enter_context(patch("any_llm.completion", side_effect=mock_completion_side_effect))
    stack.enter_context(patch("any_llm.acompletion", new=AsyncMock(return_value=fake_chat_completion)))
    stack.enter_context(patch("any_llm.embedding", return_value=fake_embedding))
    stack.enter_context(patch("any_llm.aembedding", new=AsyncMock(return_value=fake_embedding)))
    stack.enter_context(patch("any_llm.responses", side_effect=mock_responses_side_effect))
    stack.enter_context(patch("any_llm.aresponses", new=AsyncMock(return_value=fake_response)))
    stack.enter_context(patch("any_llm.api.messages", side_effect=mock_messages_side_effect))
    stack.enter_context(patch("any_llm.api.amessages", new=AsyncMock(return_value=fake_message_response)))
    stack.enter_context(patch("any_llm.list_models", return_value=fake_models))
    stack.enter_context(patch("any_llm.alist_models", new=AsyncMock(return_value=fake_models)))
    stack.enter_context(patch("any_llm.rerank", return_value=fake_rerank))
    stack.enter_context(patch("any_llm.arerank", new=AsyncMock(return_value=fake_rerank)))
    stack.enter_context(patch("any_llm.create_batch", return_value=fake_batch))
    stack.enter_context(patch("any_llm.retrieve_batch", return_value=fake_batch))
    stack.enter_context(patch("any_llm.list_batches", return_value=[fake_batch]))
    stack.enter_context(patch("any_llm.any_llm.AnyLLM.create", return_value=provider_mock))
    stack.enter_context(patch("any_llm.AnyLLM.get_all_provider_metadata", return_value=[fake_metadata]))
    stack.enter_context(patch("any_llm.types.completion.ChatCompletion", fake_chat_completion_class))
    stack.enter_context(patch.dict(sys.modules, _build_fake_browser_use_modules()))

    return stack


@pytest.mark.parametrize(
    "doc_file",
    list(pathlib.Path("docs").glob("**/*.md")),
    ids=str,
)
def test_all_docs(doc_file: pathlib.Path) -> None:
    if doc_file.name == "index.md":
        mock_response = Mock(choices=[Mock(message=Mock(content="Hello!"))])
        with patch("any_llm.completion", return_value=mock_response):
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
    elif doc_file.name == "quickstart.md" and "gateway" in doc_file.parts:
        mock_response = Mock(choices=[Mock(message=Mock(content="Hello!"))])
        with (
            patch("any_llm.completion", return_value=mock_response),
            patch.dict("os.environ", {"GATEWAY_MASTER_KEY": "fake-gateway-key"}),
        ):
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
    elif doc_file.name == "quickstart.md":
        mock_provider = Mock()
        mock_provider.completion.return_value = Mock(choices=[Mock(message=Mock(content="Hello!"))])
        mock_provider.get_provider_metadata.return_value = Mock(streaming=True, completion=True)

        mock_embedding_result = Mock()
        mock_embedding_result.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_embedding_result.usage = Mock(total_tokens=10)

        mock_moderation_result = Mock()
        mock_moderation_result.results = [
            Mock(flagged=True, categories={"violence": True}, category_scores={"violence": 0.99})
        ]

        def mock_completion_side_effect(*args, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs.get("stream", False):
                return [Mock(choices=[Mock(delta=Mock(content="Hello!"))])]
            return Mock(choices=[Mock(message=Mock(content="Hello!"), delta=Mock(content="Hello!"))])

        with (
            patch("any_llm.any_llm.AnyLLM.split_model_provider") as mock_split,
            patch("any_llm.any_llm.AnyLLM.create") as mock_create,
            patch("any_llm.completion") as mock_completion,
            patch("any_llm.embedding") as mock_embedding,
            patch("any_llm.moderation") as mock_moderation,
            patch("os.environ.get") as mock_env_get,
        ):
            mock_split.return_value = (LLMProvider.OPENAI, "gpt-5")
            mock_create.return_value = mock_provider
            mock_completion.side_effect = mock_completion_side_effect
            mock_embedding.return_value = mock_embedding_result
            mock_moderation.return_value = mock_moderation_result
            mock_env_get.return_value = "fake-api-key"  # Mock API key environment variables
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
    else:
        with _build_docs_patches():
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
