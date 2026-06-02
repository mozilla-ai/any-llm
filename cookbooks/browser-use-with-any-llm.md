---
title: Browser-Use with Any-LLM
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-llm/blob/main/docs/cookbooks/browser_use_with_any_llm.ipynb)

This cookbook shows how to run `browser-use` with core `any-llm` locally.

`browser-use` uses its own lightweight async chat-model protocol internally, so the recipe below adds a tiny adapter around `any_llm.acompletion`. That keeps the demo simple:

- keep `browser-use` unchanged
- switch providers by changing one `model` string
- avoid adding `langchain-anyllm` just for this demo


## Install the dependencies

If you are iterating on a local checkout of `any-llm`, replace the PyPI install below with an editable install from your repo path.


```
%pip install "any-llm-sdk[openai]" browser-use nest-asyncio python-dotenv -q
```

> **Note**: If you see a rich version conflict involving bigframes or pyiceberg, you can safely ignore it. Those packages are not used in this notebook.


```
!browser-use install
```

```
import nest_asyncio

# Jupyter already runs an event loop; nest_asyncio lets asyncio.run() and
# top-level await work inside it without raising "event loop already running".
nest_asyncio.apply()
```

## Choose a model

The default below uses OpenAI for the fastest first run, but the adapter works with any provider/model string supported by `any-llm`.


```python
import os
from getpass import getpass

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

MODEL = os.getenv("BROWSER_USE_MODEL", "openai:gpt-4o-mini")
API_BASE = os.getenv("BROWSER_USE_API_BASE")
ENV_KEY_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
}

provider = MODEL.split(":", 1)[0]
env_key = ENV_KEY_BY_PROVIDER.get(provider)

if env_key and env_key not in os.environ:
    os.environ[env_key] = getpass(f"Enter your {env_key}: ")

print(f"Using model: {MODEL}")
if API_BASE:
    print(f"Using custom API base: {API_BASE}")
```

## Create a tiny browser-use adapter

`browser-use` expects a chat model with its own async interface. This notebook adds a small adapter that translates `browser-use` messages and runtime metadata into `any-llm` calls.


```python
import inspect
from typing import Any, TypeVar

from browser_use.llm.messages import AssistantMessage, BaseMessage, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

from any_llm import acompletion

T = TypeVar("T")
ANY_LLM_RUNTIME_KWARG_NAMES = {
    name
    for name, parameter in inspect.signature(acompletion).parameters.items()
    if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    and name not in {"model", "messages", "response_format", "temperature", "provider", "api_key", "api_base"}
}
BROWSER_USE_KWARG_MAP = {"session_id": "session_label"}


def to_any_llm_message(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}

    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content}

        content: list[dict[str, Any]] = []
        for part in message.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text})
            elif part.type == "image_url":
                image_url: dict[str, Any] = {"url": part.image_url.url}
                if part.image_url.detail is not None:
                    image_url["detail"] = part.image_url.detail
                content.append({"type": "image_url", "image_url": image_url})

        return {"role": "user", "content": content}

    if isinstance(message, AssistantMessage):
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return assistant_message

    message_type = type(message)
    error_message = f"Unsupported message type: {message_type!r}"
    raise TypeError(error_message)


def to_usage(response: Any) -> ChatInvokeUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = None
    if prompt_tokens_details is not None:
        cached_tokens = prompt_tokens_details.cached_tokens

    return ChatInvokeUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        prompt_cached_tokens=cached_tokens,
        prompt_cache_creation_tokens=None,
        prompt_image_tokens=None,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


def to_any_llm_runtime_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    filtered_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        mapped_key = BROWSER_USE_KWARG_MAP.get(key, key)
        if mapped_key in ANY_LLM_RUNTIME_KWARG_NAMES:
            filtered_kwargs[mapped_key] = value
    return filtered_kwargs


class BrowserUseAnyLLM:
    # browser-use checks this attribute to skip its own key validation step;
    # credential handling is delegated to any-llm.
    _verified_api_keys: bool = True

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = 0.0,
        **model_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._provider = provider or model.split(":", 1)[0]
        self._completion_kwargs: dict[str, Any] = dict(model_kwargs)
        if provider is not None:
            self._completion_kwargs["provider"] = provider
        if api_key is not None:
            self._completion_kwargs["api_key"] = api_key
        if api_base is not None:
            self._completion_kwargs["api_base"] = api_base

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        runtime_kwargs = to_any_llm_runtime_kwargs(kwargs)
        response = await acompletion(
            model=self.model,
            messages=[to_any_llm_message(message) for message in messages],
            temperature=self.temperature,
            response_format=output_format,
            **self._completion_kwargs,
            **runtime_kwargs,
        )

        usage = to_usage(response)
        if output_format is None:
            completion = response.choices[0].message.content or ""
            return ChatInvokeCompletion(completion=completion, usage=usage)

        parsed = response.choices[0].message.parsed
        if parsed is None:
            msg = "Expected structured browser-use output, but the model returned no parsed payload."
            raise ValueError(msg)

        return ChatInvokeCompletion(completion=parsed, usage=usage)
```

## Smoke test the adapter

It is worth confirming the model wiring before you launch a browser session.


```
llm = BrowserUseAnyLLM(
    model=MODEL,
    api_base=API_BASE,
    temperature=0.0,
)

smoke_result = await llm.ainvoke(
    [
        SystemMessage(content="You are concise."),
        UserMessage(content="Say hello in exactly five words."),
    ]
)

print(smoke_result.completion)
print(smoke_result.usage)
```

## Run a browser-use task

This task is public, easy to narrate, and looks good on video. The browser profile below defaults to `headless=True`, which is much more reliable in notebooks, containers, and remote Linux environments.

Set `BROWSER_USE_HEADLESS=false` only if you are running locally on your own laptop or desktop and want to record the visible browser.


```
from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

HEADLESS = os.getenv("BROWSER_USE_HEADLESS", "true").lower() not in {"0", "false", "no"}

profile = BrowserProfile(
    headless=HEADLESS,
    channel="chromium",
    chromium_sandbox=False,
    enable_default_extensions=False,
    args=["--no-sandbox", "--disable-dev-shm-usage"],
)
browser = BrowserSession(browser_profile=profile)

task = """
Go to https://news.ycombinator.com.
Open the first story.
Return:
1. the story title
2. the destination URL
3. a one-sentence summary
4. one reason this is a good Any-LLM demo
"""

agent = Agent(
    task=task,
    llm=llm,
    browser_session=browser,
    use_vision=False,
    max_actions_per_step=3,
)

try:
    history = await agent.run(max_steps=12)
    print(history.final_result())
finally:
    if hasattr(browser, "kill"):
        await browser.kill()
    elif hasattr(browser, "stop"):
        await browser.stop()
```

## Troubleshooting

If you see `Cannot connect to host 127.0.0.1:<port>`, the local Chromium process died before its CDP endpoint came up.

Try this:

- keep `BROWSER_USE_HEADLESS=true`
- restart the notebook kernel so `browser-use` can launch a fresh browser process
- rerun `browser-use install`
- on Linux or in containers, run `python -m playwright install chromium --with-deps`
- switch to `BROWSER_USE_HEADLESS=false` only on a local desktop or laptop when you want to record video

Also let `Agent.run()` start the browser session for you. Pre-starting and reusing half-failed sessions makes notebook debugging harder.


## Video flow

1. Run the smoke test once to show the adapter is real.
2. Run the Hacker News task with `BROWSER_USE_HEADLESS=false` on your local machine.
3. Change only `MODEL` to another provider/model string you already use with `any-llm`.
4. Recreate `llm = BrowserUseAnyLLM(model=MODEL, api_base=API_BASE, temperature=0.0)`.
5. Re-run the same task and call out that the browser workflow stayed the same while the backend changed in one line.

