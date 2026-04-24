#!/usr/bin/env python3
"""Generate API reference documentation from Python source code.

Extracts docstrings and signatures from the any-llm source and generates
GitBook-compatible markdown pages. Generated files are written to
docs/src/content/docs/api/ and should never be committed to git.

Usage:
    python scripts/generate_api_docs.py
"""

from __future__ import annotations

import inspect
import re
import sys
import textwrap
import typing
from pathlib import Path
from typing import Any, get_type_hints

import any_llm.api as api_module
from any_llm import AnyLLM
from any_llm import exceptions as exc_module
from any_llm.types import completion as completion_types
from any_llm.types import messages as messages_types
from any_llm.types import provider as provider_types
from any_llm.types import responses as responses_types

DOCS_API_DIR = Path(__file__).parent.parent / "docs" / "api"


def _short_name(name: str) -> str:
    """Strip module prefixes to produce a short readable type name."""
    # Remove common module prefixes
    prefixes = [
        "any_llm.types.completion.",
        "any_llm.types.responses.",
        "any_llm.types.messages.",
        "any_llm.types.batch.",
        "any_llm.types.model.",
        "any_llm.types.provider.",
        "any_llm.constants.",
        "any_llm.",
        "collections.abc.",
        "openai.types.responses.",
        "openai.types.chat.",
        "openai.types.",
        "openresponses_types.",
        "typing.",
        "typing_extensions.",
    ]
    for prefix in prefixes:
        name = name.removeprefix(prefix)
    return name


def _clean_qualified_names(text: str) -> str:
    """Remove module prefixes from qualified names in generated text."""
    replacements = [
        ("any_llm.types.completion.", ""),
        ("any_llm.types.responses.", ""),
        ("any_llm.types.messages.", ""),
        ("any_llm.types.batch.", ""),
        ("any_llm.types.model.", ""),
        ("any_llm.types.provider.", ""),
        ("any_llm.constants.", ""),
        ("any_llm.", ""),
        ("collections.abc.", ""),
        ("openai.types.responses.", ""),
        ("openai.types.chat.", ""),
        ("openai.types.", ""),
        ("openresponses_types.", ""),
        ("typing.", ""),
        ("typing_extensions.", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Strip any remaining "lowercase.module.path.ClassName" → "ClassName" patterns
    # that arise from SDK version differences in how types are qualified.
    text = re.sub(r"\b[a-z][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*\.([A-Z]\w*)", r"\1", text)
    # Strip Annotated[X, PropertyInfo(...)] → X at text level as a fallback
    # for cases where the type-level __metadata__ check above didn't fire.
    text = re.sub(r"Annotated\[(.+),\s*PropertyInfo\([^)]*\)\]", r"\1", text)

    # Normalize Union[X, Y, Z] → X | Y | Z to match Python 3.10+ union syntax.
    def _union_to_pipe(m: re.Match[str]) -> str:
        return m.group(1).replace(", ", " | ")

    return re.sub(r"Union\[([^\[\]]+)\]", _union_to_pipe, text)


def _format_annotation(annotation: Any) -> str:
    """Format a type annotation as a readable string."""
    if annotation is inspect.Parameter.empty:
        return ""
    if annotation is ...:
        return "..."
    # Annotated[X, metadata] - strip metadata, keep only the type.
    # Use __metadata__ which is always present on Annotated types regardless
    # of whether they come from typing or typing_extensions.
    if hasattr(annotation, "__metadata__"):
        return _format_annotation(typing.get_args(annotation)[0])
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)

    if origin is not None and args is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        origin_name = _short_name(origin_name)
        # Union types (X | Y)
        if origin_name == "Union" or "Union" in str(origin):
            parts = [_format_annotation(a) for a in args]
            # Replace NoneType with None
            parts = ["None" if p == "NoneType" else p for p in parts]
            return " | ".join(parts)
        # Literal types - quote string values
        if origin_name == "Literal" or "Literal" in str(origin):
            formatted_args = ", ".join(repr(a) if isinstance(a, str) else str(a) for a in args)
            return f"Literal[{formatted_args}]"
        formatted_args = ", ".join(_format_annotation(a) for a in args)
        return f"{origin_name}[{formatted_args}]"

    if hasattr(annotation, "__name__"):
        return _short_name(annotation.__name__)
    raw = str(annotation)
    return _short_name(raw)


def _format_default(default: Any) -> str:
    """Format a parameter default value."""
    if default is inspect.Parameter.empty:
        return ""
    if default is None:
        return "None"
    if isinstance(default, str):
        return f'"{default}"'
    return str(default)


def _get_signature_block(func: Any, func_name: str | None = None) -> str:
    """Render a function signature as a non-executable code block."""
    name = func_name or func.__name__
    is_async = inspect.iscoroutinefunction(func)
    prefix = "async " if is_async else ""

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return f"```\n{prefix}def {name}(...)\n```"

    params = list(sig.parameters.values())
    # Build signature lines
    lines = []
    seen_kw_only = False
    for p in params:
        if p.kind == inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            seen_kw_only = True
            if lines:
                lines.append("    *,")
        ann = _format_annotation(p.annotation) if p.annotation is not inspect.Parameter.empty else ""
        default = _format_default(p.default)
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            part = f"**{p.name}"
            if ann:
                part += f": {ann}"
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            part = f"*{p.name}"
            if ann:
                part += f": {ann}"
        else:
            part = p.name
            if ann:
                part += f": {ann}"
            if default:
                part += f" = {default}"
        lines.append(f"    {part},")

    # Return annotation
    ret = ""
    if sig.return_annotation is not inspect.Signature.empty:
        ret = f" -> {_format_annotation(sig.return_annotation)}"

    if not lines:
        result = f"```\n{prefix}def {name}(){ret}\n```"
    else:
        body = "\n".join(lines)
        result = f"```\n{prefix}def {name}(\n{body}\n){ret}\n```"

    # Clean up any remaining fully-qualified module paths in the block
    return _clean_qualified_names(result)


def _parse_docstring(docstring: str | None) -> dict[str, Any]:
    """Parse a Google-style docstring into sections."""
    result: dict[str, Any] = {"summary": "", "args": {}, "returns": "", "raises": []}
    if not docstring:
        return result

    lines = textwrap.dedent(docstring).strip().split("\n")
    current_section = "summary"
    current_arg = ""
    summary_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped in ("Args:", "Attributes:"):
            current_section = "args"
            continue
        if stripped == "Returns:":
            current_section = "returns"
            continue
        if stripped == "Raises:":
            current_section = "raises"
            continue

        if current_section == "summary":
            summary_lines.append(stripped)
        elif current_section == "args":
            # New arg: "name: description"
            m = re.match(r"^(\*{0,2}\w+)\s*(?:\(.*?\))?\s*:\s*(.*)", stripped)
            if m:
                current_arg = m.group(1)
                desc = m.group(2).strip()
                result["args"][current_arg] = desc
            elif current_arg and stripped:
                # Continuation line
                result["args"][current_arg] += " " + stripped
        elif current_section == "returns":
            if stripped:
                result["returns"] += (" " if result["returns"] else "") + stripped
        elif current_section == "raises":
            m = re.match(r"^(\w+)\s*:\s*(.*)", stripped)
            if m:
                result["raises"].append({"type": m.group(1), "desc": m.group(2).strip()})
            elif result["raises"] and stripped:
                result["raises"][-1]["desc"] += " " + stripped

    result["summary"] = " ".join(summary_lines).strip()
    return result


def _param_table(func: Any, parsed_doc: dict[str, Any]) -> str:
    """Generate a markdown parameter table merging signature and docstring info."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ""

    rows = []
    for name, param in sig.parameters.items():
        ann = _format_annotation(param.annotation) if param.annotation is not inspect.Parameter.empty else ""
        default = _format_default(param.default)
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            display_name = f"**{name}"
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            display_name = f"*{name}"
        else:
            display_name = name

        doc_desc = parsed_doc["args"].get(name, parsed_doc["args"].get(f"**{name}", ""))
        # Escape pipes in type annotations for markdown tables
        ann_escaped = ann.replace("|", "\\|")
        if not default and param.default is inspect.Parameter.empty:
            default = "*required*"
        elif not default:
            default = ""
        rows.append(f"| `{display_name}` | `{ann_escaped}` | {default} | {doc_desc} |")

    if not rows:
        return ""

    header = "| Parameter | Type | Default | Description |\n|-----------|------|---------|-------------|"
    return _clean_qualified_names(header + "\n" + "\n".join(rows))


def _generate_function_doc(
    func: Any,
    heading_level: str = "##",
    module_prefix: str = "any_llm",
    func_name: str | None = None,
) -> str:
    """Generate docs for a single function."""
    name = func_name or func.__name__
    parsed = _parse_docstring(func.__doc__)
    parts = [f"{heading_level} `{module_prefix}.{name}()`"]
    if parsed["summary"]:
        parts.append("")
        parts.append(parsed["summary"])
    parts.append("")
    parts.append(_get_signature_block(func, name))
    table = _param_table(func, parsed)
    if table:
        parts.append("")
        parts.append(f"{heading_level}# Parameters")
        parts.append("")
        parts.append(table)
    if parsed["returns"]:
        parts.append("")
        parts.append(f"**Returns:** {parsed['returns']}")
    if parsed["raises"]:
        parts.append("")
        parts.append("**Raises:**")
        for r in parsed["raises"]:
            parts.append(f"- `{r['type']}`: {r['desc']}")
    return "\n".join(parts)


def _generate_async_stub(
    func: Any,
    heading_level: str = "##",
    module_prefix: str = "any_llm",
    func_name: str | None = None,
) -> str:
    """Generate a short async variant section."""
    name = func_name or func.__name__
    parts = [f"{heading_level} `{module_prefix}.{name}()`"]
    parts.append("")
    parts.append("Async variant with the same parameters.")
    parts.append("")
    parts.append(_get_signature_block(func, name))
    return "\n".join(parts)


def _pydantic_field_table(cls: type) -> str:
    """Generate a field table for a Pydantic BaseModel or attrs-style class with annotated fields."""
    rows = []
    hints = {}
    try:
        hints = get_type_hints(cls)
    except Exception:
        # Fallback to __annotations__
        hints = getattr(cls, "__annotations__", {})

    # Get field info from Pydantic model_fields if available
    model_fields = getattr(cls, "model_fields", None)

    for field_name, field_type in hints.items():
        if field_name.startswith("_"):
            continue
        if field_name == "model_config":
            continue

        ann = _format_annotation(field_type).replace("|", "\\|")

        # Try to get description from field docstring or Pydantic field
        desc = ""
        if model_fields and field_name in model_fields:
            fi = model_fields[field_name]
            desc = fi.description or ""

        # Fallback: look for inline docstring in source
        if not desc:
            try:
                source = inspect.getsource(cls)
                # Match the pattern: field_name: type ... \n    """docstring"""
                # Use a non-greedy match and limit to the next field or end
                pattern = rf'(?:^|\n)\s+{re.escape(field_name)}\s*[:=][^\n]*\n\s+"""([^"]*(?:""[^"])*?)"""'
                m = re.search(pattern, source)
                if m:
                    desc = " ".join(m.group(1).strip().split())
            except (OSError, TypeError):
                pass

        rows.append(f"| `{field_name}` | `{ann}` | {desc} |")

    if not rows:
        return ""

    header = "| Field | Type | Description |\n|-------|------|-------------|"
    return _clean_qualified_names(header + "\n" + "\n".join(rows))


def generate_any_llm_page() -> str:
    """Generate api/any-llm.md."""
    parts = [
        "---",
        "title: AnyLLM",
        "description: The AnyLLM class - provider interface with metadata access and reusability",
        "---",
        "",
        "The `AnyLLM` class is the provider interface at the core of any-llm. Use it when you need to make multiple requests against the same provider without re-instantiating on every call.",
        "",
    ]

    # create()
    create_method = AnyLLM.create
    parsed = _parse_docstring(create_method.__doc__)
    parts.append("## Creating an Instance")
    parts.append("")
    parts.append("### `AnyLLM.create()`")
    parts.append("")
    parts.append("Factory method that returns a configured `AnyLLM` instance for the given provider.")
    parts.append("")
    parts.append(_get_signature_block(create_method, "create"))
    table = _param_table(create_method, parsed)
    if table:
        parts.append("")
        parts.append(table)
    parts.append("")
    parts.append("**Returns:** An `AnyLLM` instance bound to the specified provider.")
    parts.append("")
    parts.append(
        """```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai", api_key="sk-...")

response = llm.completion(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```"""
    )

    # Static methods
    parts.append("")
    parts.append("## Static Methods")

    # split_model_provider
    smp = AnyLLM.split_model_provider
    parts.append("")
    parts.append("### `AnyLLM.split_model_provider()`")
    parts.append("")
    parts.append('Parses a combined `"provider:model"` string into its components.')
    parts.append("")
    parts.append(_get_signature_block(smp, "split_model_provider"))
    parts.append("")
    parts.append(
        """| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Combined identifier in `"provider:model"` format (e.g., `"openai:gpt-4.1-mini"`). The legacy `"provider/model"` format is also accepted but deprecated. |"""
    )
    parts.append("")
    parts.append("**Returns:** A `(LLMProvider, model_name)` tuple.")
    parts.append("")
    parts.append("**Raises:** `ValueError` if the string does not contain a `:` or `/` delimiter.")
    parts.append("")
    parts.append(
        """```python
provider, model_name = AnyLLM.split_model_provider("anthropic:claude-sonnet-4-20250514")
# provider = LLMProvider.ANTHROPIC
# model_name = "claude-sonnet-4-20250514"
```"""
    )

    # get_all_provider_metadata
    parts.append("")
    parts.append("### `AnyLLM.get_all_provider_metadata()`")
    parts.append("")
    parts.append("Returns metadata for every supported provider, sorted alphabetically by name.")
    parts.append("")
    gapm = AnyLLM.get_all_provider_metadata
    parts.append(_get_signature_block(gapm, "get_all_provider_metadata"))
    parts.append("")
    parts.append("**Returns:** A list of [`ProviderMetadata`](types/provider.md) objects.")
    parts.append("")
    parts.append(
        """```python
for meta in AnyLLM.get_all_provider_metadata():
    print(f"{meta.name}: streaming={meta.streaming}, embedding={meta.embedding}")
```"""
    )

    # get_supported_providers
    parts.append("")
    parts.append("### `AnyLLM.get_supported_providers()`")
    parts.append("")
    parts.append("Returns a list of all supported provider key strings.")
    parts.append("")
    gsp = AnyLLM.get_supported_providers
    parts.append(_get_signature_block(gsp, "get_supported_providers"))
    parts.append("")
    parts.append('**Returns:** `list[str]` of provider keys (e.g., `["anthropic", "openai", ...]`).')

    # Instance methods
    parts.append("")
    parts.append("## Instance Methods")
    parts.append("")
    parts.append("All instance methods below are called on an `AnyLLM` object returned by `AnyLLM.create()`.")

    # completion / acompletion
    parts.append("")
    parts.append("### `completion()` / `acompletion()`")
    parts.append("")
    parts.append("Create a chat completion. See the [Completion](completion.md) reference for the full parameter list.")
    parts.append("")
    parts.append(
        """```
def completion(self, model, messages, *, stream=None, response_format=None, **kwargs)
    -> ChatCompletion | Iterator[ChatCompletionChunk] | ParsedChatCompletion

async def acompletion(self, model, messages, *, stream=None, response_format=None, **kwargs)
    -> ChatCompletion | AsyncIterator[ChatCompletionChunk] | ParsedChatCompletion
```"""
    )

    # responses / aresponses
    parts.append("")
    parts.append("### `responses()` / `aresponses()`")
    parts.append("")
    parts.append("Create a response using the OpenResponses API. See the [Responses](responses.md) reference.")
    parts.append("")
    parts.append(
        """```
def responses(self, **kwargs)
    -> ResponseResource | Response | Iterator[ResponseStreamEvent]

async def aresponses(self, **kwargs)
    -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]
```"""
    )

    # messages / amessages
    parts.append("")
    parts.append("### `messages()` / `amessages()`")
    parts.append("")
    parts.append(
        "Create a message using the Anthropic Messages API format. All providers support this through automatic conversion."
    )
    parts.append("")
    parts.append(
        """```
def messages(self, **kwargs)
    -> MessageResponse | Iterator[MessageStreamEvent]

async def amessages(self, model, messages, max_tokens, **kwargs)
    -> MessageResponse | AsyncIterator[MessageStreamEvent]
```"""
    )

    # list_models / alist_models
    parts.append("")
    parts.append("### `list_models()` / `alist_models()`")
    parts.append("")
    parts.append("List available models for this provider. See the [List Models](list-models.md) reference.")
    parts.append("")
    parts.append(
        """```
def list_models(self, **kwargs) -> Sequence[Model]
async def alist_models(self, **kwargs) -> Sequence[Model]
```"""
    )

    # create_batch / acreate_batch
    parts.append("")
    parts.append("### `create_batch()` / `acreate_batch()`")
    parts.append("")
    parts.append("Create a batch job. See the [Batch](batch.md) reference.")
    parts.append("")
    parts.append(
        """```
def create_batch(self, **kwargs) -> Batch
async def acreate_batch(self, input_file_path, endpoint, completion_window="24h", metadata=None, **kwargs) -> Batch
```"""
    )

    # get_provider_metadata
    parts.append("")
    parts.append("### `get_provider_metadata()`")
    parts.append("")
    parts.append("Returns metadata for this provider instance's class.")
    parts.append("")
    gpm = AnyLLM.get_provider_metadata
    parts.append(_get_signature_block(gpm, "get_provider_metadata"))
    parts.append("")
    parts.append(
        "**Returns:** A [`ProviderMetadata`](types/provider.md) object describing the provider's capabilities."
    )
    parts.append("")
    parts.append(
        """```python
llm = AnyLLM.create("mistral")
meta = llm.get_provider_metadata()
print(f"Supports streaming: {meta.streaming}")
print(f"Supports embedding: {meta.embedding}")
print(f"Supports responses: {meta.responses}")
```"""
    )

    return "\n".join(parts) + "\n"


def generate_completion_page() -> str:
    """Generate api/completion.md."""
    sync_fn = api_module.completion
    async_fn = api_module.acompletion
    parsed = _parse_docstring(sync_fn.__doc__)

    parts = [
        "---",
        "title: Completion",
        "description: Create chat completions with any provider",
        "---",
        "",
        "The `completion` and `acompletion` functions are the primary way to generate chat completions across all supported providers. They accept an OpenAI-compatible parameter set and return OpenAI-compatible response types.",
        "",
        "## `any_llm.completion()`",
        "",
        _get_signature_block(sync_fn),
        "",
        "## `any_llm.acompletion()`",
        "",
        "Async variant with the same parameters. Returns `ChatCompletion | AsyncIterator[ChatCompletionChunk]`.",
        "",
        _get_signature_block(async_fn),
        "",
        "## Parameters",
        "",
        _param_table(sync_fn, parsed),
        "",
        "## Return Value",
        "",
        "- **Non-streaming** (`stream=None` or `stream=False`): Returns a [`ChatCompletion`](types/completion.md) object.",
        "- **Streaming** (`stream=True`): Returns an `Iterator[ChatCompletionChunk]` (sync) or `AsyncIterator[ChatCompletionChunk]` (async).",
        "- **Structured output** (when `response_format` is a Pydantic model or dataclass): Returns a `ParsedChatCompletion[T]` with a `.choices[0].message.parsed` field containing the deserialized object.",
        "",
        "## Usage",
        "",
        "### Basic completion",
        "",
        """```python
from any_llm import completion

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```""",
        "",
        "### Streaming",
        "",
        """```python
for chunk in completion(
    model="gpt-4.1-mini",
    provider="openai",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```""",
        "",
        "### Async",
        "",
        """```python
import asyncio
from any_llm import acompletion

async def main():
    response = await acompletion(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```""",
        "",
        "### Structured output",
        "",
        """```python
from pydantic import BaseModel
from any_llm import completion

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

response = completion(
    model="gpt-4.1-mini",
    provider="openai",
    messages=[{"role": "user", "content": "Tell me about Paris."}],
    response_format=CityInfo,
)
city = response.choices[0].message.parsed
print(f"{city.name}, {city.country} - pop. {city.population}")
```""",
        "",
        "### Tool calling",
        "",
        """```python
from any_llm import completion

def get_weather(location: str, unit: str = "F") -> str:
    \"\"\"Get weather information for a location.

    Args:
        location: The city or location to get weather for
        unit: Temperature unit, either 'C' or 'F'

    Returns:
        Current weather description
    \"\"\"
    return f"Weather in {location} is sunny and 75{unit}!"

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
)
```""",
    ]
    return "\n".join(parts) + "\n"


def generate_responses_page() -> str:
    """Generate api/responses.md."""
    sync_fn = api_module.responses
    async_fn = api_module.aresponses
    parsed = _parse_docstring(sync_fn.__doc__)

    parts = [
        "---",
        "title: Responses",
        "description: OpenResponses API for agentic AI systems",
        "---",
        "",
        "The `responses` and `aresponses` functions implement the [OpenResponses specification](https://github.com/openresponsesspec/openresponses), a vendor-neutral API for agentic AI systems. This API supports multi-turn conversations, tool use, and streaming events.",
        "",
        "## Return Types",
        "",
        "The return type depends on the provider and whether streaming is enabled:",
        "",
        "| Condition | Return Type |",
        "|-----------|-------------|",
        "| OpenResponses-compliant provider (non-streaming) | `openresponses_types.ResponseResource` |",
        "| OpenAI-native provider (non-streaming) | `openai.types.responses.Response` |",
        "| Streaming (`stream=True`) | `Iterator[ResponseStreamEvent]` (sync) or `AsyncIterator[ResponseStreamEvent]` (async) |",
        "",
        "## `any_llm.responses()`",
        "",
        _get_signature_block(sync_fn),
        "",
        "## `any_llm.aresponses()`",
        "",
        "Async variant with the same parameters. Returns `ResponseResource | Response | AsyncIterator[ResponseStreamEvent]`.",
        "",
        _get_signature_block(async_fn),
        "",
        "## Parameters",
        "",
        _param_table(sync_fn, parsed),
        "",
        "## Usage",
        "",
        "### Basic response",
        "",
        """```python
from any_llm import responses

result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is the capital of France?",
)
print(result.output_text)
```""",
        "",
        "### With instructions",
        "",
        """```python
result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Translate to French: Hello, how are you?",
    instructions="You are a professional translator. Always respond with only the translation.",
)
```""",
        "",
        "### Streaming",
        "",
        """```python
for event in responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Tell me a short story.",
    stream=True,
):
    print(event)
```""",
        "",
        "### Multi-turn with `previous_response_id`",
        "",
        """```python
first = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="My name is Alice.",
    store=True,
)

second = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is my name?",
    previous_response_id=first.id,
)
```""",
        "",
        '{% hint style="info" %}',
        "Not all providers support the Responses API. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.responses` programmatically.",
        "{% endhint %}",
    ]
    return "\n".join(parts) + "\n"


def generate_embedding_page() -> str:
    """Generate api/embedding.md."""
    sync_fn = api_module.embedding
    async_fn = api_module.aembedding
    parsed = _parse_docstring(sync_fn.__doc__)

    parts = [
        "---",
        "title: Embedding",
        "description: Create text embeddings with any provider",
        "---",
        "",
        "The `embedding` and `aembedding` functions create vector embeddings from text using a unified interface across all providers that support embeddings.",
        "",
        "## `any_llm.embedding()`",
        "",
        _get_signature_block(sync_fn),
        "",
        "## `any_llm.aembedding()`",
        "",
        "Async variant with the same parameters.",
        "",
        _get_signature_block(async_fn),
        "",
        "## Parameters",
        "",
        _param_table(sync_fn, parsed),
        "",
        "## Return Value",
        "",
        "Returns a [`CreateEmbeddingResponse`](types/completion.md) containing:",
        "",
        "- `data` -- list of `Embedding` objects, each with an `embedding` vector (`list[float]`) and an `index`.",
        "- `model` -- the model used.",
        "- `usage` -- token usage information with `prompt_tokens` and `total_tokens`.",
        "",
        "## Usage",
        "",
        "### Single text",
        "",
        """```python
from any_llm import embedding

result = embedding(
    model="text-embedding-3-small",
    provider="openai",
    inputs="Hello, world!",
)

vector = result.data[0].embedding
print(f"Dimensions: {len(vector)}")
print(f"Tokens used: {result.usage.total_tokens}")
```""",
        "",
        "### Batch embedding",
        "",
        """```python
result = embedding(
    model="text-embedding-3-small",
    provider="openai",
    inputs=["First sentence", "Second sentence", "Third sentence"],
)

for item in result.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```""",
        "",
        "### Async",
        "",
        """```python
import asyncio
from any_llm import aembedding

async def main():
    result = await aembedding(
        model="text-embedding-3-small",
        provider="openai",
        inputs="Hello, world!",
    )
    print(f"Dimensions: {len(result.data[0].embedding)}")

asyncio.run(main())
```""",
        "",
        '{% hint style="info" %}',
        "Not all providers support embeddings. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.embedding` programmatically.",
        "{% endhint %}",
    ]
    return "\n".join(parts) + "\n"


def generate_list_models_page() -> str:
    """Generate api/list-models.md."""
    sync_fn = api_module.list_models
    async_fn = api_module.alist_models
    parsed = _parse_docstring(sync_fn.__doc__)

    parts = [
        "---",
        "title: List Models",
        "description: List available models for a provider",
        "---",
        "",
        "The `list_models` and `alist_models` functions return the available models for a given provider.",
        "",
        "## `any_llm.list_models()`",
        "",
        _get_signature_block(sync_fn),
        "",
        "## `any_llm.alist_models()`",
        "",
        "Async variant with the same parameters.",
        "",
        _get_signature_block(async_fn),
        "",
        "## Parameters",
        "",
        _param_table(sync_fn, parsed),
        "",
        "## Return Value",
        "",
        "Returns a `Sequence` of [`Model`](types/model.md) objects. Each `Model` has at minimum an `id` field containing the model identifier string.",
        "",
        "## Usage",
        "",
        """```python
from any_llm import list_models

models = list_models("openai")
for model in models:
    print(model.id)
```""",
        "",
        "### Async",
        "",
        """```python
import asyncio
from any_llm import alist_models

async def main():
    models = await alist_models("mistral")
    for model in models:
        print(model.id)

asyncio.run(main())
```""",
        "",
        "### Using the AnyLLM class",
        "",
        """```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai")
models = llm.list_models()
print(f"Available models: {len(models)}")
```""",
        "",
        '{% hint style="info" %}',
        "Not all providers support listing models. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.list_models` programmatically.",
        "{% endhint %}",
    ]
    return "\n".join(parts) + "\n"


def generate_batch_page() -> str:
    """Generate api/batch.md."""
    parts = [
        "---",
        "title: Batch",
        "description: Process multiple requests asynchronously at lower cost",
        "---",
        "",
        '{% hint style="warning" %}',
        "The Batch API is experimental and may change in future releases. Provider support is limited - check the [providers page](../providers.md) for availability.",
        "{% endhint %}",
        "",
        "The Batch API lets you submit multiple requests as a single job for asynchronous processing, typically at lower cost than real-time requests.",
        "",
        "## How It Works",
        "",
        "1. Prepare a JSONL file where each line is a batch request object.",
        "2. Call `create_batch()` with the file path and target endpoint.",
        "3. any-llm uploads the file to the provider and creates the batch job.",
        "4. Poll with `retrieve_batch()` to check status.",
        "5. When complete, download results from the provider.",
        "",
        "### Input File Format",
        "",
        "The input file must be a JSONL file where each line follows this structure:",
        "",
        '```json\n{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hello"}]}}\n{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "World"}]}}\n```',
    ]

    # create_batch
    cb = api_module.create_batch
    cb_parsed = _parse_docstring(cb.__wrapped__.__doc__ if hasattr(cb, "__wrapped__") else cb.__doc__)
    parts.append("")
    parts.append("## `any_llm.create_batch()`")
    parts.append("")
    parts.append("Create a batch job by uploading a local JSONL file.")
    parts.append("")
    parts.append(_get_signature_block(cb.__wrapped__ if hasattr(cb, "__wrapped__") else cb, "create_batch"))
    parts.append("")
    parts.append("### `any_llm.acreate_batch()`")
    parts.append("")
    parts.append("Async variant with the same parameters.")
    parts.append("")
    parts.append("## Parameters (create)")
    parts.append("")
    parts.append(_param_table(cb.__wrapped__ if hasattr(cb, "__wrapped__") else cb, cb_parsed))
    parts.append("")
    parts.append("**Returns:** A [`Batch`](types/batch.md) object.")

    # retrieve_batch
    rb = api_module.retrieve_batch
    rb_parsed = _parse_docstring(rb.__wrapped__.__doc__ if hasattr(rb, "__wrapped__") else rb.__doc__)
    parts.append("")
    parts.append("## `any_llm.retrieve_batch()`")
    parts.append("")
    parts.append("Retrieve the current status and details of a batch job.")
    parts.append("")
    parts.append(_get_signature_block(rb.__wrapped__ if hasattr(rb, "__wrapped__") else rb, "retrieve_batch"))
    parts.append("")
    parts.append("### `any_llm.aretrieve_batch()`")
    parts.append("")
    parts.append("Async variant with the same parameters.")
    parts.append("")
    parts.append("## Parameters (retrieve)")
    parts.append("")
    parts.append(_param_table(rb.__wrapped__ if hasattr(rb, "__wrapped__") else rb, rb_parsed))
    parts.append("")
    parts.append("**Returns:** A [`Batch`](types/batch.md) object.")

    # cancel_batch
    canb = api_module.cancel_batch
    parts.append("")
    parts.append("## `any_llm.cancel_batch()`")
    parts.append("")
    parts.append("Cancel an in-progress batch job.")
    parts.append("")
    parts.append(_get_signature_block(canb.__wrapped__ if hasattr(canb, "__wrapped__") else canb, "cancel_batch"))
    parts.append("")
    parts.append("### `any_llm.acancel_batch()`")
    parts.append("")
    parts.append("Async variant with the same parameters.")
    parts.append("")
    parts.append("**Returns:** The cancelled [`Batch`](types/batch.md) object.")

    # list_batches
    lb = api_module.list_batches
    lb_parsed = _parse_docstring(lb.__wrapped__.__doc__ if hasattr(lb, "__wrapped__") else lb.__doc__)
    parts.append("")
    parts.append("## `any_llm.list_batches()`")
    parts.append("")
    parts.append("List batch jobs for a provider.")
    parts.append("")
    parts.append(_get_signature_block(lb.__wrapped__ if hasattr(lb, "__wrapped__") else lb, "list_batches"))
    parts.append("")
    parts.append("### `any_llm.alist_batches()`")
    parts.append("")
    parts.append("Async variant with the same parameters.")
    parts.append("")
    parts.append("## Parameters (list)")
    parts.append("")
    parts.append(_param_table(lb.__wrapped__ if hasattr(lb, "__wrapped__") else lb, lb_parsed))
    parts.append("")
    parts.append("**Returns:** A `Sequence` of [`Batch`](types/batch.md) objects.")

    # Usage
    parts.append("")
    parts.append("## Usage")
    parts.append("")
    parts.append(
        """```python
from any_llm import create_batch, retrieve_batch, list_batches

# Create a batch job
batch = create_batch(
    provider="openai",
    input_file_path="requests.jsonl",
    endpoint="/v1/chat/completions",
)
print(f"Batch created: {batch.id}, status: {batch.status}")

# Check status
batch = retrieve_batch("openai", batch.id)
print(f"Status: {batch.status}")

# List all batches
batches = list_batches("openai")
for b in batches:
    print(f"{b.id}: {b.status}")
```"""
    )

    return "\n".join(parts) + "\n"


def generate_exceptions_page() -> str:
    """Generate api/exceptions.md."""
    parts = [
        "---",
        "title: Exceptions",
        "description: Unified exception hierarchy for all providers",
        "---",
        "",
        "any-llm provides a unified exception hierarchy so you can handle errors consistently regardless of which provider is being used. When unified exceptions are enabled, provider-specific SDK errors are automatically mapped to the appropriate any-llm exception type.",
        "",
        '{% hint style="info" %}',
        "**Opt-in Feature:** Unified exception handling is opt-in. Set the `ANY_LLM_UNIFIED_EXCEPTIONS=1` environment variable to enable automatic conversion from provider-specific exceptions.",
        "{% endhint %}",
        "",
        "## Exception Hierarchy",
        "",
        "All exceptions inherit from `AnyLLMError`:",
        "",
        "```",
        "AnyLLMError",
    ]

    # List exception classes from the module
    exception_classes = []
    for name in dir(exc_module):
        obj = getattr(exc_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Exception)
            and obj is not exc_module.AnyLLMError
            and obj.__module__ == exc_module.__name__
        ):
            exception_classes.append(name)

    for i, name in enumerate(exception_classes):
        prefix = "└── " if i == len(exception_classes) - 1 else "├── "
        parts.append(f"{prefix}{name}")

    parts.append("```")

    # AnyLLMError
    base = exc_module.AnyLLMError
    parts.append("")
    parts.append("## `AnyLLMError`")
    parts.append("")
    parsed = _parse_docstring(base.__doc__)
    if parsed["summary"]:
        parts.append(f"{parsed['summary']}")
        parts.append("")
    parts.append(_get_signature_block(base.__init__, "AnyLLMError"))
    parts.append("")
    parts.append(
        """| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error message. |
| `original_exception` | `Exception \\| None` | The original SDK exception that triggered this error. |
| `provider_name` | `str \\| None` | Name of the provider that raised the error (if available). |"""
    )
    parts.append("")
    parts.append(
        'The string representation includes the provider name when available: `"[openai] Rate limit exceeded"`.'
    )

    # Provider errors (simple ones)
    simple_errors = [
        ("RateLimitError", "Raised when the API rate limit is exceeded."),
        ("AuthenticationError", "Raised when authentication with the provider fails (invalid or missing API key)."),
        ("InvalidRequestError", "Raised when the request to the provider is malformed or contains invalid parameters."),
        ("ProviderError", "Raised when the provider encounters an internal error (5xx-class errors)."),
        ("ContentFilterError", "Raised when content is blocked by the provider's safety filter."),
        ("ModelNotFoundError", "Raised when the requested model is not found or not available."),
        ("ContextLengthExceededError", "Raised when the input exceeds the model's maximum context length."),
    ]

    parts.append("")
    parts.append("## Provider Errors")

    for name, desc in simple_errors:
        cls = getattr(exc_module, name)
        parts.append("")
        parts.append(f"### `{name}`")
        parts.append("")
        parts.append(desc)
        parts.append("")
        parts.append(f"```\nclass {name}(AnyLLMError): ...\n```")
        parts.append("")
        parts.append(f'Default message: `"{cls.default_message}"`')

    # Configuration errors
    parts.append("")
    parts.append("## Configuration Errors")

    # MissingApiKeyError
    parts.append("")
    parts.append("### `MissingApiKeyError`")
    parts.append("")
    parts.append("Raised when a required API key is not provided via the parameter or environment variable.")
    parts.append("")
    parts.append(
        """```
class MissingApiKeyError(AnyLLMError):
    def __init__(self, provider_name: str, env_var_name: str) -> None: ...
```"""
    )
    parts.append("")
    parts.append(
        """| Attribute | Type | Description |
|-----------|------|-------------|
| `provider_name` | `str` | Name of the provider requiring the key. |
| `env_var_name` | `str` | Environment variable name that was checked. |"""
    )
    parts.append("")
    parts.append(
        'Example message: `"No openai API key provided. Please provide it in the config or set the OPENAI_API_KEY environment variable."`'
    )

    # UnsupportedProviderError
    parts.append("")
    parts.append("### `UnsupportedProviderError`")
    parts.append("")
    parts.append("Raised when an unsupported provider is specified.")
    parts.append("")
    parts.append(
        """```
class UnsupportedProviderError(AnyLLMError):
    def __init__(self, provider_key: str, supported_providers: list[str]) -> None: ...
```"""
    )
    parts.append("")
    parts.append(
        """| Attribute | Type | Description |
|-----------|------|-------------|
| `provider_key` | `str` | The unsupported provider key that was specified. |
| `supported_providers` | `list[str]` | List of valid provider keys. |"""
    )

    # UnsupportedParameterError
    parts.append("")
    parts.append("### `UnsupportedParameterError`")
    parts.append("")
    parts.append("Raised when a parameter is not supported by the provider.")
    parts.append("")
    parts.append(
        """```
class UnsupportedParameterError(AnyLLMError):
    def __init__(self, parameter_name: str, provider_name: str, additional_message: str | None = None) -> None: ...
```"""
    )
    parts.append("")
    parts.append(
        """| Attribute | Type | Description |
|-----------|------|-------------|
| `parameter_name` | `str` | The unsupported parameter name. |
| `provider_name` | `str` | Name of the provider (also accessible via the inherited `provider_name` attribute). |"""
    )

    # Usage
    parts.append("")
    parts.append("## Usage")
    parts.append("")
    parts.append(
        """```python
from any_llm import completion
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError,
)

try:
    response = completion(
        model="gpt-4.1-mini",
        provider="openai",
        messages=[{"role": "user", "content": "Hello!"}],
    )
except RateLimitError as e:
    print(f"Rate limited by {e.provider_name}: {e.message}")
    # Access the original provider exception for details
    print(f"Original: {e.original_exception}")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except ContextLengthExceededError as e:
    print(f"Input too long: {e.message}")
except AnyLLMError as e:
    # Catch-all for any other any-llm error
    print(f"Error: {e}")
```"""
    )

    return "\n".join(parts) + "\n"


def generate_messages_page() -> str:
    """Generate api/messages.md."""
    sync_fn = api_module.messages
    async_fn = api_module.amessages
    parsed = _parse_docstring(sync_fn.__doc__)

    parts = [
        "---",
        "title: Messages",
        "description: Anthropic Messages API for all providers",
        "---",
        "",
        "The `messages` and `amessages` functions use the Anthropic Messages API format. All providers support this through automatic conversion, so you can use the same Anthropic-style message format regardless of backend.",
        "",
        "## `any_llm.messages()`",
        "",
        _get_signature_block(sync_fn),
        "",
        "## `any_llm.amessages()`",
        "",
        "Async variant with the same parameters. Returns `MessageResponse | AsyncIterator[MessageStreamEvent]`.",
        "",
        _get_signature_block(async_fn),
        "",
        "## Parameters",
        "",
        _param_table(sync_fn, parsed),
        "",
        "## Return Value",
        "",
        "- **Non-streaming**: Returns a [`MessageResponse`](types/messages.md) object.",
        "- **Streaming** (`stream=True`): Returns an `Iterator[MessageStreamEvent]` (sync) or `AsyncIterator[MessageStreamEvent]` (async).",
        "",
        "## Usage",
        "",
        "### Basic message",
        "",
        """```python
from any_llm.api import messages

response = messages(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
)
print(response.content[0].text)
```""",
        "",
        "### With system prompt",
        "",
        """```python
response = messages(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    messages=[{"role": "user", "content": "Translate to French: Hello"}],
    max_tokens=1024,
    system="You are a professional translator.",
)
```""",
        "",
        "### Async",
        "",
        """```python
import asyncio
from any_llm.api import amessages

async def main():
    response = await amessages(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=1024,
    )
    print(response.content[0].text)

asyncio.run(main())
```""",
    ]
    return "\n".join(parts) + "\n"


def generate_types_completion_page() -> str:
    """Generate api/types/completion.md."""
    parts = [
        "---",
        "title: Completion Types",
        "description: Data models and types for completion operations",
        "---",
        "",
        "The completion types used by `any_llm.completion()` and `any_llm.acompletion()` are re-exports from the [OpenAI Python SDK](https://github.com/openai/openai-python), extended where needed to support additional fields like reasoning content.",
        "",
        "## Primary Types",
        "",
        "### `ChatCompletion`",
        "",
        "The response object for a non-streaming completion request. Extends `openai.types.chat.ChatCompletion` with support for reasoning content in the message choices.",
        "",
        "**Import:** `from any_llm.types.completion import ChatCompletion`",
        "",
        "Key fields:",
        "",
    ]

    parts.append(
        _pydantic_field_table(completion_types.ChatCompletion)
        or """| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique completion identifier. |
| `choices` | `list[Choice]` | List of completion choices. Each choice has a `message` with `content`, `role`, and optionally `reasoning` and `tool_calls`. |
| `model` | `str` | The model used. |
| `usage` | `CompletionUsage \\| None` | Token usage (prompt, completion, total). |"""
    )

    parts.extend(
        [
            "",
            "### `ChatCompletionChunk`",
            "",
            "A single chunk in a streaming completion response. Extends `openai.types.chat.ChatCompletionChunk`.",
            "",
            "**Import:** `from any_llm.types.completion import ChatCompletionChunk`",
            "",
            "Key fields:",
            "",
            """| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Completion identifier (same across all chunks). |
| `choices` | `list[ChunkChoice]` | Each chunk choice has a `delta` with incremental `content`, `role`, and optionally `reasoning`. |
| `model` | `str` | The model used. |""",
            "",
            "### `ChatCompletionMessage`",
            "",
            "A message within a completion response. Extends `openai.types.chat.ChatCompletionMessage` with a `reasoning` field.",
            "",
            "**Import:** `from any_llm.types.completion import ChatCompletionMessage`",
            "",
            """| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Message role (e.g., `"assistant"`). |
| `content` | `str \\| None` | Text content of the message. |
| `reasoning` | `Reasoning \\| None` | Reasoning/thinking content (when the model supports it). |
| `tool_calls` | `list[ChatCompletionMessageToolCall] \\| None` | Tool calls requested by the model. |
| `annotations` | `list[dict] \\| None` | Annotations attached to the message. |""",
            "",
            "### `ParsedChatCompletion`",
            "",
            "Returned when `response_format` is a Pydantic `BaseModel` subclass or a dataclass type. Extends `ChatCompletion` with a generic type parameter.",
            "",
            "**Import:** `from any_llm import ParsedChatCompletion`",
            "",
            "Access the parsed object via `response.choices[0].message.parsed`, which will be an instance of the type passed as `response_format`.",
            "",
            "### `CreateEmbeddingResponse`",
            "",
            "Response object for embedding requests. Re-exported directly from `openai.types.CreateEmbeddingResponse`.",
            "",
            "**Import:** `from any_llm.types.completion import CreateEmbeddingResponse`",
            "",
            """| Field | Type | Description |
|-------|------|-------------|
| `data` | `list[Embedding]` | List of embedding objects, each with an `embedding` vector and `index`. |
| `model` | `str` | The model used. |
| `usage` | `Usage` | Token usage with `prompt_tokens` and `total_tokens`. |""",
            "",
            "### `ReasoningEffort`",
            "",
            "A literal type controlling reasoning depth for models that support it.",
            "",
            "**Import:** `from any_llm.types.completion import ReasoningEffort`",
            "",
            """```
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "auto"]
```""",
            "",
            'The value `"auto"` (the default) maps to each provider\'s own default reasoning level.',
            "",
            "## Internal Types",
            "",
            "### `CompletionParams`",
            "",
            "Normalized parameters for chat completions, used internally to pass structured parameters from the public API to provider implementations.",
            "",
            "**Import:** `from any_llm.types.completion import CompletionParams`",
            "",
        ]
    )

    table = _pydantic_field_table(completion_types.CompletionParams)
    if table:
        parts.append(table)

    parts.extend(
        [
            "",
            "## Additional Re-exports",
            "",
            "The following types are also available from `any_llm.types.completion`:",
            "",
            """| Type | Origin | Description |
|------|--------|-------------|
| `CompletionUsage` | `openai.types.CompletionUsage` | Token usage counts. |
| `Function` | `openai.types.chat` | Function definition within a tool call. |
| `Embedding` | `openai.types.Embedding` | Single embedding vector with index. |
| `ChoiceDeltaToolCall` | `openai.types.chat` | Tool call delta in streaming chunks. |""",
            "",
            "For full field-level documentation of the base OpenAI types, see the [OpenAI Python SDK reference](https://github.com/openai/openai-python).",
        ]
    )
    return "\n".join(parts) + "\n"


def generate_types_responses_page() -> str:
    """Generate api/types/responses.md."""
    parts = [
        "---",
        "title: Responses Types",
        "description: Data models for the OpenResponses API",
        "---",
        "",
        "The Responses API types come from two sources depending on the provider:",
        "",
        "- **OpenResponses-compliant providers** return `ResponseResource` from the [`openresponses-types`](https://pypi.org/project/openresponses-types/) package.",
        "- **OpenAI-native providers** return `Response` from the `openai` SDK.",
        "- **Streaming** always yields `ResponseStreamEvent` objects.",
        "",
        "## Primary Types",
        "",
        "### `ResponseResource`",
        "",
        "The response object from providers implementing the [OpenResponses specification](https://github.com/openresponsesspec/openresponses).",
        "",
        "**Import:** `from openresponses_types import ResponseResource`",
        "",
        "**Package:** [`openresponses-types`](https://pypi.org/project/openresponses-types/)",
        "",
        "This is the primary return type for OpenResponses-compliant providers. It provides a standardized interface for accessing response content, tool calls, and metadata.",
        "",
        "### `Response`",
        "",
        "The response object from OpenAI's native Responses API. Re-exported from `openai.types.responses.Response`.",
        "",
        "**Import:** `from any_llm.types.responses import Response`",
        "",
        "This is returned by providers that use OpenAI's API directly (e.g., the `openai` provider).",
        "",
        "### `ResponseStreamEvent`",
        "",
        "A single event in a streaming response. Re-exported from `openai.types.responses.ResponseStreamEvent`.",
        "",
        "**Import:** `from any_llm.types.responses import ResponseStreamEvent`",
        "",
        "Stream events represent incremental updates during response generation, including content deltas, tool call events, and completion signals.",
        "",
        "### `ResponseInputParam`",
        "",
        "The input type accepted by the `input_data` parameter of `responses()` and `aresponses()`. Re-exported from `openai.types.responses.ResponseInputParam`.",
        "",
        "**Import:** `from any_llm.types.responses import ResponseInputParam`",
        "",
        "This is typically a list of message items that can include text, images, and tool-related content.",
        "",
        "### `ResponseOutputMessage`",
        "",
        "An output message within a response. Re-exported from `openai.types.responses.ResponseOutputMessage`.",
        "",
        "**Import:** `from any_llm.types.responses import ResponseOutputMessage`",
        "",
        "## Internal Types",
        "",
        "### `ResponsesParams`",
        "",
        "Normalized parameters for the Responses API, used internally to pass structured parameters from the public API to provider implementations.",
        "",
        "**Import:** `from any_llm.types.responses import ResponsesParams`",
        "",
    ]

    table = _pydantic_field_table(responses_types.ResponsesParams)
    if table:
        parts.append(table)

    parts.extend(
        [
            "",
            "## Type Mapping Summary",
            "",
            """| Type | Source | Used When |
|------|--------|-----------|
| `ResponseResource` | `openresponses-types` | OpenResponses-compliant providers, non-streaming |
| `Response` | `openai.types.responses` | OpenAI-native providers, non-streaming |
| `ResponseStreamEvent` | `openai.types.responses` | All providers, streaming (`stream=True`) |
| `ResponseInputParam` | `openai.types.responses` | Input parameter type |""",
            "",
            "For full details on the OpenResponses specification, see the [OpenResponses GitHub repository](https://github.com/openresponsesspec/openresponses). For OpenAI response types, see the [OpenAI Python SDK](https://github.com/openai/openai-python).",
        ]
    )
    return "\n".join(parts) + "\n"


def generate_types_messages_page() -> str:
    """Generate api/types/messages.md."""
    parts = [
        "---",
        "title: Messages Types",
        "description: Data models for the Anthropic Messages API",
        "---",
        "",
        "The Messages API types are Pydantic models used by `any_llm.api.messages()` and `any_llm.api.amessages()`.",
        "",
        "## Primary Types",
        "",
        "### `MessageResponse`",
        "",
        "Full response from the Messages API.",
        "",
        "**Import:** `from any_llm.types.messages import MessageResponse`",
        "",
    ]

    table = _pydantic_field_table(messages_types.MessageResponse)
    if table:
        parts.append(table)

    parts.extend(
        [
            "",
            "### `MessageContentBlock`",
            "",
            "Content block in a Messages API response.",
            "",
            "**Import:** `from any_llm.types.messages import MessageContentBlock`",
            "",
        ]
    )

    table = _pydantic_field_table(messages_types.MessageContentBlock)
    if table:
        parts.append(table)

    parts.extend(
        [
            "",
            "### `MessageUsage`",
            "",
            "Token usage information for Messages API.",
            "",
            "**Import:** `from any_llm.types.messages import MessageUsage`",
            "",
        ]
    )

    table = _pydantic_field_table(messages_types.MessageUsage)
    if table:
        parts.append(table)

    parts.extend(
        [
            "",
            "### `MessageStreamEvent`",
            "",
            "Union of Anthropic SDK stream event types, re-exported from the `anthropic` package:",
            "",
            "- `MessageStartEvent` — `type: 'message_start'`, `message: Message`",
            "- `MessageDeltaEvent` — `type: 'message_delta'`, `delta: Delta`, `usage: MessageDeltaUsage`",
            "- `MessageStopEvent` — `type: 'message_stop'`",
            "- `ContentBlockStartEvent` — `type: 'content_block_start'`, `index: int`, `content_block: ContentBlock`",
            "- `ContentBlockDeltaEvent` — `type: 'content_block_delta'`, `index: int`, `delta: RawContentBlockDelta`",
            "- `ContentBlockStopEvent` — `type: 'content_block_stop'`, `index: int`",
            "",
            "**Import:** `from any_llm.types.messages import MessageStreamEvent`",
            "",
        ]
    )

    parts.extend(
        [
            "",
            "## Internal Types",
            "",
            "### `MessagesParams`",
            "",
            "Normalized parameters for the Anthropic Messages API, used internally to pass structured parameters from the public API to provider implementations.",
            "",
            "**Import:** `from any_llm.types.messages import MessagesParams`",
            "",
        ]
    )

    table = _pydantic_field_table(messages_types.MessagesParams)
    if table:
        parts.append(table)

    return "\n".join(parts) + "\n"


def generate_types_model_page() -> str:
    """Generate api/types/model.md."""
    parts = [
        "---",
        "title: Model Types",
        "description: Data models for model operations",
        "---",
        "",
        "The `Model` type represents a single model returned by `any_llm.list_models()` and `any_llm.alist_models()`.",
        "",
        "## `Model`",
        "",
        "Re-exported from `openai.types.model.Model`.",
        "",
        "**Import:** `from any_llm.types.model import Model`",
        "",
        """| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | The model identifier (e.g., `"gpt-4.1-mini"`, `"mistral-small-latest"`). |
| `created` | `int` | Unix timestamp (seconds) of when the model was created. |
| `object` | `str` | Always `"model"`. |
| `owned_by` | `str` | The organization that owns the model. |""",
        "",
        "## Usage",
        "",
        """```python
from any_llm import list_models

models = list_models("openai")
for model in models:
    print(f"{model.id} (owned by {model.owned_by})")
```""",
        "",
        '{% hint style="info" %}',
        "The `Model` type is a direct re-export from the OpenAI SDK. any-llm normalizes all provider responses into this format so you get a consistent interface regardless of which provider you query.",
        "{% endhint %}",
    ]
    return "\n".join(parts) + "\n"


def generate_types_provider_page() -> str:
    """Generate api/types/provider.md."""
    parts = [
        "---",
        "title: Provider Types",
        "description: Data models for provider operations",
        "---",
        "",
        "The `ProviderMetadata` type describes a provider's capabilities and configuration. It is returned by `AnyLLM.get_provider_metadata()` and `AnyLLM.get_all_provider_metadata()`.",
        "",
        "## `ProviderMetadata`",
        "",
        "A Pydantic `BaseModel` containing provider information and feature flags.",
        "",
        "**Import:** `from any_llm.types.provider import ProviderMetadata`",
        "",
    ]

    table = _pydantic_field_table(provider_types.ProviderMetadata)
    if table:
        parts.append(table)
    else:
        parts.append(
            """| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Provider identifier (e.g., `"openai"`, `"anthropic"`). Matches the provider directory name. |
| `env_key` | `str` | Environment variable name for the API key (e.g., `"OPENAI_API_KEY"`). |
| `env_api_base` | `str \\| None` | Environment variable for overriding the API base URL, if supported. |
| `doc_url` | `str` | Link to the provider's documentation. |
| `class_name` | `str` | Internal Python class name (e.g., `"OpenaiProvider"`). |
| `streaming` | `bool` | Whether the provider supports streaming completions. |
| `image` | `bool` | Whether the provider supports image inputs in completions. |
| `pdf` | `bool` | Whether the provider supports PDF inputs in completions. |
| `embedding` | `bool` | Whether the provider supports the Embedding API. |
| `reasoning` | `bool` | Whether the provider supports reasoning/thinking traces. |
| `responses` | `bool` | Whether the provider supports the Responses API. |
| `completion` | `bool` | Whether the provider supports the Completion API. |
| `messages` | `bool` | Whether the provider supports the Messages API. |
| `list_models` | `bool` | Whether the provider supports listing available models. |
| `batch_completion` | `bool` | Whether the provider supports the Batch API. |"""
        )

    parts.extend(
        [
            "",
            "## Usage",
            "",
            "### Single provider",
            "",
            """```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai")
meta = llm.get_provider_metadata()

print(f"Provider: {meta.name}")
print(f"API key env var: {meta.env_key}")
print(f"Supports streaming: {meta.streaming}")
print(f"Supports embedding: {meta.embedding}")
print(f"Supports responses: {meta.responses}")
```""",
            "",
            "### All providers",
            "",
            """```python
from any_llm import AnyLLM

for meta in AnyLLM.get_all_provider_metadata():
    features = []
    if meta.streaming:
        features.append("streaming")
    if meta.embedding:
        features.append("embedding")
    if meta.reasoning:
        features.append("reasoning")
    if meta.responses:
        features.append("responses")
    print(f"{meta.name}: {', '.join(features) or 'completion only'}")
```""",
        ]
    )
    return "\n".join(parts) + "\n"


def generate_types_batch_page() -> str:
    """Generate api/types/batch.md."""
    parts = [
        "---",
        "title: Batch Types",
        "description: Data models for batch operations",
        "---",
        "",
        "The `Batch` type represents a batch job returned by the [Batch API](../batch.md) functions.",
        "",
        "## `Batch`",
        "",
        "Re-exported from `openai.types.Batch`.",
        "",
        "**Import:** `from any_llm.types.batch import Batch`",
        "",
        """| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique batch identifier. |
| `object` | `str` | Always `"batch"`. |
| `endpoint` | `str` | The API endpoint used for all requests in the batch. |
| `input_file_id` | `str` | ID of the uploaded input file. |
| `completion_window` | `str` | Time frame for batch processing (e.g., `"24h"`). |
| `status` | `str` | Current status: `"validating"`, `"in_progress"`, `"finalizing"`, `"completed"`, `"failed"`, `"expired"`, `"cancelling"`, or `"cancelled"`. |
| `output_file_id` | `str \\| None` | ID of the output file (available when status is `"completed"`). |
| `error_file_id` | `str \\| None` | ID of the error file (if any requests failed). |
| `created_at` | `int` | Unix timestamp of batch creation. |
| `in_progress_at` | `int \\| None` | Unix timestamp of when processing started. |
| `expires_at` | `int \\| None` | Unix timestamp of when the batch expires. |
| `finalizing_at` | `int \\| None` | Unix timestamp of when finalization started. |
| `completed_at` | `int \\| None` | Unix timestamp of completion. |
| `failed_at` | `int \\| None` | Unix timestamp of failure. |
| `expired_at` | `int \\| None` | Unix timestamp of expiration. |
| `cancelling_at` | `int \\| None` | Unix timestamp of cancellation request. |
| `cancelled_at` | `int \\| None` | Unix timestamp of cancellation completion. |
| `request_counts` | `BatchRequestCounts \\| None` | Counts of total, completed, and failed requests. |
| `metadata` | `dict[str, str] \\| None` | Custom metadata attached to the batch. |""",
        "",
        "## `BatchRequestCounts`",
        "",
        "Re-exported from `openai.types.batch_request_counts.BatchRequestCounts`.",
        "",
        "**Import:** `from any_llm.types.batch import BatchRequestCounts`",
        "",
        """| Field | Type | Description |
|-------|------|-------------|
| `total` | `int` | Total number of requests in the batch. |
| `completed` | `int` | Number of completed requests. |
| `failed` | `int` | Number of failed requests. |""",
        "",
        "## Usage",
        "",
        """```python
from any_llm import create_batch, retrieve_batch

batch = create_batch(
    provider="openai",
    input_file_path="requests.jsonl",
    endpoint="/v1/chat/completions",
)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.status}")

# Poll for completion
import time
while batch.status not in ("completed", "failed", "expired", "cancelled"):
    time.sleep(30)
    batch = retrieve_batch("openai", batch.id)
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

if batch.status == "completed":
    print(f"Output file: {batch.output_file_id}")
```""",
        "",
        '{% hint style="info" %}',
        "The `Batch` and `BatchRequestCounts` types are direct re-exports from the OpenAI SDK. any-llm normalizes all provider batch responses into this format.",
        "{% endhint %}",
    ]
    return "\n".join(parts) + "\n"


# Map of output file paths to generator functions
PAGES: dict[str, Any] = {
    "any-llm.md": generate_any_llm_page,
    "completion.md": generate_completion_page,
    "responses.md": generate_responses_page,
    "embedding.md": generate_embedding_page,
    "list-models.md": generate_list_models_page,
    "batch.md": generate_batch_page,
    "exceptions.md": generate_exceptions_page,
    "messages.md": generate_messages_page,
    "types/completion.md": generate_types_completion_page,
    "types/responses.md": generate_types_responses_page,
    "types/messages.md": generate_types_messages_page,
    "types/model.md": generate_types_model_page,
    "types/provider.md": generate_types_provider_page,
    "types/batch.md": generate_types_batch_page,
}


def main() -> int:
    """Generate all API reference documentation pages."""
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_API_DIR / "types").mkdir(parents=True, exist_ok=True)

    for rel_path, generator in PAGES.items():
        output_path = DOCS_API_DIR / rel_path
        content = generator()
        output_path.write_text(content, encoding="utf-8")
        print(f"  {output_path}")

    print(f"\n✓ Generated {len(PAGES)} API reference pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
