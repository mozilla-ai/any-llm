#!/usr/bin/env python3
"""Generate docs/providers.md from provider metadata in the any-llm source code.

providers.md is a build artifact - it is not committed to the repository.
Run this script (or scripts/convert_to_gitbook.py) to regenerate it locally.

Usage:
    python scripts/generate_provider_table.py
"""

import sys
from pathlib import Path

from any_llm import AnyLLM
from any_llm.constants import ProviderTier
from any_llm.types.provider import ProviderMetadata

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROVIDERS_FILE = DOCS_DIR / "providers.md"

PROVIDERS_PREAMBLE = """\
---
title: Supported Providers
description: Complete list of LLM providers supported by any-llm including OpenAI, Anthropic, Mistral, and more
---

`any-llm` supports multiple providers. Provider source code is in [`src/any_llm/providers/`](https://github.com/mozilla-ai/any-llm/tree/main/src/any_llm/providers).

Is your endpoint OpenAI-compatible but not listed below? You are not blocked: use [`AnyLLM.create_openai_compatible`](quickstart.md#custom-openai-compatible-endpoints). Prefer it over pointing the `openai` provider at a custom `api_base`, which misreports the provider identity as `openai`, silently sends any `OPENAI_API_KEY` in your environment to the custom endpoint, and rejects keyless local servers.

## Support tiers

The **Tier** column is a support promise, not a statement about how the provider is implemented:

- ✅ **Verified**: we hold an API key, integration tests run against the provider in CI, and we fix breakage.
- 🤝 **Community**: verified live by the contributor when it was added, then community-maintained. No CI key, so its integration tests skip in CI. Report breakage by opening an issue.

A provider can be verified whether it ships as a code folder or as a single config row, and a config-only gateway we hold a key for is verified.

"""


def generate_provider_table(providers: list[ProviderMetadata]) -> str:
    """Generate a markdown table from provider metadata."""
    if not providers:
        return "No providers found."

    table_lines = [
        "| ID | Tier | Key | Base | Responses | Completion | Streaming<br>(Completions) | Reasoning<br>(Completions) | Image <br>(Completions) | Embedding | List Models | Batch |",
        "|----|------|-----|------|-----------|------------|--------------------------|--------------------------|-----------|-----------|-------------|-------|",
    ]

    for provider in providers:
        env_key = provider.env_key
        env_api_base = provider.env_api_base or ""

        provider_key = provider.name
        provider_id_link = f"[`{provider_key}`]({provider.doc_url})"

        stream_supported = "✅" if provider.streaming else "❌"
        image_supported = "✅" if provider.image else "❌"
        embedding_supported = "✅" if provider.embedding else "❌"
        reasoning_supported = "✅" if provider.reasoning else "❌"
        responses_supported = "✅" if provider.responses else "❌"
        completion_supported = "✅" if provider.completion else "❌"
        list_models_supported = "✅" if provider.list_models else "❌"
        batch_supported = "✅" if provider.batch_completion else "❌"

        tier_label = "✅ Verified" if provider.tier is ProviderTier.VERIFIED else "🤝 Community"

        row = (
            f"| {provider_id_link} | {tier_label} | {env_key} | {env_api_base} | {responses_supported} | "
            f"{completion_supported} | {stream_supported} | {reasoning_supported} | {image_supported} | "
            f"{embedding_supported} | {list_models_supported} | {batch_supported} |"
        )
        table_lines.append(row)

    return "\n".join(table_lines)


def main() -> int:
    """Generate docs/providers.md."""
    provider_metadata = AnyLLM.get_all_provider_metadata()
    table = generate_provider_table(provider_metadata)
    content = PROVIDERS_PREAMBLE + table + "\n"

    PROVIDERS_FILE.write_text(content, encoding="utf-8")
    print(f"✓ Provider table written to {PROVIDERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
