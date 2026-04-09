#!/usr/bin/env python3
"""Generate the provider compatibility table for the Starlight docs site.

This script reads provider metadata from the any-llm source code and injects
a markdown table into the providers.md documentation page. It replaces content
between PROVIDER-TABLE-START and PROVIDER-TABLE-END markers.

Usage:
    python scripts/generate_provider_table.py          # Generate and inject
    python scripts/generate_provider_table.py --check   # Verify table is up-to-date
"""

import argparse
import sys
from pathlib import Path

from any_llm import AnyLLM
from any_llm.types.provider import ProviderMetadata

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROVIDERS_FILE = DOCS_DIR / "providers.md"
START_MARKER = "<!-- PROVIDER-TABLE-START -->"
END_MARKER = "<!-- PROVIDER-TABLE-END -->"


def generate_provider_table(providers: list[ProviderMetadata]) -> str:
    """Generate a markdown table from provider metadata."""
    if not providers:
        return "No providers found."

    table_lines = [
        "| ID | Key | Base | Responses | Completion | Streaming<br>(Completions) | Reasoning<br>(Completions) | Image <br>(Completions) | Embedding | List Models | Batch |",
        "|----|-----|------|-----------|------------|--------------------------|--------------------------|-----------|-----------|-------------|-------|",
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

        row = (
            f"| {provider_id_link} | {env_key} | {env_api_base} | {responses_supported} | {completion_supported} | "
            f"{stream_supported} | {reasoning_supported} | {image_supported} | {embedding_supported} | "
            f"{list_models_supported} | {batch_supported} |"
        )
        table_lines.append(row)

    return "\n".join(table_lines)


def inject_table(file_path: Path, table: str) -> str:
    """Inject the provider table into the markdown file between markers."""
    content = file_path.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        print(f"Error: markers not found in {file_path}", file=sys.stderr)
        sys.exit(1)

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    return content[: start_idx + len(START_MARKER)] + "\n" + table + "\n" + content[end_idx:]


def main() -> int:
    """Generate or check the provider compatibility table."""
    parser = argparse.ArgumentParser(description="Generate provider compatibility table")
    parser.add_argument("--check", action="store_true", help="Check if table is up-to-date")

    args = parser.parse_args()

    provider_metadata = AnyLLM.get_all_provider_metadata()
    table = generate_provider_table(provider_metadata)
    new_content = inject_table(PROVIDERS_FILE, table)

    if args.check:
        current_content = PROVIDERS_FILE.read_text(encoding="utf-8")
        if current_content != new_content:
            print("Provider table is out of date.", file=sys.stderr)
            print("Run 'python scripts/generate_provider_table.py' to update it.", file=sys.stderr)
            return 1
        print("✓ Provider table is up to date")
        return 0

    PROVIDERS_FILE.write_text(new_content, encoding="utf-8")
    print(f"✓ Provider table written to {PROVIDERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
