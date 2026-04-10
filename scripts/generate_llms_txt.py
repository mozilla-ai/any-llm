#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt files after the Starlight docs build.

These files follow the llmstxt.org standard for making documentation
accessible to AI systems.

Usage:
    python scripts/generate_llms_txt.py
"""

import os
import re
import sys
from pathlib import Path

DOCS_CONTENT_DIR = Path(__file__).parent.parent / "docs"
BUILD_OUTPUT_DIR = Path(__file__).parent.parent / "docs"
BASE_URL = "https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/gitbook-docs/"
MARKDOWN_EXTENSION = ".md"
MDX_EXTENSION = ".mdx"
ENCODING = "utf-8"
TOC_PATTERN = r"^\s*\[\[TOC\]\]\s*$"
MARKDOWN_LINK_PATTERN = r"\[([^\]]+)\]\(([^)]+\.md)\)"
MARKDOWN_LINK_REPLACEMENT = r"[\1](#\2)"

ORDERED_FILES = [
    "index.mdx",
    "quickstart.md",
    "providers.md",
    "api/any-llm.md",
    "api/responses.md",
    "api/completion.md",
    "api/embedding.md",
    "api/messages.md",
    "api/exceptions.md",
    "api/list-models.md",
    "api/batch.md",
    "api/types/completion.md",
    "api/types/responses.md",
    "api/types/messages.md",
    "api/types/model.md",
    "api/types/provider.md",
    "api/types/batch.md",
    "platform/overview.md",
    "gateway/overview.md",
    "gateway/quickstart.md",
    "gateway/authentication.md",
    "gateway/budget-management.md",
    "gateway/configuration.md",
    "gateway/api-reference.md",
    "gateway/troubleshooting.md",
    "gateway/docker-deployment.md",
]


def create_file_title(file_path: str) -> str:
    """Create a clean title from file path."""
    if file_path in ("index.md", "index.mdx"):
        return "Introduction"

    name = file_path
    for ext in (MDX_EXTENSION, MARKDOWN_EXTENSION):
        name = name.replace(ext, "")
    return name.replace("_", " ").replace("-", " ").replace("/", " - ").title()


def extract_description_from_markdown(content: str) -> str:
    """Extract a description from markdown content."""
    if not content:
        return ""

    lines = content.split("\n")
    title_found = False
    in_frontmatter = False

    for line in lines:
        stripped = line.strip()

        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue

        if not stripped:
            continue

        if stripped.startswith("# ") and not title_found:
            title_found = True
            continue

        if not title_found:
            continue

        if (
            stripped.startswith(("!!! ", "<", ":::", "##", "```", "---", "|", "- ", "* ", "import "))
            or (stripped.startswith("[") and stripped.endswith("]"))
            or re.match(r"^\d+\.", stripped)
        ):
            continue

        if len(stripped) > 20:
            description = stripped
            description = re.sub(r"\*\*([^*]+)\*\*", r"\1", description)
            description = re.sub(r"\*([^*]+)\*", r"\1", description)
            description = re.sub(r"`([^`]+)`", r"\1", description)
            return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

    return ""


def clean_markdown_content(content: str, file_path: str) -> str:
    """Clean markdown content for concatenation."""
    # Remove frontmatter
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            content = content[end_idx + 3 :].lstrip()

    # Remove Astro imports
    content = re.sub(r"^import\s+.*$", "", content, flags=re.MULTILINE)

    content = re.sub(TOC_PATTERN, "", content, flags=re.MULTILINE)
    content = re.sub(MARKDOWN_LINK_PATTERN, MARKDOWN_LINK_REPLACEMENT, content)

    return f"<!-- Source: {file_path} -->\n\n{content}"


def get_all_content_files() -> list[str]:
    """Get ordered list of content files, including any not in the explicit list."""
    ordered = list(ORDERED_FILES)

    for root, _, files in os.walk(DOCS_CONTENT_DIR):
        for f in files:
            if f.endswith((MARKDOWN_EXTENSION, MDX_EXTENSION)):
                rel_path = os.path.relpath(os.path.join(root, f), DOCS_CONTENT_DIR)
                if rel_path not in ordered:
                    ordered.append(rel_path)

    return ordered


def generate_llms_txt(ordered_files: list[str]) -> str:
    """Generate llms.txt content."""
    lines = ["# any-llm", "", "## Docs", ""]

    for file_path in ordered_files:
        full_path = DOCS_CONTENT_DIR / file_path
        if not full_path.exists():
            continue

        txt_url = f"{BASE_URL}{file_path}"
        title = create_file_title(file_path)
        content = full_path.read_text(encoding=ENCODING)
        description = extract_description_from_markdown(content)

        if description:
            lines.append(f"- [{title}]({txt_url}): {description}")
        else:
            lines.append(f"- [{title}]({txt_url})")

    return "\n".join(lines)


def generate_llms_full_txt(ordered_files: list[str]) -> str:
    """Generate llms-full.txt by concatenating all documentation."""
    sections = [
        "# any-llm Documentation",
        "",
        "> Complete documentation for any-llm - A Python library providing a single interface to different LLM providers.",
        "",
        "This file contains all documentation pages concatenated for easy consumption by AI systems.",
        "",
        "---",
        "",
    ]

    for file_path in ordered_files:
        full_path = DOCS_CONTENT_DIR / file_path
        if not full_path.exists():
            continue

        content = full_path.read_text(encoding=ENCODING)
        cleaned = clean_markdown_content(content, file_path)
        sections.extend([f"## {file_path}", "", cleaned, "", "---", ""])

    return "\n".join(sections)


def main() -> int:
    """Generate llms.txt and llms-full.txt files."""
    ordered_files = get_all_content_files()

    BUILD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llms_txt = generate_llms_txt(ordered_files)
    llms_txt_path = BUILD_OUTPUT_DIR / "llms.txt"
    llms_txt_path.write_text(llms_txt, encoding=ENCODING)
    print(f"✓ Generated {llms_txt_path}")

    llms_full_txt = generate_llms_full_txt(ordered_files)
    llms_full_path = BUILD_OUTPUT_DIR / "llms-full.txt"
    llms_full_path.write_text(llms_full_txt, encoding=ENCODING)
    print(f"✓ Generated {llms_full_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
