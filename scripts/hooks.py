"""Generate llms.txt and llms-full.txt files for the documentation.

It also serves markdown files as text files for LLM consumption.
It was initially written in collaboration with Claude 4 Sonnet.
"""

import asyncio
import os
import re
from pathlib import Path

import httpx

from any_llm.provider import ProviderFactory
from any_llm.types.provider import ProviderMetadata


# Exceptions
class UrlValidationError(Exception):
    """Exception raised when a URL is invalid."""


# Constants
MARKDOWN_EXTENSION = ".md"
BASE_URL = "https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/main/docs/"
ENCODING = "utf-8"
EXCLUDED_DIRS = {".", "__pycache__"}
TOC_PATTERN = r"^\s*\[\[TOC\]\]\s*$"
MARKDOWN_LINK_PATTERN = r"\[([^\]]+)\]\(([^)]+\.md)\)"
MARKDOWN_LINK_REPLACEMENT = r"[\1](#\2)"


async def validate_url(urls, http_timeout=10):
    """Validate that all URLs are valid."""
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        responses = await asyncio.gather(*[client.head(url, follow_redirects=True) for url in urls])
        for response in responses:
            if response.status_code != 200:
                msg = f"URL {response.url} returned status code {response.status_code}"
                raise UrlValidationError(msg)


def generate_provider_table(providers: list[ProviderMetadata]):
    """Generate a markdown table from provider metadata."""
    if not providers:
        return "No providers found."

    # Create table header
    table_lines = [
        "| ID | Env Var | Source Code | Responses | Completion | Streaming<br>(Completions) | Reasoning<br>(Completions) | Embedding | List Models |",
        "|----|---------|-------------|-----------|------------|--------------------------|--------------------------|-----------|-------------|",
    ]

    # Add rows for each provider
    source_urls = []
    for provider in providers:
        env_key = provider.env_key

        # Use the provider key (directory name) instead of display name
        provider_key = provider.name
        source_url = f"https://github.com/mozilla-ai/any-llm/tree/main/src/any_llm/providers/{provider_key}/"
        source_urls.append(source_url)

        source_link = f"[Source]({source_url})"

        # Create provider ID as a hyperlink to the documentation URL
        provider_id_link = f"[`{provider_key}`]({provider.doc_url})"

        stream_supported = "✅" if provider.streaming else "❌"
        embedding_supported = "✅" if provider.embedding else "❌"
        reasoning_supported = "✅" if provider.reasoning else "❌"
        responses_supported = "✅" if provider.responses else "❌"
        completion_supported = "✅" if provider.completion else "❌"
        list_models_supported = "✅" if provider.list_models else "❌"

        row = (
            f"| {provider_id_link} | {env_key} | {source_link} | {responses_supported} | {completion_supported} | "
            f"{stream_supported} | {reasoning_supported} | {embedding_supported} |"
            f"{list_models_supported}"
        )
        table_lines.append(row)

    asyncio.run(validate_url(source_urls))
    return "\n".join(table_lines)


def inject_provider_table_in_markdown(markdown_content, provider_dir):
    """Inject the provider table into markdown content during build."""
    start_marker = "<!-- AUTO-GENERATED TABLE START -->"
    end_marker = "<!-- AUTO-GENERATED TABLE END -->"

    if start_marker not in markdown_content or end_marker not in markdown_content:
        return markdown_content

    provider_metadata = ProviderFactory.get_all_provider_metadata()
    provider_table = generate_provider_table(provider_metadata)

    start_idx = markdown_content.find(start_marker)
    end_idx = markdown_content.find(end_marker)

    return markdown_content[: start_idx + len(start_marker)] + "\n" + provider_table + "\n" + markdown_content[end_idx:]


def get_nav_files(nav_config):
    """Extract file paths from mkdocs navigation config in order."""
    files = []

    def extract_files(nav_item):
        if isinstance(nav_item, dict):
            for _, value in nav_item.items():
                if isinstance(value, str):
                    # This is a file reference
                    if value.endswith(MARKDOWN_EXTENSION):
                        files.append(value)
                elif isinstance(value, list):
                    # This is a nested section
                    for item in value:
                        extract_files(item)
        elif isinstance(nav_item, list):
            for item in nav_item:
                extract_files(item)

    extract_files(nav_config)
    return files


def get_all_markdown_files(docs_dir):
    """Get all markdown files in the documentation directory."""
    all_md_files = []
    for root, dirs, files in os.walk(docs_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(MARKDOWN_EXTENSION):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                all_md_files.append(rel_path)
    return all_md_files


def get_ordered_files(nav_config, docs_dir):
    """Get ordered list of markdown files based on navigation and additional files."""
    nav_files = get_nav_files(nav_config)
    all_md_files = get_all_markdown_files(docs_dir)

    # Combine nav files with any additional files, maintaining nav order
    ordered_files = []
    for file in nav_files:
        if file in all_md_files:
            ordered_files.append(file)

    # Add any remaining files not in navigation
    for file in all_md_files:
        if file not in ordered_files:
            ordered_files.append(file)

    return ordered_files


def clean_markdown_content(content, file_path):
    """Clean markdown content for better concatenation."""
    # Remove mkdocs-specific directives
    content = re.sub(TOC_PATTERN, "", content, flags=re.MULTILINE)

    # Remove or replace relative links that won't work in concatenated format
    # Convert relative md links to section references where possible
    content = re.sub(MARKDOWN_LINK_PATTERN, MARKDOWN_LINK_REPLACEMENT, content)

    # Add file path as a comment for reference
    return f"<!-- Source: {file_path} -->\n\n{content}"


def extract_description_from_markdown(content):
    """Extract a description from markdown content."""
    if not content:
        return ""

    lines = content.split("\n")
    title_found = False
    description_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Look for the main title (first H1)
        if stripped.startswith("# ") and not title_found:
            title_found = True
            continue

        # Skip if we haven't found the title yet
        if not title_found:
            continue

        # Skip common non-description elements
        if (
            stripped.startswith("!!! ")  # Admonitions  # noqa: PIE810
            or stripped.startswith("<")  # HTML tags
            or stripped.startswith(":::")  # API references
            or stripped.startswith("##")  # Subheadings
            or stripped.startswith("```")  # Code blocks
            or stripped.startswith("---")  # Horizontal rules
            or stripped.startswith("|")  # Tables
            or stripped.startswith("- ")  # Lists
            or stripped.startswith("* ")  # Lists
            or (stripped.startswith("[") and stripped.endswith("]"))  # Standalone links
            or re.match(r"^\d+\.", stripped)
        ):  # Numbered lists
            continue

        # If we found substantial text, use it as description
        if len(stripped) > 20:  # Minimum length for a meaningful description
            description_lines.append(stripped)
            # For now, just take the first good paragraph
            break

    if description_lines:
        # Clean up markdown syntax from the description
        description = " ".join(description_lines)
        # Remove markdown formatting
        description = re.sub(r"\*\*([^*]+)\*\*", r"\1", description)  # Bold
        description = re.sub(r"\*([^*]+)\*", r"\1", description)  # Italic
        description = re.sub(r"`([^`]+)`", r"\1", description)  # Code
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)  # Links

    return ""


def get_file_description(file_path, docs_dir):
    """Get descriptive text extracted from the actual file."""
    full_path = docs_dir / file_path

    if not full_path.exists():
        return ""

    content = read_file_content(full_path)
    if content is None:
        return ""

    return extract_description_from_markdown(content)


def read_file_content(file_path):
    """Safely read file content with error handling."""
    with open(file_path, encoding=ENCODING) as f:
        return f.read()


def write_file_content(file_path, content):
    """Safely write file content with error handling."""
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=ENCODING) as f:
        f.write(content)


def create_file_title(file_path):
    """Create a clean title from file path."""
    if file_path == "index.md":
        return "Introduction"

    return file_path.replace(MARKDOWN_EXTENSION, "").replace("_", " ").replace("/", " - ").title()


def generate_llms_txt(docs_dir, site_dir, nav_config):
    """Generate llms.txt file following llmstxt.org standards."""
    ordered_files = get_ordered_files(nav_config, docs_dir)

    # Generate llms.txt content
    llms_txt_content = []

    # Add header following MCP format
    llms_txt_content.append("# any-llm")
    llms_txt_content.append("")
    llms_txt_content.append("## Docs")
    llms_txt_content.append("")

    # Add individual file entries with better formatting
    for file_path in ordered_files:
        # Convert markdown path to .txt file path with full URL
        txt_url = f"{BASE_URL}{file_path}"

        title = create_file_title(file_path)
        description = get_file_description(file_path, docs_dir)

        # Format entry like MCP, but link to .txt files with full URLs
        if description:
            llms_txt_content.append(f"- [{title}]({txt_url}): {description}")
        else:
            llms_txt_content.append(f"- [{title}]({txt_url})")

    # Write the llms.txt file
    llms_txt_dest = site_dir / "llms.txt"
    write_file_content(llms_txt_dest, "\n".join(llms_txt_content))


def generate_llms_full_txt(docs_dir, site_dir, nav_config):
    """Generate llms-full.txt by concatenating all markdown documentation."""
    ordered_files = get_ordered_files(nav_config, docs_dir)

    # Generate the llms-full.txt content
    llms_full_content = []

    # Add header
    llms_full_content.extend(
        [
            "# any-llm Documentation",
            "",
            "> Complete documentation for any-llm - A Python library providing a single interface to different llm providers.",
            "",
            "This file contains all documentation pages concatenated for easy consumption by AI systems.",
            "",
            "---",
            "",
        ]
    )

    # Process each markdown file
    for file_path in ordered_files:
        full_path = docs_dir / file_path

        if full_path.exists():
            content = read_file_content(full_path)
            if content is not None:
                # Clean and process content
                cleaned_content = clean_markdown_content(content, file_path)

                # Add section separator
                llms_full_content.extend([f"## {file_path}", "", cleaned_content, "", "---", ""])

    # Write the combined content to llms-full.txt
    llms_full_txt_dest = site_dir / "llms-full.txt"
    write_file_content(llms_full_txt_dest, "\n".join(llms_full_content))


def on_post_build(config, **kwargs):
    """Generate llms.txt and llms-full.txt files, and serve markdown as text."""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])

    # Get navigation configuration
    nav_config = config.get("nav", [])

    # Generate llms.txt file
    generate_llms_txt(docs_dir, site_dir, nav_config)

    # Generate complete llms-full.txt
    generate_llms_full_txt(docs_dir, site_dir, nav_config)


def on_page_markdown(markdown, page, config, files):
    """Inject provider table into markdown content during build."""
    docs_dir = Path(config["docs_dir"])
    project_root = docs_dir.parent
    provider_dir = project_root / "src" / "any_llm" / "providers"

    # Process the markdown content
    return inject_provider_table_in_markdown(markdown, provider_dir)


def on_pre_build(config, **kwargs):
    """Pre-build hook - currently unused but kept for potential future use."""
