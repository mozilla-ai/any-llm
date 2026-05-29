"""Build the GitBook site output into site/.

Runs all doc generators, copies docs/ into site/, and writes SUMMARY.md.
The contents of site/ are pushed to the gitbook-docs branch by CI.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DOCS_SRC = Path("docs")
SITE_DIR = Path("site")
SITE_IGNORE_PATTERNS = ("*.ipynb",)


def run_generator(script_name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)], check=True)  # noqa: S603


def cookbook_summary_entries() -> list[str]:
    """Build summary entries for cookbook notebooks."""
    entries: list[str] = []
    cookbooks_dir = DOCS_SRC / "cookbooks"
    for notebook_path in sorted(cookbooks_dir.glob("*.ipynb")):
        title = notebook_path.stem.replace("_", " ").title()
        with notebook_path.open(encoding="utf-8") as notebook_file:
            notebook = json.load(notebook_file)

        cells = notebook.get("cells", [])
        if cells:
            first_cell_source = "".join(cells[0].get("source", []))
            if first_cell_source.startswith("# "):
                title = first_cell_source.lstrip("# ").strip()

        slug = notebook_path.stem.replace("_", "-") + ".md"
        entries.append(f"* [{title}](cookbooks/{slug})")

    return entries


def build_summary() -> str:
    """Build the GitBook summary."""
    cookbook_entries = cookbook_summary_entries()
    cookbook_section = "\n".join(cookbook_entries) if cookbook_entries else "* [Cookbooks](cookbooks/)"

    return f"""\
# Table of Contents

* [Introduction](index.md)
* [Quickstart](quickstart.md)
* [Providers](providers.md)

## Cookbooks

{cookbook_section}

## API Reference

* [AnyLLM](api/any-llm.md)
* [Responses](api/responses.md)
* [Completion](api/completion.md)
* [Embedding](api/embedding.md)
* [Messages](api/messages.md)
* [Exceptions](api/exceptions.md)
* [List Models](api/list-models.md)
* [Batch](api/batch.md)
* [Types](api/types/completion.md)
  * [Completion](api/types/completion.md)
  * [Responses](api/types/responses.md)
  * [Messages](api/types/messages.md)
  * [Model](api/types/model.md)
  * [Provider](api/types/provider.md)
  * [Batch](api/types/batch.md)

"""


def copy_docs_to_site() -> None:
    """Copy docs into the site artifact, excluding notebook source files."""
    shutil.copytree(DOCS_SRC, SITE_DIR, ignore=shutil.ignore_patterns(*SITE_IGNORE_PATTERNS))


def main() -> None:
    for script_name in (
        "generate_api_docs.py",
        "generate_provider_table.py",
        "generate_cookbooks.py",
    ):
        run_generator(script_name)

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    copy_docs_to_site()

    (SITE_DIR / "SUMMARY.md").write_text(build_summary(), encoding="utf-8")
    print(f"\nDone - {len(list(SITE_DIR.rglob('*.md')))} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
