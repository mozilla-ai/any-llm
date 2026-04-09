"""Build the GitBook site output into site/.

Runs all doc generators, copies docs/ into site/, and writes SUMMARY.md.
The contents of site/ are pushed to the gitbook-docs branch by CI.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DOCS_SRC = Path("docs")
SITE_DIR = Path("site")

SUMMARY = """\
# Table of Contents

* [Introduction](index.md)
* [Quickstart](quickstart.md)
* [Providers](providers.md)

## Cookbooks

* [Getting Started](cookbooks/any-llm-getting-started.md)

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

## Managed Platform

* [Overview](platform/overview.md)

## Gateway

* [Overview](gateway/overview.md)
* [Quick Start](gateway/quickstart.md)
* [Authentication](gateway/authentication.md)
* [Budget Management](gateway/budget-management.md)
* [Configuration](gateway/configuration.md)
* [API Reference](gateway/api-reference.md)
* [Troubleshooting](gateway/troubleshooting.md)
* [Docker Deployment](gateway/docker-deployment.md)
"""


def run_generator(script_name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)], check=True)


def main() -> None:
    for script_name in (
        "generate_openapi.py",
        "generate_api_docs.py",
        "generate_provider_table.py",
        "generate_cookbooks.py",
        "generate_llms_txt.py",
    ):
        run_generator(script_name)

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    shutil.copytree(DOCS_SRC, SITE_DIR)

    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY, encoding="utf-8")
    print(f"\nDone - {len(list(SITE_DIR.rglob('*.md')))} files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
