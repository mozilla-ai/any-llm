import json
from pathlib import Path

import pytest

import scripts.convert_to_gitbook as convert_to_gitbook


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_cookbook_summary_entries_use_notebook_titles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    cookbooks_dir = docs_dir / "cookbooks"
    cookbooks_dir.mkdir(parents=True)

    _write_notebook(
        cookbooks_dir / "browser_use_with_any_llm.ipynb",
        [{"cell_type": "markdown", "metadata": {}, "source": ["# Browser-Use with Any-LLM\n"]}],
    )
    _write_notebook(
        cookbooks_dir / "fallback_title.ipynb",
        [{"cell_type": "markdown", "metadata": {}, "source": ["This is not a heading.\n"]}],
    )

    monkeypatch.setattr(convert_to_gitbook, "DOCS_SRC", docs_dir)

    assert convert_to_gitbook.cookbook_summary_entries() == [
        "* [Browser-Use with Any-LLM](cookbooks/browser-use-with-any-llm.md)",
        "* [Fallback Title](cookbooks/fallback-title.md)",
    ]


def test_main_writes_dynamic_cookbook_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    site_dir = tmp_path / "site"
    cookbooks_dir = docs_dir / "cookbooks"
    platform_dir = docs_dir / "platform"

    cookbooks_dir.mkdir(parents=True)
    platform_dir.mkdir(parents=True)

    for relative_path in ("index.md", "quickstart.md", "providers.md", "platform/overview.md"):
        target = docs_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Placeholder\n", encoding="utf-8")

    _write_notebook(
        cookbooks_dir / "browser_use_with_any_llm.ipynb",
        [{"cell_type": "markdown", "metadata": {}, "source": ["# Browser-Use with Any-LLM\n"]}],
    )

    monkeypatch.setattr(convert_to_gitbook, "DOCS_SRC", docs_dir)
    monkeypatch.setattr(convert_to_gitbook, "SITE_DIR", site_dir)
    monkeypatch.setattr(convert_to_gitbook, "run_generator", lambda script_name: None)

    convert_to_gitbook.main()

    summary = (site_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "* [Browser-Use with Any-LLM](cookbooks/browser-use-with-any-llm.md)" in summary
