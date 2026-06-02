import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

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
        cookbooks_dir / "empty_notebook.ipynb",
        [],
    )
    _write_notebook(
        cookbooks_dir / "fallback_title.ipynb",
        [{"cell_type": "markdown", "metadata": {}, "source": ["This is not a heading.\n"]}],
    )

    monkeypatch.setattr(convert_to_gitbook, "DOCS_SRC", docs_dir)

    assert convert_to_gitbook.cookbook_summary_entries() == [
        "* [Browser-Use with Any-LLM](cookbooks/browser-use-with-any-llm.md)",
        "* [Empty Notebook](cookbooks/empty-notebook.md)",
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
    (docs_dir / ".gitignore").write_text("cookbooks/*.md\n", encoding="utf-8")

    _write_notebook(
        cookbooks_dir / "browser_use_with_any_llm.ipynb",
        [{"cell_type": "markdown", "metadata": {}, "source": ["# Browser-Use with Any-LLM\n"]}],
    )
    (cookbooks_dir / "browser-use-with-any-llm.md").write_text("# Generated cookbook\n", encoding="utf-8")

    monkeypatch.setattr(convert_to_gitbook, "DOCS_SRC", docs_dir)
    monkeypatch.setattr(convert_to_gitbook, "SITE_DIR", site_dir)
    monkeypatch.setattr(convert_to_gitbook, "run_generator", lambda script_name: None)

    convert_to_gitbook.main()

    summary = (site_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "* [Browser-Use with Any-LLM](cookbooks/browser-use-with-any-llm.md)" in summary
    assert (site_dir / "cookbooks" / "browser-use-with-any-llm.md").exists()
    assert not (site_dir / "cookbooks" / "browser_use_with_any_llm.ipynb").exists()
    assert not (site_dir / ".gitignore").exists()


def test_build_summary_falls_back_when_no_cookbooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(convert_to_gitbook, "cookbook_summary_entries", list)

    summary = convert_to_gitbook.build_summary()

    assert "* [Cookbooks](cookbooks/)" in summary


def test_run_generator_invokes_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_run = Mock()

    monkeypatch.setattr(convert_to_gitbook, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", mock_run)

    convert_to_gitbook.run_generator("generate_docs.py")

    mock_run.assert_called_once_with(
        [sys.executable, str(tmp_path / "generate_docs.py")],
        check=True,
    )
