import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from mktestdocs import check_md_file


@pytest.mark.parametrize(
    "doc_file",
    list(pathlib.Path("docs").glob("**/*.md")),
    ids=str,
)
def test_all_docs(doc_file: pathlib.Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
    with (
        patch("any_llm.provider.ProviderFactory.create_provider"),
    ):
        check_md_file(fpath=doc_file, memory=True)
