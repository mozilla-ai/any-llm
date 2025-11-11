import pathlib
from unittest.mock import Mock, patch

import pytest
from mktestdocs import check_md_file

from any_llm.constants import LLMProvider


@pytest.mark.parametrize(
    "doc_file",
    list(pathlib.Path("docs").glob("**/*.md")),
    ids=str,
)
def test_all_docs(doc_file: pathlib.Path) -> None:
    if doc_file.name == "quickstart.md" and "gateway" in str(doc_file):
        mock_provider = Mock()
        mock_provider.completion.return_value = Mock(choices=[Mock(message=Mock(content="Hello!"))])

        with (
            patch("any_llm.any_llm.AnyLLM.split_model_provider") as mock_split,
            patch("any_llm.any_llm.AnyLLM.create") as mock_create,
        ):
            mock_split.return_value = (LLMProvider.OPENAI, "gpt-5")
            mock_create.return_value = mock_provider
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
    else:
        check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
