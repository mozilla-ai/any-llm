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
    if doc_file.name == "quickstart.md":
        mock_provider = Mock()
        mock_provider.completion.return_value = Mock(choices=[Mock(message=Mock(content="Hello!"))])
        mock_provider.get_provider_metadata.return_value = Mock(streaming=True, completion=True)
        
        mock_embedding_result = Mock()
        mock_embedding_result.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_embedding_result.usage = Mock(total_tokens=10)

        with (
            patch("any_llm.any_llm.AnyLLM.split_model_provider") as mock_split,
            patch("any_llm.any_llm.AnyLLM.create") as mock_create,
            patch("any_llm.completion") as mock_completion,
            patch("any_llm.embedding") as mock_embedding,
            patch("os.environ.get") as mock_env_get,
        ):
            mock_split.return_value = (LLMProvider.OPENAI, "gpt-5")
            mock_create.return_value = mock_provider
            mock_completion.return_value = Mock(choices=[Mock(message=Mock(content="Hello!"), delta=Mock(content="Hello!"))])
            mock_embedding.return_value = mock_embedding_result
            mock_env_get.return_value = "fake-api-key"  # Mock API key environment variables
            check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
    else:
        check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
