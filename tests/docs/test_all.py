import pathlib

import pytest
from mktestdocs import check_md_file


@pytest.mark.parametrize(
    "doc_file",
    list(pathlib.Path("docs").glob("**/*.md")),
    ids=str,
)
def test_all_docs(doc_file: pathlib.Path) -> None:
    check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
