import json
from pathlib import Path

import scripts.generate_cookbooks as generate_cookbooks


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_notebook_to_mdx_uses_python_fence_for_valid_async_function(tmp_path: Path) -> None:
    notebook_path = tmp_path / "valid_async.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Valid Async\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "async def run_demo():\n",
                    "    await thing()\n",
                ],
            },
        ],
    )

    content = generate_cookbooks.notebook_to_mdx(notebook_path)

    assert "```python\nasync def run_demo():\n    await thing()\n\n```" in content


def test_notebook_to_mdx_uses_plain_fence_for_top_level_await(tmp_path: Path) -> None:
    notebook_path = tmp_path / "top_level_await.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Top Level Await\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "await llm.ainvoke([])\n",
                ],
            },
        ],
    )

    content = generate_cookbooks.notebook_to_mdx(notebook_path)

    assert "```\nawait llm.ainvoke([])\n\n```" in content
    assert "```python\nawait llm.ainvoke([])\n\n```" not in content
