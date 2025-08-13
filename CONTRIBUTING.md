# Contributing to mozilla.ai any-llm

Thank you for your interest in contributing to this repository! This project supports the mozilla.ai goal of empowering choice and transparency when it comes to LLM usage and selection.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

## **Guidelines for Contributions**

**Install**

We recommend to use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
source .venv/bin/activate
uv sync --all-extras -U --python=3.13
```

**Lint**

Ensure all the checks pass:

```bash
uv run pre-commit run --all-files --verbose
```

**Tests**

Test changes locally to ensure functionality.

```bash
pytest -v tests/unit
pytest -v tests/integration -n auto

**Docs**

Update docs for changes to functionality and maintain consistency with existing docs.

```bash
mkdocs serve
```
