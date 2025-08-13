# Contributing to mozilla.ai any-llm

Thank you for your interest in contributing to this repository! This project supports the mozilla.ai goal of empowering developers to integrate AI capabilities into their projects using open-source tools and models.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

## **Guidelines for Contributions**

**Install**

We recommend to use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
source .venv/bin/activate
uv sync --dev --extra all
```

**Lint**

Ensure all the checks pass:

```bash
pre-commit run --all-files
```

**Tests**

Test changes locally to ensure functionality.

```bash
pytest -v tests
```

**Docs**

Update docs for changes to functionality and maintain consistency with existing docs.

```bash
mkdocs serve
```
