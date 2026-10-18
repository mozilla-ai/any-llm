# Documentation Guidelines

`docs/` holds hand-authored GitBook documentation. Agent guidance files (`AGENTS.md`, `CLAUDE.md`) are excluded from the published site by `SITE_IGNORE_PATTERNS` in `scripts/convert_to_gitbook.py`.

- Generated files (`api/`, `providers.md`, `cookbooks/any-llm-getting-started.md`) are build artifacts produced by `scripts/convert_to_gitbook.py` and are not committed to the repository.
- `scripts/generate_api_docs.py` (run by `convert_to_gitbook.py`) derives signatures and Pydantic field tables, but each page's list of sections is hand-written: the method sections on the AnyLLM page and the exception sections on the exceptions page. A new public method or exception appears in the API reference only once a section for it is added to that script.
- The final publish artifact is `site/`, built by CI and pushed to the `gitbook-docs` branch that GitBook watches.
- Build the GitBook site locally with `uv run python scripts/convert_to_gitbook.py` (output in `site/`).
