# Documentation Guidelines

`docs/` holds hand-authored GitBook documentation in a flat markdown layout.

- Generated files (`api/`, `providers.md`, `cookbooks/any-llm-getting-started.md`) are build artifacts produced by `scripts/convert_to_gitbook.py` and are not committed to the repository.
- The final publish artifact is `site/`, built by CI and pushed to the `gitbook-docs` branch that GitBook watches.
- Build the GitBook site locally with `uv run python scripts/convert_to_gitbook.py` (output in `site/`).
