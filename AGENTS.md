# Repository Guidelines

Every `CLAUDE.md` in this repo is a symlink to the `AGENTS.md` beside it. Always edit `AGENTS.md` directly; never modify a `CLAUDE.md`, and never create one that holds content.

Put guidance in the narrowest file that covers it: `docs/AGENTS.md` for documentation, a skill under `.claude/skills/` for a multi-step procedure, and this file only for what applies repo-wide. Link to the source of truth (CONTRIBUTING.md, config, scripts) instead of restating it; a fact kept in two places goes stale in one of them.

## Where to Look First

- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup, commands, test matrix, and contribution workflow.

## Coding Style & Naming Conventions

- Provider code lives under `src/any_llm/providers/<provider>/` (keep provider-specific behavior isolated there).
- **Override decorator**: When overriding methods from base classes (like `AnyLLM`), always use the `@override` decorator from `typing_extensions`. This is enforced by mypy's `explicit-override` error code. For static methods, the order is `@staticmethod` followed by `@override`.
- Prefer direct attribute access (e.g., `obj.field`) over `getattr(obj, "field")` when the field is typed. This enables `ruff` and `mypy` to catch errors at lint time. Only use `getattr`/`setattr` when working with truly dynamic attributes or when type information is unavailable.
- Comment only what the code cannot say: a non-obvious reason, a workaround, or a constraint. Narration of the change and of prior behavior belongs in the commit message.
- In prose (docs, comments, docstrings, commit messages, PR descriptions), separate clauses with commas, semicolons, colons, parentheses, or periods instead of em dashes or `--`. Code, CLI flags such as `--all-extras`, and en-dash ranges such as `3–4` are unaffected.

## Testing Guidelines

- Add/adjust tests with every change (happy path + error cases). Integration tests should `pytest.skip(...)` when credentials/services aren’t available.
- New code should target ~85%+ coverage (see `CONTRIBUTING.md`). Write tests for every branch in new code, including error/raise paths and edge cases, so that patch coverage passes in CI.
- Do not use class-based test grouping (`class TestFoo:`). All tests should be standalone functions.
- Do not add decorative section-separator comments (e.g., `# -----------` banners). Well-named test functions and natural file ordering are sufficient.
- Place imports at the top of test files unless the import is for an optional dependency that may not be installed (e.g., provider-specific SDKs like `mistralai`, `cohere`). In that case, inline imports inside the test function are acceptable to avoid breaking the entire file.
- Retries apply only to `tests/integration`, where provider overload and rate limits cause transient failures; `tests/integration/conftest.py` marks those tests `flaky`. Unit and docs tests run once, so a failure there is a real failure.
- When the integration suite is red, follow the `integration-test-triage` skill (`.claude/skills/integration-test-triage/SKILL.md`) before assuming a regression.
- The dataclass/dict structured-output path (`parse_responses_output`) is separate from the Pydantic `responses.parse()` path; a bug can hit one and not the other, so test both.

## Commit & Pull Request Guidelines

- Commits follow the project’s history: Conventional Commits such as `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should follow [.github/pull_request_template.md](.github/pull_request_template.md): clear description, linked issues (e.g., `Fixes #123`), completed checklist, and AI-usage disclosure when applicable.

## Definition of done

Before requesting review, every PR must clear:

1. **`uv run pre-commit run --all-files`** clean (ruff lint + format, `mypy` strict). Don't drop `# type: ignore` based on local mypy; CI `run-linter` is authoritative.
2. **`uv run pytest tests/unit`** green, with tests for every branch in changed code (~85% patch coverage; Codecov gates it).
3. **Integration tests for any provider/feature you touched**, run with real keys locally or via the `run-integration-tests` label on the PR. Don't claim a provider works without running it.
4. **Every skip and fix is root-caused**, with a concrete reason (HTTP status, deprecated model, missing CI infra), never "flaky" or "does not reliably support".

## Mypy and Provider SDKs

- Provider SDKs are optional and may not be installed locally. When missing, mypy treats their types as `Any`, which makes `# type: ignore` comments appear "unused" even though they suppress real errors in CI.
- Do not remove `# type: ignore` comments based on local mypy output. CI (`run-linter`) is the authoritative environment.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
