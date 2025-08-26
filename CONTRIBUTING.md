# Contributing to mozilla.ai any-llm

Thank you for your interest in contributing to this repository! This project supports the mozilla.ai goal of empowering choice and transparency when it comes to LLM usage and selection.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

## **Guidelines for Contributions**

### Ground Rules

- Review issue discussion fully before starting work. Engage in the thread first when an issue is under discussion.
- PRs must build on agreed direction where ones exist. If there is no agreed direction, seek consensus from the core maintainers.
- PRs with "drive-by" unrelated changes or untested refactors will be closed.
- Untested or failing code is not eligible for review.
- PR description *must* follow the PR template and explain *what* changed, *why*, and *how to test*.
- Links to related issues are required.
- Duplicate PRs will be automatically closed.
- Only have 1-2 PRs open at a time. Any further PRs will be closed.

**Maintainers reserve the right to close issues and PRs that do not align with the library roadmap.**

### Code Clarity and Style
- **Readability first:** Code must be self-documenting—if it is not self-explanatory, it should include clear, concise comments where logic is non-obvious.
- **Consistent Style:** Follow existing codebase style (e.g., function naming, docstring format)
- **No dead/debug code:** Remove commented-out blocks, leftover print statements, unrelated refactors
- Failure modes must be documented and handled with robust exception handling.

For more details on writing self-documenting code, check out [this guide](https://swimm.io/learn/documentation-tools/tips-for-creating-self-documenting-code).

### Testing Requirements
- **Coverage:** All new functionality must include unit tests covering both happy paths and relevant edge cases.
- **Passing tests:** pre-commit must pass with all checks (see below on how to run).
- **No silent failures:** Tests should fail loudly on errors. No `assert True` placeholders.

### Scope and Size
- **One purpose per PR:** No kitchen-sink PRs mixing bugfixes, refactors, and features.
- **Small, reviewable chunks:** If your PR is too large to review in under 30 minutes, break it up into chunks.
    - Each chunk must be independently testable and reviewable
    - If you can't explain why it can't be split, expect an automatic request for refactoring.
- Pull requests that are **large** (>500 LOC changed) or span multiple subsystems will be closed with automatic requests for refactoring.
- If the PR is to implement a new feature, please first make a GitHub issue to suggest the feature and allow for discussion. We reserve the right to close feature implementations and request discussion via an issue.

## **Local Setup**

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

## **Adding A New Provider**

### Step 1: Provider Class
Providers should go in the `any_llm/providers` folder. It should have the following directory structure:

```
📂 <your_provider>/
 ├── 📄 __init__.py        # Re-exports your provider class
 ├── 📄 your_provider.py   # Main provider implementation
 └── 📁 ...                # Any extra files (utils, configs, etc.)
```

At minimum, the `__init__.py` file should contain this:

```python
from any_llm.your_provider.your_provider import YourProvider

__all__ = ["YourProvider"]
```

Providers must inherit from the `Provider` class found in `any_llm.provider`. All abstract methods must be implemented and class variables must be set.

#### OpenAI API Compatible Providers
If you are using an OpenAI API compatible client, you can inherit from [BaseOpenAIProvider](https://github.com/mozilla-ai/any-llm/blob/main/src/any_llm/providers/openai/base.py). See the [LMStudio Provider](https://github.com/mozilla-ai/any-llm/blob/main/src/any_llm/providers/lmstudio/lmstudio.py) for an example.

### Step 2: Add `ProviderName`
In `src/any_llm/provider.py`, add a field to `ProviderName` for your provider.

### Step 3: Add Tests

Unit and integration tests must be added for each new provider.

Unit tests should be added to `tests/unit/providers`.

Integration tests are run on a matrix and configs can be added to `tests/conftest.py`. Here is what you need to update:

| Variable                                                                                                                                           | Notes                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [provider_reasoning_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L9)           | Default reasoning model                                                                                  |
| [provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L26)                    | Default model                                                                                            |
| [embedding_provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L60C5-L60C33) | Default embedding model                                                                                  |
| [provider_extra_kwargs_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L79)             | Extra kwargs to pass to provider factory. Include things like `base_url` here. DO NOT include `api_key`. |

🗒️ NOTE: Use the smallest reasonable possible models. Choice of model may be changed by core contributors.

### Notes and Gotchas

- Not all APIs support the parameters given by the `any-llm` completions, embeddings, and responses APIs, functions. If this is the case for your provider, make sure to add checks and raise a `ParameterError` if any unsupported parameters are passed. Be extra careful about this if you are making OpenAI API compatible provider.
- Your provider may require specific keyword arguments in its instantiation (for example, `base_url` for some cloud providers). If this is the case, make sure to validate upon provider instantiation and note in the docstring.
