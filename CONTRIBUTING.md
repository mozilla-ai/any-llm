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
