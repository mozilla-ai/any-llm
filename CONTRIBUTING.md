# Contributing to any-llm

Thank you for your interest in contributing to any-llm! 🎉

We're building a simple, unified interface for working with multiple LLM providers, and we welcome contributions from developers of all experience levels. Whether you're fixing a typo, adding a new provider, or improving our architecture, your help is appreciated.

## Before You Start

### Check for Duplicates

Before creating a new issue or starting work:
- [ ] Search [existing issues](https://github.com/mozilla-ai/any-llm/issues) for duplicates
- [ ] Check [open pull requests](https://github.com/mozilla-ai/any-llm/pulls) to see if someone is already working on it
- [ ] For bugs, verify it still exists in the `main` branch

### Discuss Major Changes First

For significant changes, please open an issue **before** starting work:

- New provider integrations
- API changes or new public methods
- Architectural changes
- Breaking changes
- New dependencies

**Use the `rfc` label** for design discussions. This ensures alignment with project goals and saves everyone time.

### Read Our Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). We're committed to maintaining a welcoming, inclusive community.

## Development Setup

### Prerequisites

- **Python 3.11 or newer**
- **Git**
- **uv** (or your preferred package manager)
- **API keys** for any providers you want to test

### Quick Start
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) as your Python package and project manager.

```bash
# 1. Fork the repository on GitHub
# Click the "Fork" button at https://github.com/mozilla-ai/any-llm

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/any-llm.git
cd any-llm

# 3. Add upstream remote
git remote add upstream https://github.com/mozilla-ai/any-llm.git

# 4. Create a virtual environment
uv venv
source .venv/bin/activate
uv sync --all-extras -U --python=3.13

# 5. Ensure all checks pass
UV_SYSTEM_PYTHON=0 uv run pre-commit run --all-files --verbose
```

> **Note:** mypy errors about missing modules (e.g. `groq`, `mistralai`) are
> expected when running locally without provider extras installed. These are
> optional dependencies and CI is the authoritative environment for mypy checks.

```bash
# 7. Verify your setup
pytest -v tests/unit
pytest -v tests/integration -n auto
```

### Setting Up API Keys

Create a `.env` file in the project root (this file is gitignored):

```bash
# Add keys for providers you want to test
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
# Add others as needed
```

Alternatively, export environment variables:

```bash
export OPENAI_API_KEY="your_key_here"
```

**⚠️ Never commit API keys!** Always use environment variables or `.env` files.

## Making Changes

### 1. Create a Branch

Always work on a feature branch, never directly on `main`:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `provider/` - New provider integrations
- `refactor/` - Code improvements without behavior changes

### 2. Make Changes
Make your changes! Read [Adding a New Provider](#adding-a-new-provider) first if that is what you are doing, since some providers need a config row rather than code, and some need no PR at all.

### 3. Write Tests

**Every change needs tests!** This is non-negotiable.

#### Test Requirements

- **New features**: Add tests covering happy path and error cases
- **Bug fixes**: Add a test that reproduces the bug
- **Provider integrations**: Comprehensive test suite required, except for config-only gateways added as a registry row, which are covered by the generic registry tests (see [2a](#2a-config-only-gateways-add-a-registry-row))
- **Target**: Minimum 85% coverage for new code


### 4. Update Documentation

Documentation is as important as code!

Update when you:
- Add a new feature
- Change existing behavior
- Add a new provider
- Fix a bug that affects usage

Documentation to update:
- **Docstrings** in code (required)
- **README.md** if changing core functionality
- Authored docs in `docs/` (e.g. `docs/quickstart.md`) when adding or changing features

Authored docs live in `docs/`. Generated files (`docs/api/`, `docs/providers.md`, `docs/cookbooks/any-llm-getting-started.md`) are built by CI and not committed to the repository. The final publish artifact is `site/`, which CI builds and pushes to the `gitbook-docs` branch.

To preview the rendered output locally:

```bash
uv run python scripts/convert_to_gitbook.py
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for Anthropic Claude 3.5 Sonnet"
git commit -m "Fix streaming response handling for OpenAI"
git commit -m "Update documentation for Azure OpenAI configuration"

# Less helpful commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

## Adding a New Provider

Not every provider needs code, and some need no PR at all. Start here.

### 0. Do You Need a Provider Entry?

**If your endpoint speaks the OpenAI API, you are never blocked.** Point any-llm straight at it:

```python
from any_llm import AnyLLM

llm = AnyLLM.create_openai_compatible(
    name="mygateway",
    api_base="https://mygateway.example/v1",
    api_key="your-key",  # optional for keyless local servers
)
```

The provider reports the name you give it rather than reporting itself as `openai`, and is used exactly like any other provider instance. See [Custom OpenAI-compatible Endpoints](docs/quickstart.md#custom-openai-compatible-endpoints). This is the right path for private gateways, self-hosted servers, and anything you do not need us to ship.

Open a PR when you want the provider **listed in our docs and resolvable by name** for everyone else. Which path that PR takes depends on whether the provider needs code:

| Your provider...                                                                                                     | Goes in                                            | See |
| -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --- |
| speaks the OpenAI API and needs nothing beyond a base URL, an API key env var, and capability flags                   | a row in `src/any_llm/providers/registry.py`        | [2a](#2a-config-only-gateways-add-a-registry-row) |
| needs custom auth, non-OpenAI request or response shapes, param translation, or model-list quirks                     | a folder under `src/any_llm/providers/`             | [2b](#2b-providers-that-need-code) |

The deciding question is whether the protocol *requires* the code, judged by the reviewer. Shipping an official SDK is not by itself a reason for a folder, and adding an override that is not needed does not turn a config-only gateway into a code provider.

### Provider Tiers

Every listed provider sits in one of two tiers. The tier is a support promise, not a statement about code shape: a config-only provider we hold keys for stays verified.

- **Verified**: we hold API keys, integration tests run in CI, and we fix breakage. These are the providers the README and docs promote.
- **Community**: verified live by the contributor when the PR is opened, then maintained by the community. No CI key, so the provider is not in the CI provider list and its integration tests skip there.

New third-party gateways land as **community** by default. CI cannot use repository secrets on fork PRs, so contribution-time live verification is the only bar we can actually enforce; we do not ask contributors for ongoing proof.

Promotion to verified is a maintainer action, not something a PR can do; the full list of edits it takes is in [2a](#2a-config-only-gateways-add-a-registry-row). Open an issue if you think a provider deserves it.

**Removal.** A community entry is removed when the gateway shuts down, or when users report sustained breakage that nobody steps up to fix. Cheap addition is only sustainable if removal is equally cheap, so removal PRs are routine and do not need the original contributor's sign-off.

The tier shows up as a column in the generated [provider table](https://docs.mozilla.ai/any-llm/providers), sourced from `VERIFIED_PROVIDERS` in `src/any_llm/constants.py`.

### 1. Check Requirements

Before requesting or implementing:

- [ ] Provider has an official Python SDK **OR** well-documented REST API
- [ ] Provider is actively maintained and supported
- [ ] Provider's interface is compatible with any-llm's design
- [ ] No existing issue/PR for adding this provider

A provider that needs a **code folder** must also clear a usage bar: a substantial user base or unique capabilities. A folder is the expensive category, in review attention now and in maintenance and eventual cleanup later, so it carries that bar in both tiers. Registry rows waive it because a row costs one line to add and one line to remove. Asking for the **verified** tier raises the bar further, since it commits us to holding a key and fixing breakage.

### 2a. Config-only Gateways: Add a Registry Row

Add one row to `PROVIDER_REGISTRY` in `src/any_llm/providers/registry.py`:

```python
"examplegw": OpenAICompatibleProviderConfig(
    name="examplegw",
    api_base="https://api.examplegw.test/v1",
    env_api_key_name="EXAMPLEGW_API_KEY",
    env_api_base_name="EXAMPLEGW_API_BASE",
    provider_documentation_url="https://docs.examplegw.test",
),
```

That row is the whole change. There is no provider class, no folder, no `__init__.py`, no `pyproject.toml` extra, and no `LLMProvider` member: the name resolves by string everywhere a provider name is accepted, including `AnyLLM.create("examplegw")`, `"examplegw:model"`, and `provider="examplegw"`.

- [ ] Add the row, setting capability flags for what the gateway actually supports. Flags default to the conservative baseline of completion, streaming, and model listing; everything else is opt-in. Do not set a flag you have not exercised against the live endpoint.
- [ ] Paste live verification output in the PR (see below).

Two things you may notice in the tree that do *not* apply to a new row. The registry providers already there carry a package directory and an `LLMProvider` member; those are compatibility shims from before they were migrated to rows, kept so their deep-import paths and `LLMProvider.<NAME>` references keep working. And a row is not added to the `tests/conftest.py` model maps, because a community gateway has no CI key.

**Promotion to the verified tier** is a maintainer change, and it is the one case where a row does need the rest of the scaffolding:

- a repository secret holding the key
- an `EXPECTED_PROVIDERS` entry in `.github/workflows/tests-integration.yaml`, which forces the integration tests to run rather than skip
- the name in `VERIFIED_PROVIDERS` in `src/any_llm/constants.py`, which is what labels it verified in the generated table; `tests/unit/test_provider_tiers.py` asserts these two lists match
- an `LLMProvider` member, so the enum-driven test matrix picks the provider up, which in turn requires a `pyproject.toml` extra plus an `all` group entry and a package directory under `src/any_llm/providers/`, because `tests/unit/test_provider.py` and `tests/unit/test_provider_pyproject_options.py` assert both
- `tests/conftest.py` entries for the capabilities the provider supports

**Live verification.** Run the following against the real endpoint with your own key and paste the output into the PR, with the key redacted:

```python
from any_llm import AnyLLM

llm = AnyLLM.create("examplegw", api_key="...")

# completion
print(llm.completion(model="<model>", messages=[{"role": "user", "content": "Say hi"}]).choices[0].message.content)

# streaming
for chunk in llm.completion(model="<model>", messages=[{"role": "user", "content": "Count to 3"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# model listing
print([model.id for model in llm.list_models()][:5])
```

If the gateway does not support one of these, set the matching flag to `False` on the row (`supports_completion_streaming=False`, `supports_list_models=False`) and say so in the PR instead of pasting that call.

### 2b. Providers That Need Code

Implement the provider keeping this checklist in mind:

```
src/any_llm/
├── providers/
│   ├── 📂 <your_provider>/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 your_provider.py   # Main provider implementation
│   │   ├── 📁 ...                # Any extra files (utils, configs, etc.)
```

**Required Implementation**:

- [ ] Create provider module in `src/any_llm/providers/`<br>
In `src/any_llm/constants.py`, add a member to `LLMProvider` for your provider.
- [ ] Handle provider-specific errors gracefully
- [ ] Add type hints and docstrings
- [ ] Use official SDK when available
- [ ] Add an extra to `pyproject.toml` optional dependencies and add the name to the `all` group (`tests/unit/test_provider_pyproject_options.py` asserts both)
- [ ] Export the provider class from your package's `__init__.py` <br>
<p>

At minimum, `src/any_llm/providers/your_provider/__init__.py` should contain:

```python
from .your_provider import YourProvider

__all__ = ["YourProvider"]
```

Providers must inherit from the `AnyLLM` class found in `any_llm.any_llm`. All abstract methods must be implemented and class variables must be set, including the `SUPPORTS_*` capability flags. When overriding a base-class method, use the `@override` decorator from `typing_extensions`.

**Testing Requirements**:

- [ ] Unit tests for all provider functions
- [ ] Integration tests with real API (mocked in CI)
- [ ] Error handling tests
- [ ] Streaming tests (if applicable)
- [ ] Test suite in `tests/unit/providers`
- [ ] Minimum 85% coverage for provider code

Unit tests are required in both tiers. Integration tests depend on the tier. A **verified**-tier provider needs entries in the relevant `tests/conftest.py` maps below, so that it exercises the CI matrix, and only for the capabilities the provider actually supports. A **community**-tier provider has no CI key, so it skips those maps and pastes live verification output in the PR instead; use the commands in [2a](#2a-config-only-gateways-add-a-registry-row), substituting your provider name and dropping any operation the provider does not support.

For verified providers, add your test config to the following in `tests/conftest.py`:

| Variable                                                                                                                                           | Notes                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [provider_reasoning_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L9)           | Default reasoning model                                                                                  |
| [provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L26)                    | Default model                                                                                            |
| [embedding_provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L60C5-L60C33) | Default embedding model                                                                                  |
| [provider_client_config](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L79)             | Extra kwargs to pass to provider factory. Include things like `base_url` here. DO NOT include `api_key`. |


**Documentation Requirements**:

- [ ] Add provider metadata to the source code so it appears in the generated `providers.md` table.
- [ ] Update installation instructions.


## Submitting Your Contribution

### 1. Push to Your Fork

```bash
# Commit your changes
git add .
git commit -m "feat: add support for Example provider"

# Push to your fork
git push origin feature/example-provider
```

### 2. Create a Pull Request

1. Go to https://github.com/mozilla-ai/any-llm
2. Click "New Pull Request"
3. Click "compare across forks"
4. Select your fork and branch
5. Fill out the [PR template](pull_request_template.md) completely
6. Click "Create Pull Request"


## Review Process

### What to Expect

1. **Initial Response**: Within **5 business days**
2. **Simple Fixes**: Usually merged within **1 week**
3. **Complex Features**: May take **2-3 weeks** for thorough review
4. **Provider Integrations**: Often require **2-3 review cycles**

### During Review

- Maintainers will provide constructive feedback
- Address comments with new commits (don't force push)
- Ask questions if feedback is unclear
- Be patient and respectful
- CI must pass before merge

### If Your PR Goes Stale

- No activity for **30+ days** may result in closure
- You can always reopen and continue later
- Let us know if you need help finishing
- We can find another contributor to complete it


## Your First Contribution

New to open source? Welcome! Here's how to get started:

### Step 1: Find an Issue

Look for issues labeled:
- `good-first-issue` - Perfect for newcomers
- `help-wanted` - Community contributions welcome
- `documentation` - Often accessible for beginners

### Step 2: Claim the Issue

Comment on the issue:
> "Hi! I'd like to work on this. Is it still available?"

We'll assign it to you and provide guidance.

### Step 3: Ask Questions Early

Don't spend days stuck! Ask questions:
- In the issue comments
- In GitHub Discussions
- Tag `@maintainers` if needed

### Step 4: Start Small

Your first PR doesn't have to be perfect:
- Fix a typo
- Improve documentation
- Add a test
- Fix a small bug

### Step 5: Learn and Grow

Every expert was once a beginner. We're here to help you grow as a contributor!

## Code of Conduct

This project follows Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

In brief:
- **Be respectful and inclusive**
- **Focus on constructive feedback**
- **Help create a welcoming environment**
- **Report concerns** to maintainers

See our full [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Questions?

- 💬 Open a [GitHub Discussion](https://github.com/mozilla-ai/any-llm/discussions)
- 🐛 Report a [Bug](https://github.com/mozilla-ai/any-llm/issues/new?template=bug_report.md)
- 💡 Request a [Feature](https://github.com/mozilla-ai/any-llm/issues/new?template=feature_request.md)


We're excited to have you as part of the any-llm community! 🚀

---

**License**: By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).
