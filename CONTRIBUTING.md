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
uv run pre-commit run --all-files --verbose

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
Make your changes! Read our [Implementation Checklist](#2-implementation-checklist) if adding a new provider

### 3. Write Tests

**Every change needs tests!** This is non-negotiable.

#### Test Requirements

- **New features**: Add tests covering happy path and error cases
- **Bug fixes**: Add a test that reproduces the bug
- **Provider integrations**: Comprehensive test suite required
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
- **docs/providers.md** when adding providers

```bash
mkdocs serve
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

## Developing the Gateway

If you are contributing to the `any-llm-gateway` (the FastAPI proxy), the setup differs slightly from the SDK. We use a layered Docker Compose setup:

1. `docker-compose.yml`: Defines the production image (used by end-users).
2. `docker-compose.override.yml`: Overrides the image with your local build context.

### Running the Gateway Locally

Because the `docker-compose.override.yml` is checked into the repository, Docker will automatically detect it and build from your source code.

1. **Navigate to the gateway directory**

   ```bash
   cd any-llm-gateway
   ```

2. **Run with Build** Run the following command to build your local changes and start the service:
```bash
docker compose up --build
```

3. **Verify Local Version** To ensure you are running your local version, check the logs or the health endpoint. The service runs on http://localhost:8000.

### Gateway Tests
To run tests specifically for the gateway:

```bash
# From the project root
pytest tests/gateway
```

## Adding a New Provider

Adding provider support is a major contribution! Here's the complete process:

### 1. Check Requirements

Before requesting or implementing:

- [ ] Provider has an official Python SDK **OR** well-documented REST API
- [ ] Provider is actively maintained and supported
- [ ] Provider has substantial user base or unique capabilities
- [ ] Provider's interface is compatible with any-llm's design
- [ ] No existing issue/PR for adding this provider


### 2. Implementation Checklist

Implement the provider keeping this checklist in mind:

```
any_llm/
├── providers/
│   ├── 📂 <your_provider>/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 your_provider.py   # Main provider implementation
│   │   ├── 📁 ...                # Any extra files (utils, configs, etc.)
```

**Required Implementation**:

- [ ] Create provider module in `any_llm/providers/`<br>
In `src/any_llm/provider.py`, add a field to `ProviderName` for your provider.
- [ ] Handle provider-specific errors gracefully
- [ ] Add type hints and docstrings
- [ ] Use official SDK when available
- [ ] Add to `pyproject.toml` optional dependencies
- [ ] Add provider to `any_llm/__init__.py` <br>
<p>

At minimum, the `__init__.py` file should contain :

```python
from any_llm.your_provider.your_provider import YourProvider

__all__ = ["YourProvider"]
```

Providers must inherit from the `Provider` class found in `any_llm.provider`. All abstract methods must be implemented and class variables must be set.

**Testing Requirements**:

- [ ] Unit tests for all provider functions
- [ ] Integration tests with real API (mocked in CI)
- [ ] Error handling tests
- [ ] Streaming tests (if applicable)
- [ ] Test suite in `tests/unit/providers`
- [ ] Minimum 85% coverage for provider code

Add your test config to the following in `tests/conftest.py`:

| Variable                                                                                                                                           | Notes                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [provider_reasoning_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L9)           | Default reasoning model                                                                                  |
| [provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L26)                    | Default model                                                                                            |
| [embedding_provider_model_map](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L60C5-L60C33) | Default embedding model                                                                                  |
| [provider_client_config](https://github.com/mozilla-ai/any-llm/blob/2aa7401a857c65efe94f9af7d2d7503330b63ab9/tests/conftest.py#L79)             | Extra kwargs to pass to provider factory. Include things like `base_url` here. DO NOT include `api_key`. |


**Documentation Requirements**:

- [ ] Add to `docs/providers.md` with available capabilities.
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
