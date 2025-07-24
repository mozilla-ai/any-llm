# Installation Guide for Any LLM

Any LLM supports two primary installation scenarios:

## 1. Direct Usage

If you are directly using Any LLM, you should install the package along with the extras for the provider you plan to use. For example:

```bash
pip install any-llm[mistral]
```

This command installs Any LLM along with the dependencies required for the `mistral` provider. You can replace `mistral` with the name of the provider you intend to use (e.g., `openai`, `huggingface`, etc.).

## 2. Integration into Another Library

If you are integrating Any LLM into your own library, you should install the base package:

```bash
pip install any-llm
```

In this scenario, the end user of your library will be responsible for installing the dependencies for the provider they want to use. Any LLM is designed to handle missing dependencies gracefully. If a user tries to use a provider without the required dependencies installed, they will encounter a runtime exception that clearly explains what they need to do to resolve the issue.

## Runtime Exceptions

When a provider-specific dependency is missing, Any LLM will raise an exception at runtime. The exception message will include instructions on how to install the required dependencies. For example:

```
Provider 'mistral' requires the 'mistral-sdk' package. Please install it using:

pip install any-llm[mistral]
```

This design ensures that users can install and use Any LLM without unnecessary dependencies, while still providing clear guidance when additional packages are needed.

## Summary

- **Direct Usage**: Install with extras for the provider you plan to use.
- **Integration**: Install the base package and let end users handle provider-specific dependencies.

For more details, refer to the [Quickstart Guide](./quickstart.md).
