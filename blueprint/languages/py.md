# any-llm-py: Python Implementation Guidelines

This document refines the core concepts and architecture of [any-llm.md](./any_llm.md) with
a focus on python-specific implementation details.

## Architecture

### Async

Method should be async first.

All the sync methods must be a simple wrapper of the corresponding async method.

