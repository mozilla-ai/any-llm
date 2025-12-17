# any-llm-sdk → any-llm

⚠️ **DEPRECATION NOTICE** ⚠️

The `any-llm-sdk` package has been renamed to `any-llm`.

## Migration Guide

### Update Your Installation

**Old:**
```bash
pip install any-llm-sdk
```

**New:**
```bash
pip install any-llm
```

### Update Your Dependencies

If you have `any-llm-sdk` in your `requirements.txt`, `pyproject.toml`, or other dependency files:

**requirements.txt:**
```diff
- any-llm-sdk>=0.15.0
+ any-llm>=0.16.0
```

**pyproject.toml:**
```diff
dependencies = [
-    "any-llm-sdk[all]>=0.15.0",
+    "any-llm[all]>=0.16.0",
]
```

### No Code Changes Required

Your import statements will continue to work without any changes:

```python
from any_llm import completion, AnyLLM
# Works exactly the same way!
```

## What This Package Does

This package (`any-llm-sdk`) now serves as a lightweight redirect to the real package (`any-llm`). When you install `any-llm-sdk`, it automatically installs `any-llm` as a dependency and re-exports all its functionality.

## Why the Change?

The shorter package name `any-llm` better matches the project name and is more concise for users to type and remember.

## Timeline

- **Now**: Both packages work. `any-llm-sdk` shows a deprecation warning.
- **Future**: `any-llm-sdk` stub package will be maintained for backwards compatibility but may eventually be deprecated.

## More Information

For the latest documentation and updates, visit:
- **Documentation**: https://mozilla-ai.github.io/any-llm/
- **GitHub**: https://github.com/mozilla-ai/any-llm
- **PyPI (new)**: https://pypi.org/project/any-llm/
