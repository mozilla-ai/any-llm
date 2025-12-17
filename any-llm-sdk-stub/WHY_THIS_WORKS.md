# Why This Stub Package Approach Works

This document explains the technical details of how the stub package ensures a seamless transition.

## The Problem

When you rename a package on PyPI, there's no built-in redirect mechanism. This creates several issues:

1. **Existing installations break**: Users with `any-llm-sdk` in requirements can't install
2. **Documentation is outdated**: Old tutorials and Stack Overflow answers point to the wrong package
3. **Breaking changes**: Users need to update code and dependencies simultaneously
4. **Ecosystem fragmentation**: Some projects use the old name, some use the new name

## The Solution: Stub Package Pattern

The stub package acts as a "pointer" to the real package. Here's how it works:

### 1. Dependency Management

**pyproject.toml** (stub package):
```toml
dependencies = [
  "any-llm>=0.16.0",
]
```

When someone runs `pip install any-llm-sdk`, pip:
- Installs the `any-llm-sdk` package (the stub)
- Sees it depends on `any-llm>=0.16.0`
- Installs `any-llm` automatically
- Both packages end up installed

### 2. Module Re-exports

**src/any_llm/__init__.py** (stub package):
```python
from any_llm import *
```

This re-exports everything from the real package, so:
```python
from any_llm import completion  # Works the same from both packages!
```

### 3. Deprecation Warning

The stub package shows a warning on import:
```python
warnings.warn(
    "The 'any-llm-sdk' package has been renamed...",
    DeprecationWarning,
)
```

This:
- Doesn't break code (it's just a warning)
- Educates users about the migration
- Can be silenced if needed with warnings filters

## Why This Is Better Than Alternatives

### ❌ Alternative 1: Just rename and leave users behind
**Problem**: Breaks all existing installations

### ❌ Alternative 2: Maintain two separate packages
**Problem**: Double the maintenance burden, potential for drift

### ✅ Alternative 3: Stub package (our approach)
**Benefits**:
- Zero code changes needed
- Automatic dependency resolution
- Clear migration path
- Easy to maintain (stub rarely changes)

## Real-World Example: How It Works For Users

### Scenario 1: User with old dependency

**requirements.txt**:
```
any-llm-sdk>=0.15.0
```

When they run `pip install -r requirements.txt`:
1. Installs `any-llm-sdk` (stub)
2. Stub depends on `any-llm>=0.16.0`
3. Installs `any-llm>=0.16.0`
4. Their code works! They see a deprecation warning.
5. They update their requirements at their convenience.

### Scenario 2: User with new dependency

**requirements.txt**:
```
any-llm>=0.16.0
```

When they run `pip install -r requirements.txt`:
1. Installs `any-llm>=0.16.0`
2. Done! No stub needed.

### Scenario 3: Mixed dependencies (transitional period)

**requirements.txt**:
```
any-llm-sdk>=0.15.0
any-llm>=0.16.0
```

When they run `pip install -r requirements.txt`:
1. Installs `any-llm>=0.16.0` (satisfies second requirement)
2. Installs `any-llm-sdk` (satisfies first requirement)
3. Stub's dependency on `any-llm>=0.16.0` is already satisfied
4. Works correctly, no version conflicts

## Technical Details

### Package Identity

Each package has its own identity on PyPI:
- `any-llm` → `https://pypi.org/project/any-llm/`
- `any-llm-sdk` → `https://pypi.org/project/any-llm-sdk/`

But they both install to the same module: `any_llm/`

This is **allowed** because:
- PyPI doesn't enforce a 1:1 relationship between package name and module name
- Multiple packages can provide the same module (though usually they conflict)
- In our case, the stub intentionally re-exports the main package

### Version Synchronization

The stub package version should match the main package version:
- `any-llm==0.16.0` (the real package)
- `any-llm-sdk==0.16.0` (the stub, depends on any-llm>=0.16.0)

This makes it clear which versions correspond and simplifies troubleshooting.

### Import Resolution

When Python imports `any_llm`:
1. Checks `site-packages/any_llm/`
2. Finds `__init__.py` from whichever package was installed last
3. That `__init__.py` imports from the real package
4. Everything just works™

## Edge Cases Handled

### Edge Case 1: User installs stub first, then uninstalls main package
```bash
pip install any-llm-sdk  # Installs both
pip uninstall any-llm    # Removes main package
```
**Result**: Imports will fail. This is expected - you can't remove the real package!

**Fix**: `pip install any-llm` or `pip install --reinstall any-llm-sdk`

### Edge Case 2: Version conflicts
```bash
pip install any-llm==0.16.0
pip install any-llm-sdk==0.15.0  # Older stub
```
**Result**: pip will try to resolve dependencies. The stub's dependency on `any-llm` might cause pip to upgrade/downgrade.

**Fix**: Keep versions synchronized. Always publish both packages with the same version.

### Edge Case 3: User has both in requirements with different version constraints
```
any-llm>=0.16.0
any-llm-sdk<0.16.0
```
**Result**: pip can't satisfy both constraints and will fail.

**Fix**: Update to use only `any-llm>=0.16.0`.

## Maintenance Going Forward

### During Active Development (Now - 6 months)
- Publish both packages with every release
- Maintain version synchronization
- Monitor for user issues

### After Transition Period (6+ months)
- Continue publishing stub package (minimal effort)
- Consider making stub package more aggressive with warnings
- Eventually: Stub package can stop being updated (still points to any-llm)

### Long-term (1+ years)
- Stub package can be "archived" with a final release
- Final release always depends on latest any-llm
- Users eventually migrate away naturally

## Comparison with Other Projects

This pattern is used by many Python projects:

- `typing` → `typing_extensions` (backports)
- `pkg_resources` → `importlib.metadata` (stdlib transition)
- Many renamed packages in the ecosystem

PyPI explicitly allows and supports this pattern!

## Conclusion

The stub package approach provides:
- ✅ Zero-breaking-change migration
- ✅ Clear deprecation path
- ✅ Minimal maintenance burden
- ✅ User-friendly transition
- ✅ Ecosystem compatibility

It's the recommended approach for package renames in the Python ecosystem.
