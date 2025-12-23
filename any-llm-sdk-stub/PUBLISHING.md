# Publishing Guide for any-llm-sdk Stub Package

This guide explains how to publish the `any-llm-sdk` stub package to PyPI.

## Prerequisites

1. You must have maintainer access to the `any-llm-sdk` package on PyPI
2. Install build tools:
   ```bash
   pip install build twine
   ```

## Publishing Steps

### 1. Build the Package

From the `any-llm-sdk-stub` directory:

```bash
cd any-llm-sdk-stub
python -m build
```

This creates distribution files in the `dist/` directory.

### 2. Check the Package

Verify the package is correctly formed:

```bash
twine check dist/*
```

### 3. Upload to TestPyPI (Optional but Recommended)

Test the upload first:

```bash
twine upload --repository testpypi dist/*
```

Then test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ any-llm-sdk
```

### 4. Upload to PyPI

Once you've verified everything works:

```bash
twine upload dist/*
```

You'll be prompted for your PyPI credentials (or use an API token).

## Version Management

The version is managed by `setuptools_scm` and automatically derived from git tags in the parent repository. The stub package should use the same version number as the main `any-llm` package for consistency.

To create a new version:

```bash
# In the root of the main repository
git tag v0.16.0
git push origin v0.16.0
```

Then rebuild and publish the stub package.

## Publishing Checklist

- [ ] Main `any-llm` package is published to PyPI
- [ ] Version tag exists in git
- [ ] Built the stub package: `python -m build`
- [ ] Checked the package: `twine check dist/*`
- [ ] (Optional) Tested on TestPyPI
- [ ] Uploaded to PyPI: `twine upload dist/*`
- [ ] Verified installation: `pip install any-llm-sdk`
- [ ] Confirmed deprecation warning appears

## Ongoing Maintenance

This stub package should be republished whenever a new major version of `any-llm` is released to ensure the dependency constraint stays up to date. The stub package version should match the main package version.

## Troubleshooting

### "File already exists" error

If you get an error that the file already exists on PyPI, you need to bump the version number. Clean the dist directory and rebuild:

```bash
rm -rf dist/
python -m build
twine upload dist/*
```

### Import errors

Make sure `any-llm` is installed and importable. The stub package depends on it, so it should be automatically installed when users install `any-llm-sdk`.
