# GitBook Documentation Branch

The `gitbook-docs` branch contains **generated** GitBook-compatible documentation,
automatically updated by GitHub Actions on every push to `main`.

**Do not edit this branch manually** — all changes will be overwritten.

## How it works

1. `scripts/generate_openapi.py` generates the OpenAPI spec from the gateway source
2. `scripts/generate_api_docs.py` generates API reference from Python docstrings
3. `scripts/generate_provider_table.py` injects the provider compatibility table
4. `scripts/generate_cookbooks.py` converts Jupyter notebooks to Markdown
5. `scripts/generate_llms_txt.py` generates llms.txt and llms-full.txt
6. `scripts/convert_to_gitbook.py` copies all docs into `site/` and writes `SUMMARY.md`
7. The contents of `site/` are pushed to this branch
8. GitBook syncs from this branch
