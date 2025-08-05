#!/usr/bin/env bash
# Thank you to https://jaredkhan.com/blog/mypy-pre-commit for this super helpful script!
# This script is called by the pre-commit hook.
set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")/.."

# Install the dependencies into the mypy env.
# Note that this can take seconds to run.
python -m pip install -U -e '.[all]' --quiet

# Run on all files.
python -m mypy src/
python -m mypy tests/
