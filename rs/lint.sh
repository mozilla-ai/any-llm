#!/bin/bash
set -e

# Ensure required components are installed
rustup component add rustfmt clippy 2>/dev/null || true

cargo fmt
cargo clippy --fix --allow-dirty --allow-staged
