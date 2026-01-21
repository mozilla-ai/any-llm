#!/bin/bash
set -e

K_FILTER="${1:-}"

if [ -n "$K_FILTER" ]; then
    cargo test -- "$K_FILTER"
else
    cargo test
fi
