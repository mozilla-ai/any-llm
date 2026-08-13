#!/usr/bin/env bash
# Retry a command with exponential backoff.
#
# Usage: retry.sh <attempts> <initial-backoff-seconds> -- <command> [args...]
#
# Every local-provider model download in the integration workflow is a single-shot network call
# against huggingface.co or lmstudio.ai. A stalled download used to fail the whole job: on
# 2026-08-13 `lms get` aborted with "Download failed: Timed-out" at 0.02% of 1.14 GB and took
# the ollama and LM Studio tests down with it, none of which were broken.
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <attempts> <initial-backoff-seconds> -- <command> [args...]" >&2
  exit 2
fi

attempts=$1
shift
backoff=$1
shift
[ "${1:-}" = "--" ] && shift

for attempt in $(seq 1 "$attempts"); do
  # Capture the status directly off the command. Reading $? after an `if "$@"; then ... fi`
  # yields the status of the `if` compound (0 when the branch is not taken), not of the
  # command, which silently turned an exhausted retry into a passing step.
  "$@"
  status=$?

  if [ "$status" -eq 0 ]; then
    exit 0
  fi

  if [ "$attempt" -eq "$attempts" ]; then
    echo "::error::'$*' failed after ${attempts} attempts (last exit code ${status})"
    exit "$status"
  fi

  echo "Attempt ${attempt}/${attempts} of '$*' failed (exit ${status}); retrying in ${backoff}s..."
  sleep "$backoff"
  backoff=$((backoff * 2))
done
