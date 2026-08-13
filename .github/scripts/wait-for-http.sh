#!/usr/bin/env bash
# Poll an HTTP endpoint until it returns a success status, or give up.
#
# Usage: wait-for-http.sh <url> <timeout-seconds> [description]
#
# Two things this does that the previous inline form
# (`timeout 60 bash -c 'until curl -s URL >/dev/null; do sleep 1; done'`) did not:
#
#   1. Requires a 2xx. Plain `curl -s` exits 0 for any HTTP response, including llama.cpp's
#      /health returning 503 "loading model" and LM Studio's /v1/models answering before any
#      model is loaded. Either one counted as ready and let tests start against a server that
#      could not serve them. `curl -fsS` treats >= 400 as failure.
#   2. Explains itself on timeout. The old form died as a bare "Process completed with exit
#      code 124" with no indication of which server never came up, or why.
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <url> <timeout-seconds> [description]" >&2
  exit 2
fi

url=$1
timeout=$2
description=${3:-$url}

start=$SECONDS
last_output=""

while [ $((SECONDS - start)) -lt "$timeout" ]; do
  if last_output=$(curl -fsS --max-time 10 "$url" 2>&1); then
    echo "${description} ready after $((SECONDS - start))s"
    exit 0
  fi
  sleep 2
done

echo "::error::${description} did not become ready at ${url} within ${timeout}s"
echo "Last curl output: ${last_output:-<none>}"
exit 1
