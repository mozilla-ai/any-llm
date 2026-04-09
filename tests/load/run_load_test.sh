#!/usr/bin/env bash
# Orchestrator for the load-test comparison.
#
# Starts:
#   - Postgres (via docker) if TEST_DATABASE_URL is not set
#   - fake_provider on :9999
#   - gateway on :4000 (configured to point at fake_provider)
#
# Then runs k6 and tears everything down.
#
# Usage:
#   ./tests/load/run_load_test.sh [sync|async]
#
# The [sync|async] argument is used only for labeling the output. You are
# responsible for checking out the branch you want to test (main vs
# julian/async-asyncpg) before running.
#
# Prerequisites:
#   - k6      ->   brew install k6          (see README.md)
#   - docker  ->   for the ephemeral postgres (skipped if TEST_DATABASE_URL set)
#   - uv      ->   for running the gateway + fake provider

set -euo pipefail

LABEL="${1:-run}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Fake provider delay config (passed through as CLI args)
FAKE_DELAY_MS="${FAKE_DELAY_MS:-0}"
FAKE_JITTER_SIGMA="${FAKE_JITTER_SIGMA:-0}"
FAKE_DELAY_MIN_MS="${FAKE_DELAY_MIN_MS:-0}"
FAKE_DELAY_MAX_MS="${FAKE_DELAY_MAX_MS:-0}"

VUS="${VUS:-100}"
DURATION="${DURATION:-30s}"
GATEWAY_PORT="${GATEWAY_PORT:-4000}"
FAKE_PORT="${FAKE_PORT:-9999}"
WORKERS="${WORKERS:-1}"
# Budget validation strategy. One env var gives all three benchmark scenarios:
#   BUDGET_STRATEGY=for_update  FOR UPDATE held across entire request (legacy default)
#   BUDGET_STRATEGY=cas         lock-free conditional UPDATE, no FOR UPDATE (recommended)
#   BUDGET_STRATEGY=disabled    skip validate_user_budget entirely
BUDGET_STRATEGY="${BUDGET_STRATEGY:-for_update}"
# Usage log writer strategy:
#   LOG_WRITER_STRATEGY=single  write each event inline, 1 txn per event (default)
#   LOG_WRITER_STRATEGY=batch   queue + flush in batches (up to 100 rows / 1s)
LOG_WRITER_STRATEGY="${LOG_WRITER_STRATEGY:-single}"
# Fixed seed by default so the jitter sampler produces the same sequence
# across sync vs async runs. Override with RNG_SEED to vary.
RNG_SEED="${RNG_SEED:-42}"
# Persistent results directory (survives branch switches because it's untracked).
RESULTS_DIR="${RESULTS_DIR:-$ROOT/tests/load/results}"
mkdir -p "$RESULTS_DIR"

# --- Prerequisite checks ------------------------------------------------------
if ! command -v k6 >/dev/null 2>&1; then
  echo "error: k6 is not installed."
  echo "  macOS:   brew install k6"
  echo "  linux:   https://k6.io/docs/get-started/installation/"
  echo "  docker:  docker run --rm -i grafana/k6 run - < tests/load/load_test.js"
  exit 1
fi

# --- Postgres -----------------------------------------------------------------
if [[ -z "${TEST_DATABASE_URL:-}" ]]; then
  echo "[setup] starting postgres container"
  docker rm -f loadtest-pg >/dev/null 2>&1 || true
  docker run -d --name loadtest-pg \
    -e POSTGRES_USER=loadtest -e POSTGRES_PASSWORD=loadtest -e POSTGRES_DB=loadtest \
    -p 54329:5432 postgres:17 >/dev/null
  export DATABASE_URL="postgresql://loadtest:loadtest@localhost:54329/loadtest"
  for _ in $(seq 1 30); do
    if docker exec loadtest-pg pg_isready -U loadtest >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
else
  export DATABASE_URL="$TEST_DATABASE_URL"
fi

# --- Fake provider ------------------------------------------------------------
echo "[setup] starting fake_provider on :$FAKE_PORT (delay=${FAKE_DELAY_MS}ms sigma=${FAKE_JITTER_SIGMA} seed=${RNG_SEED})"
uv run --extra gateway python tests/load/fake_provider.py \
  --host 127.0.0.1 --port "$FAKE_PORT" \
  --delay-ms "$FAKE_DELAY_MS" \
  --jitter-sigma "$FAKE_JITTER_SIGMA" \
  --delay-min-ms "$FAKE_DELAY_MIN_MS" \
  --delay-max-ms "$FAKE_DELAY_MAX_MS" \
  --seed "$RNG_SEED" \
  > /tmp/fake_provider.log 2>&1 &
FAKE_PID=$!

# --- Gateway ------------------------------------------------------------------
echo "[setup] starting gateway on :$GATEWAY_PORT ($LABEL, $WORKERS workers, budget=$BUDGET_STRATEGY, log_writer=$LOG_WRITER_STRATEGY)"
export GATEWAY_MASTER_KEY="loadtest-master-key"
export GATEWAY_BOOTSTRAP_API_KEY="true"
export GATEWAY_BUDGET_STRATEGY="$BUDGET_STRATEGY"
export GATEWAY_LOG_WRITER_STRATEGY="$LOG_WRITER_STRATEGY"
uv run --extra gateway any-llm-gateway serve \
  --config tests/load/gateway-config.yml \
  --host 127.0.0.1 --port "$GATEWAY_PORT" --workers "$WORKERS" \
  > /tmp/gateway.log 2>&1 &
GATEWAY_PID=$!

cleanup() {
  echo "[teardown] shutting down"
  kill "$STATS_PID" 2>/dev/null || true
  kill "$GATEWAY_PID" "$FAKE_PID" 2>/dev/null || true
  wait "$GATEWAY_PID" "$FAKE_PID" "$STATS_PID" 2>/dev/null || true
  if [[ -z "${TEST_DATABASE_URL:-}" ]]; then
    docker rm -f loadtest-pg >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# wait for gateway
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -sf "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null || { echo "gateway didn't start"; cat /tmp/gateway.log; exit 1; }

# --- Gateway process sampler (sum CPU%/RSS across all worker pids, every 1s)
# Uvicorn forks workers under the launcher PID; sum their stats for a total.
(
  echo "timestamp,cpu_pct,rss_mb,n_procs"
  while kill -0 "$GATEWAY_PID" 2>/dev/null; do
    # find all python processes under the uv wrapper (master + N workers)
    PIDS=$(pgrep -f "any-llm-gateway serve" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$PIDS" ]]; then
      ps -o %cpu=,rss= -p "$PIDS" 2>/dev/null | \
        awk -v t="$(date +%s)" 'BEGIN{c=0;r=0;n=0} {c+=$1; r+=$2; n++} END {printf "%s,%.1f,%.1f,%d\n", t, c, r/1024, n}'
    fi
    sleep 1
  done
) > "/tmp/gateway-stats-${LABEL}.csv" &
STATS_PID=$!

# --- Create an API key via master key ----------------------------------------
KEY_RESPONSE=$(curl -sf -X POST "http://127.0.0.1:$GATEWAY_PORT/v1/keys" \
  -H "X-AnyLLM-Key: Bearer $GATEWAY_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_name":"loadtest"}')
KEY=$(printf '%s' "$KEY_RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')
echo "[setup] created gateway key ${KEY:0:12}… (user creation handled by k6 setup())"

# --- Run k6 -------------------------------------------------------------------
echo "[run] k6 -> $LABEL (VUS=$VUS DURATION=$DURATION)"
k6 run \
  -e KEY="$KEY" \
  -e MASTER_KEY="$GATEWAY_MASTER_KEY" \
  -e GATEWAY="http://127.0.0.1:$GATEWAY_PORT" \
  -e VUS="$VUS" \
  -e DURATION="$DURATION" \
  tests/load/load_test.js | tee "/tmp/k6-${LABEL}.txt"

mv load_results.json "/tmp/k6-${LABEL}.json" 2>/dev/null || true

# --- Verify log rows persisted ------------------------------------------------
# Shut the gateway down cleanly FIRST so any BatchLogWriter lifespan hook drains.
# Then count rows in usage_logs; compare vs k6's iteration count.
echo ""
echo "[verify] stopping gateway to drain log writer"
kill -TERM "$GATEWAY_PID" 2>/dev/null || true
wait "$GATEWAY_PID" 2>/dev/null || true

if [[ -z "${TEST_DATABASE_URL:-}" ]]; then
  LOG_COUNT=$(docker exec loadtest-pg psql -U loadtest -d loadtest -At -c "SELECT COUNT(*) FROM usage_logs" 2>/dev/null || echo "?")
else
  LOG_COUNT=$(psql "$TEST_DATABASE_URL" -At -c "SELECT COUNT(*) FROM usage_logs" 2>/dev/null || echo "?")
fi
# k6 iteration count (total requests made across warmup + main scenarios)
ITERATIONS=$(python3 -c "
import json, sys
d = json.load(open('/tmp/k6-${LABEL}.json'))
try:
    print(int(d['metrics']['iterations']['values']['count']))
except Exception:
    print('?')
" 2>/dev/null || echo "?")
echo ""
echo "=== usage_logs persistence check ($LABEL) ==="
echo "  k6 iterations completed : $ITERATIONS"
echo "  rows in usage_logs      : $LOG_COUNT"
if [[ "$LOG_COUNT" =~ ^[0-9]+$ ]] && [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
  if [[ "$LOG_COUNT" -eq "$ITERATIONS" ]]; then
    echo "  coverage                : 100% (no rows dropped)"
  else
    echo "  coverage                : $((LOG_COUNT * 100 / ITERATIONS))% ($((ITERATIONS - LOG_COUNT)) rows dropped)"
  fi
fi

# --- Summarize gateway process stats -----------------------------------------
STATS_SUMMARY=""
if [[ -s "/tmp/gateway-stats-${LABEL}.csv" ]]; then
  STATS_SUMMARY=$(awk -F',' '
    NR==1 { next }
    { cpu_sum+=$2; if ($2>cpu_max) cpu_max=$2; rss_sum+=$3; if ($3>rss_max) rss_max=$3; np=$4; n++ }
    END {
      if (n>0) {
        printf "  processes      %d (master + workers)\n", np
        printf "  samples        %d\n", n
        printf "  cpu %% avg/max   %.1f / %.1f (summed across processes)\n", cpu_sum/n, cpu_max
        printf "  rss MB avg/max  %.1f / %.1f (summed across processes)\n", rss_sum/n, rss_max
      }
    }
  ' "/tmp/gateway-stats-${LABEL}.csv")
  echo ""
  echo "=== gateway process stats ($LABEL) ==="
  echo "$STATS_SUMMARY"
fi

# --- Persist results to tests/load/results/ ----------------------------------
cp "/tmp/k6-${LABEL}.txt" "$RESULTS_DIR/k6-${LABEL}.txt"
cp "/tmp/k6-${LABEL}.json" "$RESULTS_DIR/k6-${LABEL}.json" 2>/dev/null || true
cp "/tmp/gateway-stats-${LABEL}.csv" "$RESULTS_DIR/gateway-stats-${LABEL}.csv"

echo ""
echo "[done] results saved to $RESULTS_DIR:"
echo "  k6-${LABEL}.txt        k6 summary"
echo "  k6-${LABEL}.json       k6 full metrics"
echo "  gateway-stats-${LABEL}.csv  gateway process cpu/rss samples"
