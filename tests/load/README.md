# Gateway load test — sync vs async throughput

This directory contains a self-contained load test that demonstrates the
throughput win from converting the gateway's DB layer from sync `psycopg2` to
async `asyncpg`. It does **not** call any real LLM provider — it points the
gateway at a local noop fake that returns a canned `ChatCompletion` response
with a configurable per-request delay.

## What it measures

Two k6 scenarios run back-to-back:

| Scenario | User IDs | What it stresses |
|---|---|---|
| `distinct_users` | unique per VU | pure per-request gateway overhead; measures the ceiling on concurrent requests given one uvicorn worker |
| `same_user` | one shared user_id | DB-row contention — on the sync build this serializes on `SELECT FOR UPDATE` held across the fake "LLM" call |

## The short story on sync vs async

**This branch (`julian/async-asyncpg`) contains no sync DB code.** Every gateway
DB call goes through `sqlalchemy.ext.asyncio.AsyncSession` + `asyncpg`. To get
a before/after comparison you run the load test against both branches:

```bash
# 1. checkout main (sync), run the load test, save results as "sync"
git checkout main
./tests/load/run_load_test.sh sync

# 2. checkout this branch (async), run again, save as "async"
git checkout julian/async-asyncpg
./tests/load/run_load_test.sh async

# 3. inspect /tmp/k6-sync.txt and /tmp/k6-async.txt side-by-side
diff -u /tmp/k6-sync.txt /tmp/k6-async.txt
```

The `run_load_test.sh` argument (`sync`/`async`) is just a label — the script
doesn't modify your checkout. Whichever code is on disk is what gets
benchmarked.

## Prerequisites

### Install k6

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install k6` |
| Linux (apt) | `sudo gpg -k && sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69 && echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \| sudo tee /etc/apt/sources.list.d/k6.list && sudo apt-get update && sudo apt-get install k6` |
| Windows (Chocolatey) | `choco install k6` |
| Docker (no install) | `docker run --rm -i grafana/k6 run - < tests/load/load_test.js` |
| Binary download | https://github.com/grafana/k6/releases |

Verify: `k6 version`.

### Other prerequisites

- Docker (for the ephemeral Postgres container) — skipped if
  `TEST_DATABASE_URL` is set and points at an existing Postgres
- `uv` + this project's `gateway` extra (`uv sync --extra gateway`)

## One-off runs

```bash
# noop (no artificial LLM delay) — measures pure gateway overhead
FAKE_DELAY_MS=0 ./tests/load/run_load_test.sh async

# realistic LLM-ish latency: median 200ms with a long tail
FAKE_DELAY_MS=200 FAKE_JITTER_SIGMA=0.4 ./tests/load/run_load_test.sh async

# crank up the load
VUS=200 DURATION=60s ./tests/load/run_load_test.sh async
```

## Configuration knobs

### Fake provider (`fake_provider.py`)

The fake provider is a click CLI. Run it standalone with:

```bash
uv run --extra gateway python tests/load/fake_provider.py --help
```

| CLI flag | Default | Shell-script env | What it does |
|---|---|---|---|
| `--delay-ms` | `0` | `FAKE_DELAY_MS` | Median per-request delay in ms. `0` = return immediately. |
| `--jitter-sigma` | `0.0` | `FAKE_JITTER_SIGMA` | Log-normal sigma around the median. `0` = fixed delay. Realistic values: `0.2` (tight), `0.4` (moderate), `0.6` (long-tail). |
| `--delay-min-ms` | `0` | `FAKE_DELAY_MIN_MS` | Hard floor clamp after sampling. |
| `--delay-max-ms` | `0` (unbounded) | `FAKE_DELAY_MAX_MS` | Hard ceiling clamp after sampling. |
| `--host` | `127.0.0.1` | — | Bind host |
| `--port` | `9999` | `FAKE_PORT` | Bind port |

With `--delay-ms 200 --jitter-sigma 0.4` the sampled delays look
roughly like:
- p50: ~200ms
- p95: ~390ms
- p99: ~510ms

That mirrors the shape of real LLM non-streaming latencies reasonably well
(most responses clustered near a median, a tail of slow ones).

### Load test (`load_test.js`)

| Env var | Default | What it does |
|---|---|---|
| `KEY` | (required) | Gateway API key — created by `run_load_test.sh` automatically |
| `GATEWAY` | `http://localhost:4000` | Gateway base URL |
| `MODEL` | `openai:fake` | Model string sent in requests |
| `VUS` | `100` | Virtual users per scenario |
| `DURATION` | `30s` | Duration per scenario |

## Expected shape of results

With `FAKE_DELAY_MS=0` (noop upstream) and 100 VUs on a single worker:

| Scenario | Branch | Expected throughput |
|---|---|---|
| `distinct_users` | `main` (sync) | low; bottlenecked by sync DB calls blocking the event loop per request |
| `distinct_users` | `julian/async-asyncpg` | substantially higher; async DB calls yield, event loop interleaves |
| `same_user` | `main` (sync) | **very low** — requests serialize on `SELECT FOR UPDATE` of the user row held across the upstream call |
| `same_user` | `julian/async-asyncpg` | same as distinct_users — no event-loop blocking even under row-lock contention |

The **same_user** scenario is the headline result: on `main` the gateway
effectively serializes all requests for a single user, because the sync
`SELECT FOR UPDATE` in `validate_user_budget` blocks the single async event
loop while waiting on a contended row lock. On `julian/async-asyncpg` that
wait yields, so other VUs' requests make progress.

> ⚠️ Note: the exact numbers depend on your hardware, Postgres config, and
> whether the fake provider is adding jitter. The **ratio** between sync and
> async is what matters.

## How the setup works under the hood

`run_load_test.sh` orchestrates:

1. **Postgres** — a `postgres:17` container on port 54329, unless
   `TEST_DATABASE_URL` is already set
2. **Fake provider** — `uvicorn tests.load.fake_provider:app` on port 9999
3. **Gateway** — `any-llm-gateway serve --config tests/load/gateway-config.yml`
   on port 4000, with `providers.openai.api_base` pointing at the fake
4. **API key** — created via `POST /v1/keys` using the master key
5. **k6** — runs `load_test.js` with both scenarios, 35-second gap between them
6. **Teardown** — shuts everything down on exit (including the Postgres
   container if it started one)

Output goes to `/tmp/k6-<label>.{txt,json}`.

## Watching live during a run

`run_load_test.sh` writes a CSV of the gateway process's CPU% / RSS MB every
second to `/tmp/gateway-stats-<label>.csv` and prints avg/max at the end. For
watching live in a separate terminal, use any of:

| Tool | What it shows | Install |
|---|---|---|
| `htop` | interactive CPU / mem, all processes | `brew install htop` |
| `top -pid $(pgrep -f any-llm-gateway)` | single-process CPU / mem, built-in | none |
| `nettop -p $(pgrep -f any-llm-gateway)` | per-process network bytes in/out (macOS) | none |
| `iftop` | per-interface network traffic | `brew install iftop` |
| `nmon` | combined CPU/mem/disk/net, optional CSV record with `-f` | `brew install nmon` |

The built-in `ps`-based sampler in `run_load_test.sh` is the recorded, scriptable
source of truth — the interactive tools above are for eyeballing live.

## Limitations

- **Single worker.** The gateway is launched with `--workers 1` to isolate
  the async-vs-sync signal. Running with more workers masks the win (each
  worker has its own event loop, so a blocked worker only takes down its own
  share of traffic).
- **No streaming.** The fake provider only returns non-streaming responses.
  Streaming would show an even larger win (because streaming sync holds the
  DB transaction across the full response), but is harder to mock faithfully.
- **Noop LLM.** Real providers add variable network latency that this fake
  doesn't model exactly. Use `FAKE_JITTER_SIGMA` to get closer to reality.
- **Database is not shared** across the sync and async runs unless
  `TEST_DATABASE_URL` is set. For an apples-to-apples comparison, you
  probably want a dedicated persistent Postgres and to let the script
  clean up its own tables between runs.
