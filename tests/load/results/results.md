# Load test results: gateway DB layer + budget strategy

Same noop fake provider (`FAKE_DELAY_MS=0`, `RNG_SEED=42`), same load
(100 VUs × 30s for `distinct_users` and `same_user`, preceded by a 10-VU / 5s
warmup), same hardware, same Postgres. Differences are branch + one config flag.

## Scenarios run

| Label | Branch | DB driver | `budget_strategy` |
|---|---|---|---|
| **sync** | `main` (`0510c38`) | psycopg2 + sync `Session` | n/a (always FOR UPDATE) |
| **async-for_update** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `for_update` |
| **async-cas** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `cas` |
| **async-disabled** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `disabled` |

## Headline numbers

| Scenario | successful reqs | distinct rps | distinct p50 | distinct p95 | distinct p99 | same rps | same p50 | same p95 | same p99 | CPU avg/max |
|---|---|---|---|---|---|---|---|---|---|---|
| **sync** | **391** (+160 timeouts) | stalled — pool exhausted | — | — | — | ~1 | — | — | — | 2% / 47.6% |
| **async-for_update** | 6,578 | 46.3 | 767 ms | 2207 ms | 2990 ms | **34.8** | 1185 ms | 1476 ms | 1777 ms | 77% / 100% |
| **async-cas** | 6,498 | 41.5 | 921 ms | 2017 ms | 2643 ms | **40.8** | 929 ms | 1805 ms | 2382 ms | 82% / 99% |
| **async-disabled** | **7,113** | 44.8 | 889 ms | 1394 ms | 1734 ms | 44.7 | 891 ms | 1333 ms | 1596 ms | 84% / 99% |

All async scenarios: **0 failures**. Sync: 160 timeouts.

## What each transition reveals

### sync → async-for_update (driver swap, same strategy)

> The biggest win. Same `FOR UPDATE` logic, but now it doesn't block the event loop.

- **17× more successful requests** (6,578 vs 391)
- **Zero failures** vs 160 timeouts
- Gateway goes from 2% CPU (starved on pool) to 77% CPU (doing real work)
- Clean finish at 1m14s vs sync stuck at 2m17s

### async-for_update → async-cas (same driver, strategy change)

> Eliminates the last bit of same-user contention by dropping FOR UPDATE entirely.

- `same_user` throughput: **34.8 → 40.8 req/sec** (+17%)
- `same_user` p99: **1777 → 2382 ms** — nope, the latency distribution shifts differently
- **Key insight:** the gap between `distinct_users` (46.3) and `same_user` (34.8) under `for_update` is the contention cost. `cas` closes that gap: `distinct=41.5, same=40.8` — **no penalty for concurrent requests on the same user**
- Total throughput roughly the same (async-for_update was already CPU-bound); the improvement shows up as latency consistency across scenarios

### async-cas → async-disabled (skip validation entirely)

> Upper bound: what does the gateway look like with zero budget overhead?

- **+7-9% throughput** (41-44 → 44-45 req/sec)
- p95 latency drops: `distinct 2017 → 1394 ms`, `same 1805 → 1333 ms`
- Tells us `cas` costs roughly 8% vs no validation at all — cheap
- Useful as a ceiling to measure future optimizations against

## The saturation floor

All async scenarios converge on ~85-90 req/sec total (combined distinct + same).
The single uvicorn worker is **CPU-bound** (100% peak) in every case. To go
higher, increase `--workers`. The `distinct_users` p50 is around 800-920ms
because 100 VUs competing for one worker produces a natural queue.

## Recommendation

- **Default (for_update):** historical behavior. Safe when pointed at an async-capable gateway. Same-user contention costs ~17% throughput.
- **cas (recommended):** lock-free, no same-user penalty, negligible overhead (~8%) vs not validating at all.
- **disabled:** use only if you enforce budgets out-of-band.

Upgrade path: switch `GATEWAY_BUDGET_STRATEGY=cas` in your config once you're
comfortable with the changeover. No schema change required.

## Config

| | value |
|---|---|
| VUs | 100 |
| duration per scenario | 30s |
| workers | 1 (single event loop) |
| fake upstream delay | 0 ms (noop) |
| RNG seed | 42 |
| warmup | 10 VUs × 5s |

## Raw artifacts

- `k6-sync.txt` — k6 output for sync run (partial; killed during stuck teardown)
- `k6-async.txt` / `k6-async.json` — async + `for_update` (legacy default)
- `k6-async-cas.txt` / `k6-async-cas.json` — async + `cas`
- `k6-async-disabled.txt` / `k6-async-disabled.json` — async + `disabled`
- `gateway-stats-*.csv` — per-second summed CPU% / RSS MB of all gateway processes

## Reproducing

```bash
git checkout main
./tests/load/run_load_test.sh sync

git checkout julian/async-asyncpg

# 1) legacy default
BUDGET_STRATEGY=for_update ./tests/load/run_load_test.sh async

# 2) lock-free
BUDGET_STRATEGY=cas ./tests/load/run_load_test.sh async-cas

# 3) budget checks off
BUDGET_STRATEGY=disabled ./tests/load/run_load_test.sh async-disabled

# compare
cat tests/load/results/results.md
```
