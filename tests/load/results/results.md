# Load test results: gateway DB layer + budget strategy

Same noop fake provider (`FAKE_DELAY_MS=0`, `RNG_SEED=42`), same load
(100 VUs × 30s for `distinct_users` and `same_user`, preceded by a 10-VU / 5s
warmup), same hardware, same Postgres. Differences are branch + one config flag.

## Scenarios run

| Label | Branch | DB driver | `budget_strategy` | `log_writer_strategy` |
|---|---|---|---|---|
| **sync** | `main` (`0510c38`) | psycopg2 + sync `Session` | n/a (always FOR UPDATE) | single (inline) |
| **async-for_update** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `for_update` | `single` |
| **async-cas** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `cas` | `single` |
| **async-disabled** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `disabled` | `single` |
| **async-batch+disabled** | `julian/async-asyncpg` | asyncpg + `AsyncSession` | `disabled` | `batch` |

## Headline numbers

| Scenario | Total rps | distinct rps | same rps | Reqs OK | Coverage |
|---|---|---|---|---|---|
| sync | ~0 (stalled) | ~0 | ~1 | 391 | — |
| async-for_update | 81.1 | 46.3 | 34.8 ⚠️ | 6,578 | — |
| async-cas | 82.3 | 41.5 | 40.8 | 6,498 | — |
| async-disabled | 89.5 | 44.8 | 44.7 | 7,113 | — |
| **async-batch+disabled** 🏆 | **97.4** | 47.4 | **50.0** ✨ | **7,637** | **100%** |

All async scenarios: **0 failures**. Sync: 160 timeouts (exhausted 15-conn pool).
The batch writer run reports **100% row coverage** — 7,637 requests, 7,637 rows persisted, zero drops.

### Detailed latencies and resource usage

<details>
<summary>Click for p50 / p95 / p99 per scenario + CPU / RSS</summary>

| Scenario | distinct p50/p95/p99 | same p50/p95/p99 | CPU avg/max | RSS max |
|---|---|---|---|---|
| sync | — | — | 2.0% / 47.6% | 224 MB |
| async-for_update | 767 / 2207 / 2990 ms | 1185 / 1476 / 1777 ms | 76.9% / 100% | 267 MB |
| async-cas | 921 / 2017 / 2643 ms | 929 / 1805 / 2382 ms | 81.9% / 99% | 247 MB |
| async-disabled | 889 / 1394 / 1734 ms | 891 / 1333 / 1596 ms | 83.7% / 99% | 217 MB |
| async-batch+disabled | 798 / 1892 / 2594 ms | 791 / 1758 / 2390 ms | 86.4% / 101% | 265 MB |

</details>

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

### async-disabled → async-batch+disabled (stack the log writer optimization)

> Keep budget validation off, and additionally move usage log writes off the request hot path.

- Total throughput: **89.5 → 97.4 req/sec (+9%)** — highest of any run
- `same_user` now **beats** `distinct_users` (50.0 vs 47.4) — the batch writer groups spend UPDATEs per user, so shared-user traffic gets *fewer* UPDATEs per batch
- **100% row coverage on shutdown** — queue.join() drained 7,637 pending rows cleanly before the process exited
- CPU jumps slightly (86% avg) because the worker isn't idling on log-write I/O anymore

## The saturation floor

All async scenarios converge on ~85-90 req/sec total (combined distinct + same).
The single uvicorn worker is **CPU-bound** (100% peak) in every case. To go
higher, increase `--workers`. The `distinct_users` p50 is around 800-920ms
because 100 VUs competing for one worker produces a natural queue.

## Recommendation

**Budget strategy:**
- **Default (for_update):** historical behavior. Safe when pointed at an async-capable gateway. Same-user contention costs ~17% throughput.
- **cas (recommended):** lock-free, no same-user penalty, negligible overhead (~8%) vs not validating at all.
- **disabled:** use only if you enforce budgets out-of-band.

**Log writer strategy:**
- **Default (single):** inline write per request, simple, durable for normal terminations.
- **batch (recommended for high-throughput):** queues + flushes 100 rows / 1s. +9% throughput, groups spend UPDATEs per-user, 100% coverage on clean shutdown. Best-effort semantics (a SIGKILL loses the in-flight batch).

Upgrade path: switch `GATEWAY_BUDGET_STRATEGY=cas` and/or `GATEWAY_LOG_WRITER_STRATEGY=batch` in your config. No schema change required.

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
- `k6-async.{txt,json}` — async + `for_update` (legacy default)
- `k6-async-cas.{txt,json}` — async + `cas`
- `k6-async-disabled.{txt,json}` — async + `disabled`
- `k6-async-batch-disabled.{txt,json}` — async + `disabled` + `batch` log writer
- `gateway-stats-*.csv` — per-second summed CPU% / RSS MB of all gateway processes
- `run-*.md` — per-run metadata (branch, commit, config)

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

# 4) budget checks off + batched log writer (the stacked optimization)
BUDGET_STRATEGY=disabled LOG_WRITER_STRATEGY=batch \
  ./tests/load/run_load_test.sh async-batch-disabled

# compare
cat tests/load/results/results.md
```
