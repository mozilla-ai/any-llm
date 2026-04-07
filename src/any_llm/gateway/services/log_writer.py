"""Pluggable writers for usage events.

Two strategies are provided:

* ``SingleLogWriter`` writes each event to the DB immediately, one transaction per
  event. This is the historical behavior — simple, durable, blocking.
* ``BatchLogWriter`` queues events onto an ``asyncio.Queue`` and flushes them in
  batches of up to ``max_batch`` rows, or every ``max_wait_s`` seconds,
  whichever comes first. Writes happen in a dedicated background task so the
  request hot path just enqueues and returns. The lifespan guarantees a final
  drain on shutdown via ``queue.join()``.

Both writers treat flush failures as best-effort, consistent with the historical
behavior: the error is logged, rows are dropped, and the request is never
failed as a result. Use ``gateway_usage_log_rows{result="dropped"}`` to monitor.

Choose via ``GatewayConfig.log_writer_strategy``.
"""

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.log_config import logger
from any_llm.gateway.metrics import record_usage_log_flush, set_usage_log_queue_depth
from any_llm.gateway.models.entities import UsageLog, User

SessionFactory = Callable[[], AsyncSession]


class LogWriter(Protocol):
    async def put(self, entry: UsageLog) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class SingleLogWriter:
    """Write each UsageLog synchronously, one transaction per entry."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def put(self, entry: UsageLog) -> None:
        start = time.monotonic()
        success = True
        async with self._session_factory() as db:
            try:
                await _persist_entries(db, [entry])
            except Exception:
                await db.rollback()
                success = False
                logger.exception("SingleLogWriter: failed to persist entry")
        record_usage_log_flush("single", size=1, duration_s=time.monotonic() - start, success=success)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class BatchLogWriter:
    """Queue entries; a background task flushes up to max_batch per transaction.

    A batch flushes when either ``max_batch`` items have been accumulated
    or ``max_wait_s`` seconds have elapsed since the first item arrived.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        max_batch: int = 100,
        max_wait_s: float = 1.0,
    ) -> None:
        self._session_factory = session_factory
        self._max_batch = max_batch
        self._max_wait_s = max_wait_s
        self._queue: asyncio.Queue[UsageLog] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def put(self, entry: UsageLog) -> None:
        await self._queue.put(entry)
        set_usage_log_queue_depth(self._queue.qsize())

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="batch-log-writer")

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.join()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        set_usage_log_queue_depth(0)

    async def _run(self) -> None:
        while True:
            batch = await self._gather_batch()
            set_usage_log_queue_depth(self._queue.qsize())
            start = time.monotonic()
            success = True
            try:
                async with self._session_factory() as db:
                    try:
                        await _persist_entries(db, batch)
                    except Exception:
                        await db.rollback()
                        success = False
                        logger.exception("BatchLogWriter: flush failed (%d rows lost)", len(batch))
            finally:
                record_usage_log_flush(
                    "batch", size=len(batch), duration_s=time.monotonic() - start, success=success
                )
                for _ in batch:
                    self._queue.task_done()

    async def _gather_batch(self) -> list[UsageLog]:
        batch = [await self._queue.get()]
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._max_wait_s
        while len(batch) < self._max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return batch


def create_log_writer(strategy: str, session_factory: SessionFactory) -> LogWriter:
    """Build the configured writer. Unknown strategy falls back to 'single'."""
    match strategy:
        case "batch":
            return BatchLogWriter(session_factory)
        case "single":
            return SingleLogWriter(session_factory)
        case _:
            logger.warning("Unrecognized log_writer_strategy '%s', falling back to 'single'", strategy)
            return SingleLogWriter(session_factory)


async def _persist_entries(db: AsyncSession, entries: list[UsageLog]) -> None:
    """Flush a batch: attach the UsageLog rows to this session, sum spend per user, commit."""
    if not entries:
        return
    db.add_all(entries)

    deltas: dict[str, float] = {}
    for e in entries:
        if e.user_id and e.cost:
            deltas[e.user_id] = deltas.get(e.user_id, 0.0) + float(e.cost)

    for user_id, delta in deltas.items():
        await db.execute(
            update(User)
            .where(User.user_id == user_id, User.deleted_at.is_(None))
            .values(spend=User.spend + delta)
        )

    await db.commit()
