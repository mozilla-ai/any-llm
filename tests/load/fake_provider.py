"""Fake OpenAI-compatible upstream for load testing the gateway.

Returns a canned ChatCompletion / CreateEmbedding / Messages response with a
configurable per-request delay. Does not make any real LLM calls.

Usage:
    python tests/load/fake_provider.py --delay-ms 200 --jitter-sigma 0.4

Run `python tests/load/fake_provider.py --help` for all options.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import click
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class DelayConfig:
    """Per-request delay distribution for the fake provider.

    Attributes:
        delay_ms: target median delay in ms. 0 means noop (return immediately).
        jitter_sigma: log-normal sigma around the median. 0 means fixed delay.
          Realistic LLM-ish values:
            0.2  (tight: p95 ≈ 1.4x median)
            0.4  (moderate: p95 ≈ 1.9x median)
            0.6  (long-tail: p95 ≈ 2.7x median)
        delay_min_ms: hard floor applied after sampling. 0 means no floor.
        delay_max_ms: hard ceiling applied after sampling. 0 means no ceiling.
    """

    delay_ms: float
    jitter_sigma: float
    delay_min_ms: float
    delay_max_ms: float

    def sample_ms(self) -> float:
        """Sample a per-request delay in ms from a log-normal distribution."""
        if self.delay_ms <= 0:
            return 0.0
        if self.jitter_sigma <= 0:
            sample = self.delay_ms
        else:
            # log-normal with median = delay_ms:
            # mu = ln(median), sigma = jitter_sigma  =>  exp(mu) = median
            sample = math.exp(random.gauss(math.log(self.delay_ms), self.jitter_sigma))
        if self.delay_min_ms > 0:
            sample = max(sample, self.delay_min_ms)
        if self.delay_max_ms > 0:
            sample = min(sample, self.delay_max_ms)
        return sample

    async def sleep(self) -> None:
        ms = self.sample_ms()
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)


def _chat_completion_response(model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _embedding_response(model: str, inputs: Any) -> dict[str, Any]:
    count = len(inputs) if isinstance(inputs, list) else 1
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 8} for i in range(count)],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def _anthropic_message_response(model: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _make_routes(delay: DelayConfig) -> APIRouter:
    router = APIRouter()

    @router.post("/chat/completions")
    async def chat_completions(body: dict[str, Any]) -> JSONResponse:
        await delay.sleep()
        return JSONResponse(_chat_completion_response(body.get("model", "fake")))

    @router.post("/embeddings")
    async def embeddings(body: dict[str, Any]) -> JSONResponse:
        await delay.sleep()
        return JSONResponse(_embedding_response(body.get("model", "fake"), body.get("input", "")))

    @router.post("/messages")
    async def messages(body: dict[str, Any]) -> JSONResponse:
        await delay.sleep()
        return JSONResponse(_anthropic_message_response(body.get("model", "fake")))

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "delay_ms": str(delay.delay_ms),
            "jitter_sigma": str(delay.jitter_sigma),
        }

    return router


def make_app(delay: DelayConfig) -> FastAPI:
    app = FastAPI(title="fake-llm-provider")
    # Mount routes under both / and /v1 so the fake works regardless of whether
    # the caller's api_base already includes /v1.
    app.include_router(_make_routes(delay))
    app.include_router(_make_routes(delay), prefix="/v1")
    return app


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=9999, type=int, help="Port to bind")
@click.option(
    "--delay-ms",
    default=0.0,
    type=float,
    help="Target median per-request delay in ms (0 = noop, no sleep).",
)
@click.option(
    "--jitter-sigma",
    default=0.0,
    type=float,
    help="Log-normal sigma for jitter around the median (0 = fixed delay). "
    "Typical values: 0.2 (tight), 0.4 (moderate), 0.6 (long-tail).",
)
@click.option(
    "--delay-min-ms",
    default=0.0,
    type=float,
    help="Hard floor for sampled delay in ms (0 = no floor).",
)
@click.option(
    "--delay-max-ms",
    default=0.0,
    type=float,
    help="Hard ceiling for sampled delay in ms (0 = no ceiling).",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Seed for the jitter RNG. Pass the same value across runs for "
    "reproducible delay sampling.",
)
@click.option("--log-level", default="warning", help="Uvicorn log level")
def main(
    host: str,
    port: int,
    delay_ms: float,
    jitter_sigma: float,
    delay_min_ms: float,
    delay_max_ms: float,
    seed: int | None,
    log_level: str,
) -> None:
    """Run the fake LLM provider on the given host:port."""
    if seed is not None:
        random.seed(seed)
    delay = DelayConfig(
        delay_ms=delay_ms,
        jitter_sigma=jitter_sigma,
        delay_min_ms=delay_min_ms,
        delay_max_ms=delay_max_ms,
    )
    app = make_app(delay)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
