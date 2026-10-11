"""Spike endpoints: SSE through the Function URL and stage timings."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from oratio_api.settings import Settings
from oratio_api.spike.bench import BenchResult, Variant, run_bench

MAX_AUDIO_S = 170.0


class BenchRequest(BaseModel):
    seconds: float = Field(default=60.0, ge=1.0, le=MAX_AUDIO_S)
    variant: Variant = "int8"
    chunk_seconds: float | None = Field(default=None, ge=5.0, le=MAX_AUDIO_S)


async def _stage_events(events: int, interval_ms: int) -> AsyncIterator[str]:
    start = time.perf_counter()
    for index in range(events):
        payload = {"stage": index, "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 1)}
        yield f"event: stage\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(interval_ms / 1000.0)
    yield "event: done\ndata: {}\n\n"


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["spike"])

    @router.get("/stream")
    async def stream(
        events: Annotated[int, Query(ge=1, le=20)] = 5,
        interval_ms: Annotated[int, Query(ge=0, le=2000)] = 400,
    ) -> StreamingResponse:
        return StreamingResponse(
            _stage_events(events, interval_ms),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache"},
        )

    @router.post("/bench")
    def bench(request: BenchRequest) -> BenchResult:
        return run_bench(
            settings.model_dir, request.seconds, request.variant, request.chunk_seconds
        )

    return router
