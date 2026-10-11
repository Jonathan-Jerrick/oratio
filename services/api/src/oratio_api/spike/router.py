"""Spike endpoints: SSE through the Function URL, stage timings, Bedrock latency."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from oratio_api.settings import Settings
from oratio_api.spike.bench import BenchResult, Variant, run_bench

MAX_AUDIO_S = 170.0
BEDROCK_MAX_TOKENS = 64
BEDROCK_PROMPT = "In one sentence, what makes a pause effective in a speech?"


class BenchRequest(BaseModel):
    seconds: float = Field(default=60.0, ge=1.0, le=MAX_AUDIO_S)
    variant: Variant = "int8"
    chunk_seconds: float | None = Field(default=None, ge=5.0, le=MAX_AUDIO_S)


class BedrockRequest(BaseModel):
    model_id: str = Field(pattern=r"^[a-z0-9.:\-]{1,100}$")


class BedrockResult(BaseModel):
    model_id: str
    first_token_ms: float
    total_ms: float
    output_chars: int


@lru_cache(maxsize=1)
def _bedrock(region: str) -> Any:
    import boto3

    return boto3.client("bedrock-runtime", region_name=region)


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

    @router.post("/bedrock")
    def bedrock(request: BedrockRequest) -> BedrockResult:
        from botocore.exceptions import BotoCoreError, ClientError

        start = time.perf_counter()
        first_token_ms: float | None = None
        chars = 0
        try:
            response = _bedrock(settings.bedrock_region).converse_stream(
                modelId=request.model_id,
                messages=[{"role": "user", "content": [{"text": BEDROCK_PROMPT}]}],
                inferenceConfig={"maxTokens": BEDROCK_MAX_TOKENS, "temperature": 0.0},
            )
            for event in response["stream"]:
                text = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
                if text:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    chars += len(text)
        except (BotoCoreError, ClientError) as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", type(error).__name__)
            raise HTTPException(status_code=502, detail=code) from error
        total_ms = (time.perf_counter() - start) * 1000.0
        return BedrockResult(
            model_id=request.model_id,
            first_token_ms=first_token_ms if first_token_ms is not None else total_ms,
            total_ms=total_ms,
            output_chars=chars,
        )

    return router
