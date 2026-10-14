"""POST /api/analyze: validate, align, measure, ground, score; progress as server-sent events."""

import json
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from oratio_api.baselines import BaselineError, choose, load_baselines
from oratio_api.settings import Settings
from speechgrade.align import AlignmentError
from speechgrade.audio import MAX_BYTES, AudioError, decode_wav
from speechgrade.encoder import CtcEncoder
from speechgrade.pipeline import analyze

MAX_TRANSCRIPT_CHARS = 5000
ENCODER_FILE = "wav2vec2_base_int8.onnx"


@lru_cache(maxsize=1)
def _encoder(model_dir: Path) -> CtcEncoder:
    return CtcEncoder.from_onnx(model_dir / ENCODER_FILE)


def _event(name: str, payload: object) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"event: {name}\ndata: {data}\n\n"


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["analysis"])

    @router.post("/analyze")
    async def analyze_route(
        audio: Annotated[UploadFile, File()],
        transcript: Annotated[str, Form(max_length=MAX_TRANSCRIPT_CHARS)],
        baseline_id: Annotated[str | None, Form(max_length=200)] = None,
    ) -> StreamingResponse:
        data = await audio.read(MAX_BYTES + 1)

        async def events() -> AsyncIterator[str]:
            started = time.perf_counter()

            def stage(name: str, status: str) -> str:
                elapsed = round((time.perf_counter() - started) * 1000)
                return _event("stage", {"stage": name, "status": status, "elapsed_ms": elapsed})

            try:
                participant = decode_wav(data)
                baseline, _ = choose(load_baselines(settings.baseline_dir), transcript, baseline_id)
                yield stage("align", "start")
                encoder = _encoder(settings.model_dir)
                words = await run_in_threadpool(encoder.align, participant.samples, baseline.text)
                yield stage("align", "done")
                yield stage("features", "start")
                report = await run_in_threadpool(
                    analyze, participant, words, baseline.audio, list(baseline.words)
                )
                yield stage("features", "done")
                yield stage("detect", "done")
                yield stage("score", "done")
                yield _event("report", report.model_dump_json())
            except (AudioError, BaselineError, AlignmentError) as error:
                yield _event("error", {"detail": str(error)})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"cache-control": "no-cache"}
        )

    @router.get("/baselines")
    def baselines() -> list[dict[str, str]]:
        return [
            {"id": b.baseline_id, "title": b.title, "text": b.text}
            for b in load_baselines(settings.baseline_dir).values()
        ]

    return router
