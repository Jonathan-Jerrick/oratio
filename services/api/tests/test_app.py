import json
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest

from oratio_api.main import create_app
from oratio_api.settings import Settings
from oratio_api.spike.bench import SAMPLE_RATE, synth_speech

ORIGIN = "https://oratio.example"


def _client(settings: Settings) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with _client(Settings(cors_origins=[ORIGIN])) as http:
        yield http


@pytest.fixture
async def spike_client() -> AsyncIterator[httpx.AsyncClient]:
    async with _client(Settings(enable_spike=True)) as http:
        yield http


async def test_health_reports_version_and_digest(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "version", "arch", "image_digest"}


async def test_cors_allows_configured_origin_only(client: httpx.AsyncClient) -> None:
    allowed = await client.get("/api/health", headers={"origin": ORIGIN})
    other = await client.get("/api/health", headers={"origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-origin" not in other.headers


async def test_spike_routes_are_absent_by_default(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/spike/stream")).status_code == 404
    assert (await client.post("/api/spike/bench", json={})).status_code == 404


async def test_stream_emits_ordered_stage_events_then_done(spike_client: httpx.AsyncClient) -> None:
    response = await spike_client.get("/api/spike/stream", params={"events": 3, "interval_ms": 0})

    assert response.headers["content-type"].startswith("text/event-stream")
    blocks = [block.split("\n") for block in response.text.strip().split("\n\n")]
    names = [block[0].removeprefix("event: ") for block in blocks]
    stages = [json.loads(block[1].removeprefix("data: ")).get("stage") for block in blocks[:-1]]
    assert names == ["stage", "stage", "stage", "done"]
    assert stages == [0, 1, 2]


@pytest.mark.parametrize("body", [{"seconds": 171}, {"seconds": 0}, {"variant": "fp64"}])
async def test_bench_rejects_out_of_range_input(
    spike_client: httpx.AsyncClient, body: dict[str, object]
) -> None:
    assert (await spike_client.post("/api/spike/bench", json=body)).status_code == 422


async def test_bedrock_rejects_malformed_model_id(spike_client: httpx.AsyncClient) -> None:
    response = await spike_client.post("/api/spike/bedrock", json={"model_id": "bad id; rm"})

    assert response.status_code == 422


def test_synth_speech_is_deterministic_and_bounded() -> None:
    first = synth_speech(2.0)

    assert first.dtype == np.float32
    assert first.shape == (2 * SAMPLE_RATE,)
    assert np.array_equal(first, synth_speech(2.0))
    assert float(np.max(np.abs(first))) <= 0.5 + 1e-6
