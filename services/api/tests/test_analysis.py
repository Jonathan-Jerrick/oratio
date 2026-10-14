import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from synth import passage, render

from oratio_api import analysis, judges
from oratio_api.baselines import load_baselines
from oratio_api.main import create_app
from oratio_api.settings import Settings
from speechgrade.pipeline import analyze
from speechgrade.schema import Report

AUDIO, WORDS = render(passage(), seed=5)
TEXT = " ".join(w.word.lower() for w in WORDS)


class FixedAligner:
    """Stands in for the ONNX model: returns the synthetic passage's exact word timings."""

    def align(self, samples: np.ndarray, transcript: str) -> list:
        return list(WORDS)


@pytest.fixture
def baseline_dir(tmp_path: Path) -> Path:
    sf.write(tmp_path / "speech__L0p.flac", AUDIO.samples, 16_000, subtype="PCM_16")
    words = [{"text": w.word, "start": w.start, "end": w.end} for w in WORDS]
    (tmp_path / "speech__L0p.json").write_text(json.dumps({"title": "Kennedy", "words": words}))
    load_baselines.cache_clear()
    return tmp_path


@pytest.fixture
def client(baseline_dir: Path, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    monkeypatch.setattr(analysis, "_encoder", lambda model_dir: FixedAligner())
    app = create_app(Settings(baseline_dir=baseline_dir))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _wav(samples: np.ndarray, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _events(body: str) -> list[tuple[str, str]]:
    blocks = [b.split("\n", 1) for b in body.strip().split("\n\n")]
    return [(head.removeprefix("event: "), data.removeprefix("data: ")) for head, data in blocks]


async def test_analyze_streams_stages_then_a_full_report(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/analyze",
            files={"audio": ("a.wav", _wav(AUDIO.samples), "audio/wav")},
            data={"transcript": TEXT},
        )

    events = _events(response.text)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in events][-1] == "report"
    stages = [json.loads(data)["stage"] for name, data in events if name == "stage"]
    assert stages[0] == "align" and "score" in stages
    report = Report.model_validate_json(events[-1][1])
    assert report.scores.overall == pytest.approx(100.0, abs=1.0)


@pytest.mark.parametrize(
    ("audio", "transcript", "baseline_id", "message"),
    [
        (b"not audio at all", TEXT, None, "not a WAV"),
        (
            _wav(AUDIO.samples),
            "completely unrelated words about cooking pasta tonight",
            None,
            "does not match",
        ),
        (_wav(AUDIO.samples), TEXT, "nope", "unknown baseline"),
        (_wav(AUDIO.samples[:4000]), TEXT, None, "allowed"),
    ],
)
async def test_analyze_reports_bad_requests_as_error_events(
    client: httpx.AsyncClient, audio: bytes, transcript: str, baseline_id: str | None, message: str
) -> None:
    data = {"transcript": transcript} | ({"baseline_id": baseline_id} if baseline_id else {})
    async with client:
        response = await client.post(
            "/api/analyze", files={"audio": ("a.wav", audio, "audio/wav")}, data=data
        )

    (_, payload), *_ = [e for e in _events(response.text) if e[0] == "error"]
    assert message in json.loads(payload)["detail"]


async def test_overlong_transcript_is_rejected(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.post(
            "/api/analyze",
            files={"audio": ("a.wav", _wav(AUDIO.samples), "audio/wav")},
            data={"transcript": "x" * 5001},
        )

    assert response.status_code == 422


async def test_baselines_are_listed(client: httpx.AsyncClient) -> None:
    async with client:
        listed = (await client.get("/api/baselines")).json()

    assert listed[0]["id"] == "speech__L0p"
    assert listed[0]["title"] == "Kennedy"


def _flawed_report() -> Report:
    quiet, words = render(
        [
            w if i not in range(3, 6) else type(w)(w.text, w.seconds, w.f0, 0.06, w.pause)
            for i, w in enumerate(passage())
        ],
        seed=1,
    )
    return analyze(quiet, words, AUDIO, WORDS)


async def test_judges_fall_back_to_templates_without_a_key() -> None:
    report = _flawed_report()
    async with httpx.AsyncClient() as http:
        result = await judges.judge(report, None, http)

    assert result.provider == "template"
    assert set(result.dimensions) == set(report.scores.dimensions)
    assert judges.fact_check(result.overall, report) == []


async def test_llm_numbers_are_checked_and_invented_ones_replaced() -> None:
    report = _flawed_report()
    region = report.regions[0]
    good = f"{region.region_id} lost {region.deduction} points."
    content = {
        "overall": f"Your score was 12.34 and {region.region_id} hurt.",
        "dimensions": {"vocal_energy": good},
    }

    def groq(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(groq)) as http:
        result = await judges.judge(report, "test-key", http)

    assert result.provider == "groq"
    assert "12.34" not in result.overall
    assert result.dimensions["vocal_energy"] == good
    assert set(result.dimensions) == set(report.scores.dimensions)


async def test_llm_failure_falls_back_to_templates() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(503))
    ) as http:
        result = await judges.judge(_flawed_report(), "test-key", http)

    assert result.provider == "template"


def test_fact_check_flags_unknown_regions_and_numbers() -> None:
    report = _flawed_report()

    assert judges.fact_check("R99 was 3.14159 dB too quiet", report) == [
        "unknown region R99",
        "unsupported number 3.14159",
    ]
