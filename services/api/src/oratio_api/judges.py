"""POST /api/agents/judges: rubric feedback written by an LLM, every number fact-checked."""

import json
import re
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from oratio_api.settings import Settings
from speechgrade.schema import Report

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
TIMEOUT_S = 20.0
MAX_TOKENS = 900
ABSOLUTE_TOLERANCE = 0.05
RELATIVE_TOLERANCE = 0.01
SCALE_MAXIMA = (10.0, 100.0)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
PROMPT = (
    "You are the head judge of a speech-delivery panel. Write feedback from the measured report "
    "below. Rules: use only numbers that appear in the report, cite region ids like R3 when you "
    "refer to a flaw, never invent a measurement, be specific and constructive. Return JSON with "
    'keys "overall" (120 to 200 words) and "dimensions" (an object with one or two sentences '
    "for each of: {dimensions}).\n\nREPORT:\n{report}"
)


class JudgeRequest(BaseModel):
    report: Report


class JudgeResponse(BaseModel):
    overall: str
    dimensions: dict[str, str]
    verified: bool
    provider: str


def _evidence(report: Report) -> dict[str, Any]:
    return {
        "scores": report.scores.model_dump(),
        "regions": [
            {
                "id": r.region_id,
                "code": r.code,
                "dimension": r.dimension,
                "start": r.start,
                "end": r.end,
                "words": r.words,
                "rationale": r.rationale,
                "fix": r.fix,
                "deduction": r.deduction,
            }
            for r in report.regions
        ],
    }


def _known_numbers(report: Report) -> list[float]:
    found = [float(n) for n in _NUMBER.findall(json.dumps(_evidence(report)))]
    return [*found, *SCALE_MAXIMA]


def fact_check(text: str, report: Report) -> list[str]:
    """Numbers and region ids in the text that the report does not contain."""
    known = _known_numbers(report)
    ids = {r.region_id for r in report.regions}
    problems = [f"unknown region {rid}" for rid in re.findall(r"\bR\d+\b", text) if rid not in ids]
    for raw in _NUMBER.findall(re.sub(r"\bR\d+\b", "", text)):
        value = float(raw)
        if not any(
            abs(value - k) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(k)) for k in known
        ):
            problems.append(f"unsupported number {raw}")
    return problems


def template(report: Report) -> JudgeResponse:
    worst = sorted(report.regions, key=lambda r: -r.deduction)[:3]
    if worst:
        focus = " ".join(f"{r.region_id} ({r.words}): {r.fix}" for r in worst)
        overall = (
            f"Overall score {report.scores.overall:.1f} out of 100. "
            f"The largest deductions come from: {focus}"
        )
    else:
        overall = (
            f"Overall score {report.scores.overall:.1f} out of 100. No flaw regions were found."
        )
    dimensions = {}
    for name, score in report.scores.dimensions.items():
        mine = [r for r in report.regions if r.dimension == name]
        dimensions[name] = f"{score:.1f} of 10. " + (
            mine[0].rationale if mine else "No issues found."
        )
    return JudgeResponse(overall=overall, dimensions=dimensions, verified=True, provider="template")


async def judge(report: Report, api_key: str | None, client: httpx.AsyncClient) -> JudgeResponse:
    if not api_key:
        return template(report)
    prompt = PROMPT.format(
        dimensions=", ".join(report.scores.dimensions), report=json.dumps(_evidence(report))
    )
    try:
        response = await client.post(
            GROQ_URL,
            headers={"authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "seed": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        content = json.loads(response.json()["choices"][0]["message"]["content"])
        overall = str(content["overall"])
        dimensions = {k: str(v) for k, v in dict(content["dimensions"]).items()}
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return template(report)
    fallback = template(report)
    if fact_check(overall, report):
        overall = fallback.overall
    checked = {
        name: (text if not fact_check(text, report) else fallback.dimensions.get(name, ""))
        for name, text in dimensions.items()
        if name in report.scores.dimensions
    }
    for name in report.scores.dimensions:
        checked.setdefault(name, fallback.dimensions[name])
    return JudgeResponse(overall=overall, dimensions=checked, verified=True, provider="groq")


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["agents"])

    @router.post("/agents/judges")
    async def judges(request: JudgeRequest) -> JudgeResponse:
        async with httpx.AsyncClient() as client:
            return await judge(request.report, settings.groq_api_key, client)

    return router
