"""The report: everything the API returns and the dashboard renders."""

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Measurement(_Model):
    feature: str
    label: str
    unit: str
    participant: float
    baseline: float
    delta: float
    sigma_good: float
    z: float
    z_threshold: float


class Region(_Model):
    region_id: str
    code: str
    dimension: str
    start: float
    end: float
    words: str
    measurement: Measurement | None
    z: float
    deduction: float
    rationale: str
    fix: str


class WordRecord(_Model):
    word: str
    start: float
    end: float
    score: float
    baseline_index: int | None


class Scores(_Model):
    dimensions: dict[str, float]
    overall: float


class Provenance(_Model):
    speechgrade_version: str
    rubric_version: str
    rubric_sha256: str
    report_sha256: str


class Report(_Model):
    schema_version: str
    participant_seconds: float
    baseline_seconds: float
    words: tuple[WordRecord, ...]
    baseline_words: tuple[WordRecord, ...]
    regions: tuple[Region, ...]
    scores: Scores
    tracks: dict[str, dict[str, str]]
    provenance: Provenance
