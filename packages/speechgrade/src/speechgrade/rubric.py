"""The scoring rubric: thresholds, weights and wording, loaded from packaged data."""

from functools import lru_cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict

from speechgrade.hashing import content_hash


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rule(_Frozen):
    code: str
    dimension: str
    feature: str
    direction: Literal[-1, 1]
    z_threshold: float
    weight: float


class Events(_Frozen):
    fillers: tuple[str, ...]
    z: float
    z_threshold: float
    weight: float


class Scoring(_Frozen):
    z_cap: float
    duration_reference_s: float
    duration_weight_floor: float
    merge_gap_words: int


class FeatureInfo(_Frozen):
    label: str
    unit: str


class Rubric(_Frozen):
    version: str
    sigma_good: dict[str, float]
    rules: tuple[Rule, ...]
    events: Events
    dimensions: tuple[str, ...]
    scoring: Scoring
    features: dict[str, FeatureInfo]
    rationales: dict[str, str]
    fixes: dict[str, str]

    @property
    def sha256(self) -> str:
        return content_hash(self.model_dump(mode="json"))


@lru_cache(maxsize=1)
def load_rubric() -> Rubric:
    text = resources.files("speechgrade").joinpath("data/rubric.json").read_text(encoding="utf-8")
    return Rubric.model_validate_json(text)
