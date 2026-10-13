"""Dataset configuration: severity levels, eligible spans and gate thresholds."""

from functools import lru_cache
from importlib import resources

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Level(_Frozen):
    regions: tuple[int, int]
    magnitude: tuple[float, float]
    coverage: float
    global_flaw: bool


class Gates(_Frozen):
    max_clipped_share: float
    clip_level: float
    max_digital_silence_s: float
    duration_tolerance_s: float


class DatasetConfig(_Frozen):
    version: str
    levels: dict[str, Level]
    span_kinds: dict[str, tuple[str, ...]]
    global_codes: tuple[str, ...]
    phrase_words: tuple[int, int]
    pause_min_s: float
    no_pause_max_s: float
    min_word_s: float
    sweep_magnitudes: tuple[float, ...]
    gates: Gates

    def kind_of(self, code: str) -> str:
        for kind, codes in self.span_kinds.items():
            if code in codes:
                return kind
        raise KeyError(code)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(sorted(code for codes in self.span_kinds.values() for code in codes))


@lru_cache(maxsize=1)
def load_config() -> DatasetConfig:
    text = resources.files("oratio_data").joinpath("data/dataset.json").read_text(encoding="utf-8")
    return DatasetConfig.model_validate_json(text)
