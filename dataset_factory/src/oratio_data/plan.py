"""Seeded flaw planning: where each flaw goes and how strong it is, per severity level."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from oratio_data.config import DatasetConfig, Level, load_config
from oratio_data.injectors import Flaw, PlanError
from speechgrade.align import WordSpan

MAX_TRIES = 200
EVENT_CODES = frozenset({"FLU-REP", "FLU-FALSE", "FLU-OMIT", "FLU-FILL"})


@dataclass(frozen=True, slots=True)
class Placement:
    """A flaw together with the word indices it occupies, for overlap checks and labels."""

    flaw: Flaw
    first_word: int
    last_word: int
    is_global: bool = False


def _phrase_length(level: Level, words: int, regions: int, config: DatasetConfig) -> int:
    low, high = config.phrase_words
    return int(np.clip(round(level.coverage * words / regions), low, high))


def _candidates(
    code: str, words: list[WordSpan], length: int, config: DatasetConfig
) -> list[tuple[int, int]]:
    """Word index ranges where this code can be placed."""
    kind = config.kind_of(code)
    last = len(words) - 1
    if kind == "phrase":
        return [(i, i + length - 1) for i in range(len(words) - length + 1)]
    if kind == "word":
        return [(i, i) for i, w in enumerate(words) if w.end - w.start >= config.min_word_s]
    gaps = [(i, words[i + 1].start - words[i].end) for i in range(last)]
    if kind == "pause":
        return [(i, i) for i, gap in gaps if gap >= config.pause_min_s]
    return [(i, i) for i, gap in gaps if gap <= config.no_pause_max_s]


def _flaw(code: str, first: int, last: int, magnitude: float, words: list[WordSpan]) -> Flaw:
    if code == "PAU-DROP":
        return Flaw(code, words[first].end, words[first + 1].start, magnitude)
    spans = tuple((w.start, w.end) for w in words[first : last + 1])
    return Flaw(code, words[first].start, words[last].end, magnitude, spans)


def _free(first: int, last: int, taken: list[Placement]) -> bool:
    return all(last < p.first_word - 1 or first > p.last_word + 1 for p in taken if not p.is_global)


def plan_level(
    words: list[WordSpan],
    level_name: str,
    seed: int,
    has_filler: bool = False,
    config: DatasetConfig | None = None,
) -> list[Placement]:
    """Sample a plan for one severity level; the same seed always gives the same plan."""
    config = config or load_config()
    level = config.levels[level_name]
    rng = np.random.default_rng(seed)
    codes = [c for c in config.codes if has_filler or c != "FLU-FILL"]
    regions = int(rng.integers(level.regions[0], level.regions[1] + 1))
    length = _phrase_length(level, len(words), regions, config)
    placements: list[Placement] = []
    for _ in range(MAX_TRIES * regions):
        if len(placements) == regions:
            break
        code = codes[int(rng.integers(len(codes)))]
        options = [c for c in _candidates(code, words, length, config) if _free(*c, placements)]
        if not options:
            continue
        first, last = options[int(rng.integers(len(options)))]
        magnitude = float(rng.uniform(*level.magnitude))
        placements.append(Placement(_flaw(code, first, last, magnitude, words), first, last))
    if len(placements) < regions:
        raise PlanError(f"{level_name}: could place only {len(placements)} of {regions} regions")
    if level.global_flaw:
        code = config.global_codes[int(rng.integers(len(config.global_codes)))]
        magnitude = float(rng.uniform(*level.magnitude))
        whole = _flaw(code, 0, len(words) - 1, magnitude, words)
        placements.append(Placement(whole, 0, len(words) - 1, is_global=True))
    return sorted(placements, key=lambda p: (p.is_global, p.flaw.start))


def plan_sweep(
    words: list[WordSpan], code: str, seed: int, config: DatasetConfig | None = None
) -> list[Placement]:
    """One flaw at one place, at each sweep magnitude: a dose-response series."""
    config = config or load_config()
    rng = np.random.default_rng(seed)
    options = _candidates(code, words, config.phrase_words[0] + 1, config)
    if not options:
        raise PlanError(f"{code}: no eligible span in this passage")
    first, last = options[int(rng.integers(len(options)))]
    return [
        Placement(_flaw(code, first, last, magnitude, words), first, last)
        for magnitude in config.sweep_magnitudes
    ]


def coverage(placements: list[Placement], duration: float) -> float:
    """Share of the clip's source time covered by non-global flaws."""
    local = [p.flaw for p in placements if not p.is_global]
    return sum(f.end - f.start for f in local) / duration


def validate_plan(
    placements: list[Placement], level_name: str, config: DatasetConfig | None = None
) -> None:
    """Refuse a plan that breaks its level's constraints, wherever the plan came from."""
    config = config or load_config()
    level = config.levels[level_name]
    local = [p for p in placements if not p.is_global]
    if not level.regions[0] <= len(local) <= level.regions[1]:
        raise PlanError(f"{level_name}: {len(local)} regions, allowed {level.regions}")
    if sum(p.is_global for p in placements) != int(level.global_flaw):
        raise PlanError(f"{level_name}: wrong number of global flaws")
    for placement in placements:
        low, high = level.magnitude
        if placement.flaw.code not in config.codes:
            raise PlanError(f"{level_name}: unknown code {placement.flaw.code}")
        if not low <= placement.flaw.magnitude <= high:
            raise PlanError(f"{level_name}: magnitude {placement.flaw.magnitude:.2f} outside band")
    ordered = sorted(local, key=lambda p: p.first_word)
    for left, right in pairwise(ordered):
        if right.first_word <= left.last_word + 1:
            raise PlanError(f"{level_name}: regions overlap or touch at word {right.first_word}")
