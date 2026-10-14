from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from synth import passage, render

from oratio_data.config import load_config
from oratio_data.gates import check_clip
from oratio_data.injectors import PlanError, apply_plan
from oratio_data.plan import Placement, coverage, plan_level, plan_sweep, validate_plan

CONFIG = load_config()
AUDIO, WORDS = render(passage() * 5, seed=4)
LEVELS = sorted(CONFIG.levels)


def test_config_covers_the_whole_taxonomy_once() -> None:
    assert len(CONFIG.codes) == len(set(CONFIG.codes)) == 23
    assert set(CONFIG.global_codes) <= set(CONFIG.span_kinds["phrase"])


@pytest.mark.parametrize("level", LEVELS)
def test_sampled_plans_satisfy_their_level(level: str) -> None:
    for seed in range(20):
        plan = plan_level(WORDS, level, seed)

        validate_plan(plan, level)
        assert "FLU-FILL" not in {p.flaw.code for p in plan}


def test_plans_are_reproducible_and_vary_with_the_seed() -> None:
    assert plan_level(WORDS, "L3", 7) == plan_level(WORDS, "L3", 7)
    assert plan_level(WORDS, "L3", 7) != plan_level(WORDS, "L3", 8)


def test_coverage_and_strength_grow_with_the_level() -> None:
    def mean(level: str, measure: str) -> float:
        plans = [plan_level(WORDS, level, seed) for seed in range(15)]
        if measure == "coverage":
            return float(np.mean([coverage(p, AUDIO.seconds) for p in plans]))
        return float(np.mean([q.flaw.magnitude for p in plans for q in p]))

    for measure in ("coverage", "magnitude"):
        values = [mean(level, measure) for level in LEVELS]
        assert values == sorted(values)
        assert values[-1] > 2 * values[0]


def test_only_the_top_level_carries_a_global_flaw() -> None:
    assert not any(p.is_global for p in plan_level(WORDS, "L4", 1))
    (whole,) = [p for p in plan_level(WORDS, "L5", 1) if p.is_global]
    assert whole.flaw.code in CONFIG.global_codes
    assert (whole.first_word, whole.last_word) == (0, len(WORDS) - 1)


def test_pause_codes_land_on_matching_gaps() -> None:
    def gap(placement: Placement) -> float:
        return WORDS[placement.first_word + 1].start - WORDS[placement.first_word].end

    placements = [p for seed in range(40) for p in plan_level(WORDS, "L4", seed)]
    pauses = [p for p in placements if p.flaw.code in ("PAU-DROP", "PAU-LONG")]
    mid = [p for p in placements if p.flaw.code == "PAU-MID"]

    assert pauses and mid
    assert all(gap(p) >= CONFIG.pause_min_s for p in pauses)
    assert all(gap(p) <= CONFIG.no_pause_max_s for p in mid)


def test_filler_is_planned_only_when_filler_audio_exists() -> None:
    plans = [plan_level(WORDS, "L4", seed, has_filler=True) for seed in range(60)]

    assert "FLU-FILL" in {p.flaw.code for plan in plans for p in plan}


def test_a_passage_too_short_for_the_level_is_refused() -> None:
    with pytest.raises(PlanError, match="could place only"):
        plan_level(WORDS[:6], "L5", 0)


def test_sweep_holds_the_place_and_steps_the_magnitude() -> None:
    sweep = plan_sweep(WORDS, "ENE-LOW", seed=2)

    assert [p.flaw.magnitude for p in sweep] == list(CONFIG.sweep_magnitudes)
    assert len({(p.first_word, p.last_word) for p in sweep}) == 1
    with pytest.raises(PlanError, match="no eligible span"):
        plan_sweep(WORDS[:1], "PAU-LONG", seed=2)


def _plan(level: str = "L3") -> list[Placement]:
    return plan_level(WORDS, level, 3)


def _too_few(plan: list[Placement]) -> list[Placement]:
    return plan[:1]


def _too_strong(plan: list[Placement]) -> list[Placement]:
    return [replace(plan[0], flaw=replace(plan[0].flaw, magnitude=0.99)), *plan[1:]]


def _unknown_code(plan: list[Placement]) -> list[Placement]:
    return [replace(plan[0], flaw=replace(plan[0].flaw, code="XX-NOPE")), *plan[1:]]


def _overlapping(plan: list[Placement]) -> list[Placement]:
    clash = replace(plan[-1], first_word=plan[0].first_word, last_word=plan[0].last_word)
    return [*plan[:-1], clash]


def _extra_global(plan: list[Placement]) -> list[Placement]:
    return [*plan, replace(plan[0], is_global=True)]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_too_few, "regions, allowed"),
        (_too_strong, "outside band"),
        (_unknown_code, "unknown code"),
        (_overlapping, "overlap or touch"),
        (_extra_global, "global flaws"),
    ],
)
def test_invalid_plans_are_refused(
    mutate: Callable[[list[Placement]], list[Placement]], message: str
) -> None:
    with pytest.raises(PlanError, match=message):
        validate_plan(mutate(_plan()), "L3")


@settings(max_examples=20, deadline=None)
@given(level=st.sampled_from(LEVELS), seed=st.integers(0, 10_000))
def test_every_sampled_plan_generates_a_clip_that_passes_the_gates(level: str, seed: int) -> None:
    plan = plan_level(WORDS, level, seed)

    result = apply_plan(AUDIO.samples, [p.flaw for p in plan], seed=seed)

    assert check_clip(result, CONFIG.gates) == []
    assert len(result.labels) == len(plan)


def test_gates_name_each_failure() -> None:
    clean = apply_plan(AUDIO.samples, [], seed=0)
    clipped = replace(clean, samples=np.ones_like(clean.samples))
    silent = clean.samples.copy()
    silent[8000:9000] = 0.0
    short = replace(clean, samples=clean.samples[:-4000])
    bad_label = apply_plan(AUDIO.samples, [_plan()[0].flaw], seed=0)
    bad_label = replace(bad_label, labels=(replace(bad_label.labels[0], end=1e6),))

    assert check_clip(clipped, CONFIG.gates) == ["clipping"]
    assert check_clip(replace(clean, samples=silent), CONFIG.gates) == ["digital_silence"]
    assert check_clip(short, CONFIG.gates) == ["duration"]
    assert check_clip(bad_label, CONFIG.gates) == ["label_bounds"]


def test_silence_already_in_the_source_is_not_held_against_the_clip() -> None:
    samples = AUDIO.samples.copy()
    samples[16_000:17_600] = 0.0
    result = apply_plan(samples, [], seed=0)
    silent = result.samples.copy()
    silent[16_000:17_600] = 0.0
    clip = replace(result, samples=silent)

    assert check_clip(clip, CONFIG.gates) == ["digital_silence"]
    assert check_clip(clip, CONFIG.gates, source_silence_s=0.1) == []
