import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from synth import passage, render

from oratio_data.dsp import SAMPLE_RATE, DspError, room_tone
from oratio_data.injectors import DIMENSIONS, Flaw, PlanError, Result, apply_plan
from speechgrade.align import WordSpan
from speechgrade.audio import Audio
from speechgrade.features import HOP_SECONDS, extract_tracks
from speechgrade.pipeline import analyze

AUDIO, WORDS = render(passage(), seed=3)
CODES = [
    "PAC-RUSH", "PAC-DRAG", "PAC-ERR", "PAU-DROP", "PAU-LONG", "PAU-MID", "INT-MONO", "INT-EXAG",
    "INT-UP", "ENE-LOW", "ENE-SPIKE", "ENE-TRAIL", "ENE-FLAT", "EMP-DROP", "EMP-MISP", "FLU-REP",
    "FLU-FALSE", "FLU-PROL", "FLU-OMIT", "FLU-FILL", "CLA-MUFF", "CLA-SWAL", "VQ-FRY",
]  # fmt: skip
MAGNITUDE_FREE = {"FLU-REP", "FLU-FALSE", "FLU-OMIT", "FLU-FILL"}
FILLER = AUDIO.samples[round(WORDS[0].start * SAMPLE_RATE) : round(WORDS[0].end * SAMPLE_RATE)]
CONTROL = apply_plan(AUDIO.samples, [], seed=0)


def _flaw(code: str, first: int, last: int, magnitude: float) -> Flaw:
    start, end = WORDS[first].start, WORDS[last].end
    if code == "PAU-DROP":
        start, end = WORDS[last].end, WORDS[last + 1].start
    spans = tuple((w.start, w.end) for w in WORDS[first : last + 1])
    return Flaw(code, start, end, magnitude, spans)


def _run(code: str, first: int, last: int, magnitude: float) -> Result:
    return apply_plan(AUDIO.samples, [_flaw(code, first, last, magnitude)], seed=0, filler=FILLER)


def _frames(start: float, end: float) -> slice:
    return slice(round(start / HOP_SECONDS) + 3, round(end / HOP_SECONDS) - 3)


def _seconds(result: Result) -> float:
    return len(result.samples) / SAMPLE_RATE


def test_empty_plan_is_a_faithful_control() -> None:
    assert CONTROL.labels == ()
    assert CONTROL.warp.output_duration == pytest.approx(AUDIO.seconds)
    assert _seconds(CONTROL) == pytest.approx(AUDIO.seconds, abs=0.002)
    ours, theirs = extract_tracks(CONTROL.samples), extract_tracks(AUDIO.samples)
    n = min(len(ours), len(theirs))
    voiced = (ours.f0_hz[:n] > 0) & (theirs.f0_hz[:n] > 0)
    assert np.median(np.abs(ours.energy_db[:n][voiced] - theirs.energy_db[:n][voiced])) < 0.5
    assert np.median(np.abs(ours.f0_hz[:n][voiced] - theirs.f0_hz[:n][voiced])) < 1.0
    assert voiced.sum() > 0.9 * (theirs.f0_hz > 0).sum()


def test_every_taxonomy_prefix_has_a_dimension() -> None:
    assert {code.split("-")[0] for code in CODES} == set(DIMENSIONS)


@pytest.mark.parametrize("code", CODES)
def test_audio_length_matches_the_warp(code: str) -> None:
    result = _run(code, 4, 7, 0.6)
    (label,) = result.labels

    assert _seconds(result) == pytest.approx(result.warp.output_duration, abs=0.01)
    assert 0.0 <= label.start <= label.end <= _seconds(result) + 0.01
    assert label.code == code
    assert np.isfinite(result.samples).all()
    assert float(np.max(np.abs(result.samples))) <= 1.0


@pytest.mark.parametrize("code", [c for c in CODES if c not in MAGNITUDE_FREE])
def test_zero_magnitude_matches_the_control(code: str) -> None:
    result = _run(code, 4, 7, 0.0)

    assert _seconds(result) == pytest.approx(_seconds(CONTROL), abs=0.01)
    ours, theirs = extract_tracks(result.samples), extract_tracks(CONTROL.samples)
    n = min(len(ours), len(theirs))
    voiced = (ours.f0_hz[:n] > 0) & (theirs.f0_hz[:n] > 0)
    assert np.median(np.abs(ours.energy_db[:n][voiced] - theirs.energy_db[:n][voiced])) < 0.5
    assert np.median(np.abs(ours.f0_hz[:n][voiced] - theirs.f0_hz[:n][voiced])) < 1.0


def test_plans_are_deterministic_for_a_seed() -> None:
    first, second = _run("PAU-LONG", 4, 4, 0.8), _run("PAU-LONG", 4, 4, 0.8)

    assert np.array_equal(first.samples, second.samples)
    assert first.labels == second.labels


def test_low_energy_drops_the_level_inside_the_label_only() -> None:
    result = _run("ENE-LOW", 3, 5, 1.0)
    (label,) = result.labels
    flawed = extract_tracks(result.samples).energy_db
    control = extract_tracks(CONTROL.samples).energy_db
    inside = _frames(WORDS[4].start, WORDS[4].end)
    outside = _frames(WORDS[9].start, WORDS[9].end)

    assert (label.start, label.end) == pytest.approx((WORDS[3].start, WORDS[5].end))
    assert np.median(flawed[inside] - control[inside]) == pytest.approx(-15.0, abs=0.5)
    assert np.median(np.abs(flawed[outside] - control[outside])) < 0.2


def test_monotone_flattens_pitch_and_exaggeration_widens_it() -> None:
    def spread(result: Result) -> float:
        f0 = extract_tracks(result.samples).f0_hz[_frames(WORDS[1].start, WORDS[5].end)]
        return float(np.std(12 * np.log2(f0[f0 > 0])))

    flat, wide = spread(_run("INT-MONO", 1, 5, 1.0)), spread(_run("INT-EXAG", 1, 5, 1.0))

    assert flat < 0.25 * spread(CONTROL)
    assert wide > 1.8 * spread(CONTROL)


def test_rushing_halves_the_span_and_shifts_later_words() -> None:
    result = _run("PAC-RUSH", 2, 5, 1.0)
    (label,) = result.labels
    original = WORDS[5].end - WORDS[2].start

    assert label.end - label.start == pytest.approx(original / 2)
    assert result.warp.apply(WORDS[8].start) == pytest.approx(WORDS[8].start - original / 2)
    assert _seconds(result) == pytest.approx(AUDIO.seconds - original / 2, abs=0.01)


def test_added_pause_is_room_tone_not_digital_silence() -> None:
    result = _run("PAU-LONG", 4, 4, 1.0)
    (label,) = result.labels
    added = result.samples[
        round((WORDS[4].end + 0.05) * SAMPLE_RATE) : round((label.end - 0.05) * SAMPLE_RATE)
    ]

    assert label.end - label.start == pytest.approx(WORDS[4].end - WORDS[4].start + 1.2, abs=0.03)
    assert len(added) > SAMPLE_RATE
    assert np.count_nonzero(added) > 0.99 * len(added)
    assert np.sqrt(np.mean(added**2)) < 0.01


def test_dropped_pause_shortens_the_gap() -> None:
    result = _run("PAU-DROP", 4, 4, 1.0)
    gap = WORDS[5].start - WORDS[4].end

    assert AUDIO.seconds - _seconds(result) == pytest.approx(0.9 * gap, abs=0.01)
    assert result.warp.apply(WORDS[5].start) == pytest.approx(WORDS[5].start - 0.9 * gap, abs=0.001)


def test_omission_removes_the_word_and_collapses_its_label() -> None:
    result = _run("FLU-OMIT", 6, 6, 1.0)
    (label,) = result.labels

    assert label.end == pytest.approx(label.start)
    assert AUDIO.seconds - _seconds(result) == pytest.approx(
        WORDS[6].end - WORDS[6].start, abs=0.01
    )


@pytest.mark.parametrize(
    ("code", "extra"), [("FLU-REP", 0.35 + 0.08), ("FLU-FALSE", 0.35 * 0.45 + 0.15)]
)
def test_repetitions_insert_a_copy_inside_the_label(code: str, extra: float) -> None:
    result = _run(code, 6, 6, 1.0)
    (label,) = result.labels

    assert _seconds(result) - AUDIO.seconds == pytest.approx(extra, abs=0.03)
    assert label.end - label.start == pytest.approx(0.35 + extra, abs=0.04)
    assert result.warp.apply(WORDS[7].start) == pytest.approx(WORDS[7].start + extra, abs=0.03)


def test_muffling_removes_high_frequencies() -> None:
    result = _run("CLA-MUFF", 3, 5, 1.0)
    inside = _frames(WORDS[4].start, WORDS[4].end)

    flawed = extract_tracks(result.samples).hf_ratio[inside]
    assert np.median(flawed) < 0.3 * np.median(extract_tracks(CONTROL.samples).hf_ratio[inside])


@pytest.mark.parametrize(
    ("flaw", "message"),
    [
        (Flaw("ENE-LOW", 1.0, 2.0, 1.5), "outside 0 to 1"),
        (Flaw("ENE-LOW", 2.0, 1.0, 0.5), "outside the clip"),
        (Flaw("ENE-LOW", 1.0, 99.0, 0.5), "outside the clip"),
        (Flaw("XYZ-NOPE", 1.0, 2.0, 0.5), "no injector"),
        (Flaw("PAC-ERR", 1.0, 2.0, 0.5), "word spans"),
        (Flaw("FLU-FILL", 1.0, 2.0, 0.5), "filler audio"),
    ],
)
def test_invalid_flaws_are_rejected(flaw: Flaw, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        apply_plan(AUDIO.samples, [flaw], seed=0)


def test_overlapping_duration_edits_are_rejected() -> None:
    plan = [_flaw("PAC-RUSH", 2, 5, 0.5), _flaw("PAC-DRAG", 4, 7, 0.5)]

    with pytest.raises(PlanError, match="duration edits conflict"):
        apply_plan(AUDIO.samples, plan, seed=0)


def test_room_tone_needs_enough_audio() -> None:
    with pytest.raises(DspError, match="too short"):
        room_tone(np.zeros(100, dtype=np.float32), 0.5, np.random.default_rng(0))


def _warped_words(result: Result, omitted: range = range(0)) -> list[WordSpan]:
    return [
        WordSpan(w.word, *result.warp.span(w.start, w.end), w.score)
        for i, w in enumerate(WORDS)
        if i not in omitted
    ]


@pytest.mark.parametrize(
    ("code", "first", "last", "expected"),
    [
        ("ENE-LOW", 3, 5, "ENE-LOW"),
        ("ENE-SPIKE", 3, 5, "ENE-SPIKE"),
        ("INT-MONO", 0, 11, "INT-MONO"),
        ("PAC-RUSH", 6, 8, "PAC-RUSH"),
        ("PAC-DRAG", 6, 8, "PAC-DRAG"),
        ("PAU-LONG", 4, 4, "PAU-LONG"),
        ("PAU-MID", 1, 1, "PAU-MID"),
        ("PAU-DROP", 4, 4, "PAU-DROP"),
        ("CLA-MUFF", 3, 5, "CLA-MUFF"),
        ("FLU-PROL", 6, 6, "FLU-PROL"),
        ("VQ-FRY", 9, 9, "VQ-FRY"),
    ],
)
def test_injected_flaw_is_found_by_the_evaluator_inside_its_label(
    code: str, first: int, last: int, expected: str
) -> None:
    result = _run(code, first, last, 1.0)
    (label,) = result.labels
    report = analyze(
        Audio(result.samples, SAMPLE_RATE),
        _warped_words(result),
        Audio(CONTROL.samples, SAMPLE_RATE),
        WORDS,
    )

    hits = [r for r in report.regions if r.code == expected]
    assert hits, f"{expected} not detected; got {[r.code for r in report.regions]}"
    assert any(r.start < label.end + 0.3 and r.end > label.start - 0.3 for r in hits)


def test_control_scores_full_marks_against_the_original() -> None:
    report = analyze(Audio(CONTROL.samples, SAMPLE_RATE), WORDS, AUDIO, WORDS)

    assert report.regions == ()


PROPERTY_CODES = [
    "PAC-RUSH", "PAC-DRAG", "ENE-LOW", "INT-MONO", "PAU-LONG", "FLU-OMIT", "FLU-REP", "CLA-MUFF",
]  # fmt: skip
_word_flaw = st.tuples(st.sampled_from(PROPERTY_CODES), st.integers(0, 3), st.floats(0.1, 1.0))


@settings(max_examples=25, deadline=None)
@given(picks=st.lists(_word_flaw, min_size=1, max_size=4, unique_by=lambda pick: pick[1]))
def test_multi_flaw_plans_keep_audio_warp_and_labels_consistent(
    picks: list[tuple[str, int, float]],
) -> None:
    plan = [_flaw(code, 3 * slot, 3 * slot + 1, magnitude) for code, slot, magnitude in picks]

    result = apply_plan(AUDIO.samples, plan, seed=1)

    assert _seconds(result) == pytest.approx(result.warp.output_duration, abs=0.01 * len(plan))
    assert len(result.labels) == len(plan)
    mapped = result.warp.apply([w.start for w in WORDS])
    assert np.all(np.diff(mapped) >= -1e-9)
    for label in result.labels:
        assert 0.0 <= label.start <= label.end <= result.warp.output_duration + 1e-6
