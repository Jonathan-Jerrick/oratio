import numpy as np
import pytest
from synth import Word, edit, passage, render

from speechgrade.audio import Audio
from speechgrade.compare import Pair, pair_words
from speechgrade.hashing import content_hash
from speechgrade.pipeline import analyze
from speechgrade.rubric import load_rubric
from speechgrade.schema import Report
from speechgrade.score import deduction, score_regions


def _report(
    participant: list[Word], baseline: list[Word] | None = None, gain: float = 1.0
) -> Report:
    audio, spans = render(participant, seed=1)
    reference, reference_spans = render(baseline or passage(), seed=2)
    louder = Audio((audio.samples * gain).astype(np.float32), audio.sample_rate)
    return analyze(louder, spans, reference, reference_spans)


def _codes(report: Report) -> set[str]:
    return {region.code for region in report.regions}


def test_identical_delivery_has_no_regions_and_a_full_score() -> None:
    report = _report(passage())

    assert report.regions == ()
    assert report.scores.overall == 100.0
    assert set(report.scores.dimensions) == set(load_rubric().dimensions)
    assert all(word.baseline_index == index for index, word in enumerate(report.words))


def test_report_is_deterministic_and_self_hashing() -> None:
    first, second = (
        _report(edit(passage(), range(3, 6), level=0.08)),
        _report(edit(passage(), range(3, 6), level=0.08)),
    )
    body = first.model_dump(mode="json")
    body["provenance"]["report_sha256"] = "0" * 64

    assert first == second
    assert first.provenance.report_sha256 == content_hash(body)
    assert first.provenance.rubric_sha256 == load_rubric().sha256


def test_quiet_phrase_is_grounded_as_low_energy_on_those_words() -> None:
    report = _report(edit(passage(), range(3, 6), level=0.06))
    region = next(r for r in report.regions if r.code == "ENE-LOW")

    assert region.words == "any price bear"
    assert region.start == pytest.approx(report.words[3].start)
    assert region.end == pytest.approx(report.words[5].end)
    assert region.measurement is not None
    assert region.measurement.z < -2.0
    assert region.measurement.unit == "dB"
    assert region.deduction > 0
    assert report.scores.dimensions["vocal_energy"] < 10.0
    assert "below" in region.rationale
    assert "any price bear" in region.rationale


def test_flat_pitch_is_grounded_as_monotone() -> None:
    report = _report([Word(w.text, w.seconds, 150.0, w.level, w.pause) for w in passage()])

    assert "INT-MONO" in _codes(report)
    assert report.scores.dimensions["intonation"] < 10.0
    assert report.scores.dimensions["pausing"] == 10.0


def test_rushed_words_are_grounded_as_fast_pacing() -> None:
    report = _report(edit(passage(), range(6, 9), seconds=0.14))
    region = next(r for r in report.regions if r.code == "PAC-RUSH")

    assert region.words == "any burden meet"
    assert report.scores.dimensions["pacing"] < 10.0


def test_dropped_overlong_and_misplaced_pauses_are_told_apart() -> None:
    dropped = _report(edit(passage(), range(4, 5), pause=0.02))
    stretched = _report(edit(passage(), range(4, 5), pause=1.4))
    misplaced = _report(edit(passage(), range(1, 2), pause=1.2))

    assert next(r for r in dropped.regions if r.code == "PAU-DROP").words == "price"
    assert next(r for r in stretched.regions if r.code == "PAU-LONG").words == "price"
    assert "PAU-MID" not in _codes(stretched)
    mid = next(r for r in misplaced.regions if r.code == "PAU-MID")
    assert mid.words == "shall"
    assert mid.end == pytest.approx(misplaced.words[2].start)
    assert "PAU-LONG" not in _codes(misplaced)


def test_muffled_phrase_is_grounded_as_a_clarity_flaw() -> None:
    report = _report(edit(passage(), range(3, 6), brightness=3.0))
    region = next(r for r in report.regions if r.code == "CLA-MUFF")

    assert region.words == "any price bear"
    assert report.scores.dimensions["clarity"] < 10.0


def test_a_uniformly_duller_voice_or_microphone_is_not_a_flaw() -> None:
    report = _report(edit(passage(), range(0, 12), brightness=1.6))

    assert "CLA-MUFF" not in _codes(report)


def test_rising_pitch_on_a_word_is_grounded_as_uptalk() -> None:
    words = passage()
    report = _report(edit(words, range(7, 8), f0_end=words[7].f0 * 1.5))

    assert next(r for r in report.regions if r.code == "INT-UP").words == "burden"


def test_fading_loudness_is_grounded_as_trailing_energy() -> None:
    words = passage()
    fading = [
        Word(w.text, w.seconds, w.f0, w.level * (0.25 if 2 <= i < 5 else 1.0) ** (i - 1), w.pause)
        for i, w in enumerate(words)
    ]
    report = _report(fading)

    assert "ENE-TRAIL" in _codes(report)


def test_stress_on_the_wrong_word_is_grounded_as_misplaced_emphasis() -> None:
    words = passage()
    report = _report(edit(words, range(5, 6), f0=230.0, level=0.5, seconds=0.6))

    assert "bear" in next(r for r in report.regions if r.code == "EMP-MISP").words


def test_a_stretched_word_is_a_prolongation_not_slow_pacing() -> None:
    report = _report(edit(passage(), range(6, 7), seconds=1.4))
    region = next(r for r in report.regions if r.code == "FLU-PROL")

    assert region.words == "any"
    assert not [r for r in report.regions if r.code == "PAC-DRAG" and r.words == "any"]
    assert report.scores.dimensions["fluency"] < 10.0


def test_uneven_syllable_timing_is_grounded_as_erratic_pacing() -> None:
    lengths = {2: 0.12, 3: 0.75, 4: 0.12, 5: 0.75, 6: 0.12}
    words = [
        Word(w.text, lengths.get(i, w.seconds), w.f0, w.level, w.pause)
        for i, w in enumerate(passage())
    ]

    assert "PAC-ERR" in _codes(_report(words))


def test_a_swallowed_syllable_is_grounded_as_a_clarity_flaw() -> None:
    baseline = edit(passage(), range(7, 8), nuclei=2, seconds=0.5)
    report = _report(edit(passage(), range(7, 8), seconds=0.5), baseline=baseline)

    assert next(r for r in report.regions if r.code == "CLA-SWAL").words == "burden"


def test_creaky_voice_is_grounded_as_vocal_fry() -> None:
    report = _report(edit(passage(), range(11, 12), f0=72.0))

    assert next(r for r in report.regions if r.code == "VQ-FRY").words == "today"


def test_fillers_repetitions_and_omissions_are_fluency_events() -> None:
    words = passage()
    spoken = [*words[:2], Word("um"), *words[2:5], words[4], *words[5:8], *words[10:]]
    report = _report(spoken)
    by_code = {region.code: region for region in report.regions if region.dimension == "fluency"}

    assert by_code["FLU-FILL"].words == "um"
    assert by_code["FLU-REP"].words == "price"
    assert by_code["FLU-OMIT"].words == "meet any"
    assert by_code["FLU-OMIT"].measurement is None
    assert "meet any" in by_code["FLU-OMIT"].fix
    assert report.scores.dimensions["fluency"] < 10.0


def test_score_falls_as_the_flaw_grows() -> None:
    scores = [
        _report(edit(passage(), range(3, 6), level=level)).scores.dimensions["vocal_energy"]
        for level in (0.30, 0.10, 0.05, 0.02)
    ]

    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


@pytest.mark.parametrize("gain", [0.4, 2.5])
def test_scores_do_not_depend_on_recording_level(gain: float) -> None:
    flawed = edit(passage(), range(3, 6), level=0.06)

    assert _report(flawed, gain=gain).scores == _report(flawed).scores


def test_scores_do_not_depend_on_speaker_pitch() -> None:
    higher = [Word(w.text, w.seconds, w.f0 * 1.5, w.level, w.pause) for w in passage()]

    assert _report(higher).regions == ()


def test_regions_are_numbered_in_time_order() -> None:
    report = _report(edit(edit(passage(), range(0, 2), level=0.05), range(8, 10), seconds=0.12))

    assert [r.region_id for r in report.regions] == [
        f"R{i}" for i in range(1, len(report.regions) + 1)
    ]
    assert [r.start for r in report.regions] == sorted(r.start for r in report.regions)


def test_pairing_marks_insertions_and_omissions() -> None:
    assert pair_words(["A", "UM", "B"], ["A", "B", "C"]) == [
        Pair(0, 0),
        Pair(1, None),
        Pair(2, 1),
        Pair(None, 2),
    ]


def test_deduction_is_capped_and_weighted_by_duration() -> None:
    scoring = load_rubric().scoring

    assert deduction(1.5, 2.0, 1.0, 2.0, scoring) == 0.0
    assert deduction(-3.0, 2.0, 1.0, 2.0, scoring) == pytest.approx(1.0)
    assert deduction(99.0, 2.0, 1.0, 5.0, scoring) == pytest.approx(scoring.z_cap)
    assert deduction(3.0, 2.0, 2.0, 0.5, scoring) == pytest.approx(1.0)
    assert deduction(3.0, 2.0, 1.0, 0.0, scoring) == pytest.approx(scoring.duration_weight_floor)


def test_dimension_scores_never_go_below_zero() -> None:
    report = _report(edit(passage(), range(0, 12), level=0.01, seconds=0.1))
    scores = score_regions(list(report.regions) * 20, load_rubric())

    assert min(scores.dimensions.values()) == 0.0
    assert 0.0 <= scores.overall <= 100.0
