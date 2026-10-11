import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from synth import Word, render

from speechgrade.features import (
    HOP_SECONDS,
    N_MFCC,
    extract_tracks,
    relative_energy,
    semitones,
    speech_mask,
)
from speechgrade.words import syllables, word_features


def _frames(span_start: float, span_end: float) -> slice:
    return slice(round(span_start / HOP_SECONDS) + 5, round(span_end / HOP_SECONDS) - 5)


def test_pitch_tracks_the_fundamental_and_marks_silence_unvoiced() -> None:
    audio, spans = render([Word("a", seconds=0.6, f0=150.0), Word("b", seconds=0.6, f0=220.0)])
    tracks = extract_tracks(audio.samples)

    assert np.median(tracks.f0_hz[_frames(spans[0].start, spans[0].end)]) == pytest.approx(
        150, abs=2
    )
    assert np.median(tracks.f0_hz[_frames(spans[1].start, spans[1].end)]) == pytest.approx(
        220, abs=2
    )
    assert not tracks.f0_hz[:20].any()


def test_tracks_share_one_frame_grid() -> None:
    audio, _ = render([Word("a", seconds=0.5)])
    tracks = extract_tracks(audio.samples)

    assert len(tracks) == 1 + len(audio.samples) // 160
    assert len(tracks.f0_hz) == len(tracks.hf_ratio) == len(tracks.cpp_db) == len(tracks)
    assert tracks.mfcc.shape == (len(tracks), N_MFCC)
    assert np.isfinite(tracks.mfcc).all()


def test_halving_amplitude_lowers_energy_by_six_db() -> None:
    audio, spans = render([Word("a", seconds=0.6, level=0.4), Word("b", seconds=0.6, level=0.2)])
    energy = extract_tracks(audio.samples).energy_db
    loud = np.median(energy[_frames(spans[0].start, spans[0].end)])
    quiet = np.median(energy[_frames(spans[1].start, spans[1].end)])

    assert loud - quiet == pytest.approx(6.02, abs=0.2)


def test_duller_voice_has_lower_high_frequency_share() -> None:
    audio, spans = render(
        [Word("a", seconds=0.6, brightness=0.5), Word("b", seconds=0.6, brightness=2.5)]
    )
    ratio = extract_tracks(audio.samples).hf_ratio

    bright = np.median(ratio[_frames(spans[0].start, spans[0].end)])
    dull = np.median(ratio[_frames(spans[1].start, spans[1].end)])
    assert bright > 5 * dull


def test_voiced_frames_have_a_higher_cepstral_peak_than_noise() -> None:
    audio, spans = render([Word("a", seconds=0.8)])
    cpp = extract_tracks(audio.samples).cpp_db

    assert np.median(cpp[_frames(spans[0].start, spans[0].end)]) > np.median(cpp[:20]) + 5


def test_speech_mask_separates_words_from_silence() -> None:
    audio, spans = render([Word("a", seconds=0.6)])
    mask = speech_mask(extract_tracks(audio.samples).energy_db)

    assert mask[_frames(spans[0].start, spans[0].end)].all()
    assert not mask[:20].any()


@settings(max_examples=12, deadline=None)
@given(gain=st.floats(0.25, 3.0))
def test_relative_energy_is_invariant_to_gain(gain: float) -> None:
    audio, _ = render([Word("a", level=0.2), Word("b", level=0.3), Word("c", level=0.25)])
    reference = extract_tracks(audio.samples)
    scaled = extract_tracks((audio.samples * gain).astype(np.float32))
    mask = speech_mask(reference.energy_db)

    difference = (
        relative_energy(scaled.energy_db)[mask] - relative_energy(reference.energy_db)[mask]
    )
    assert np.max(np.abs(difference)) < 0.05


@settings(max_examples=12, deadline=None)
@given(shift=st.floats(-4.0, 4.0))
def test_semitone_contour_is_invariant_to_speaker_pitch(shift: float) -> None:
    factor = 2.0 ** (shift / 12.0)
    base = [Word("a", f0=140.0), Word("b", f0=180.0), Word("c", f0=160.0)]
    moved = [Word(w.text, f0=w.f0 * factor) for w in base]
    audio, spans = render(base)
    shifted_audio, _ = render(moved)

    ours = word_features(spans, extract_tracks(audio.samples))
    theirs = word_features(spans, extract_tracks(shifted_audio.samples))
    for a, b in zip(ours, theirs, strict=True):
        assert a["f0_mean_st"] == pytest.approx(b["f0_mean_st"], abs=0.3)
        assert a["f0_range_st"] == pytest.approx(b["f0_range_st"], abs=0.3)


def test_semitones_are_centred_on_the_speaker_median() -> None:
    contour = semitones(np.array([0.0, 100.0, 200.0, 400.0], dtype=np.float32))

    assert np.isnan(contour[0])
    assert contour[1:].tolist() == pytest.approx([-12.0, 0.0, 12.0])
    assert np.isnan(semitones(np.zeros(4, dtype=np.float32))).all()


def test_word_features_measure_duration_pause_and_prominence() -> None:
    words = [
        Word("alpha", seconds=0.30, pause=0.05),
        Word("be", seconds=0.60, f0=200.0, level=0.45, pause=0.40),
        Word("cat", seconds=0.30, pause=0.05),
    ]
    audio, spans = render(words)
    features = word_features(spans, extract_tracks(audio.samples))

    assert features[0]["pause_after"] == pytest.approx(0.05, abs=1e-6)
    assert features[1]["pause_after"] == pytest.approx(0.40, abs=1e-6)
    assert features[2]["pause_after"] == 0.0
    assert features[0]["log_duration"] == pytest.approx(np.log(0.30 / 2))
    assert features[1]["log_duration"] == pytest.approx(np.log(0.60))
    assert max(range(3), key=lambda i: features[i]["prominence"]) == 1
    assert word_features([], extract_tracks(audio.samples)) == []


@pytest.mark.parametrize(("word", "count"), [("price", 2), ("burden", 2), ("we", 1), ("hmm", 1)])
def test_syllable_estimate_counts_vowel_groups(word: str, count: int) -> None:
    assert syllables(word) == count
