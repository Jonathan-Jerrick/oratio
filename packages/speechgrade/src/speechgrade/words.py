"""Word-level features built from frame tracks and word timings."""

import re

import numpy as np
from scipy.signal import find_peaks

from speechgrade.align import WordSpan
from speechgrade.features import (
    HOP_SECONDS,
    FloatArray,
    Tracks,
    relative_energy,
    semitones,
    speech_mask,
)

CONTEXT_WORDS = 2
MIN_VOICED_FRAMES = 3
MIN_SLOPE_FRAMES = 6
FRY_SEMITONES_BELOW_MEDIAN = 7.0
NUCLEUS_PROMINENCE_DB = 3.0
_VOWEL_GROUP = re.compile(r"[AEIOUY]+")
_STD_FLOOR = 1e-6
_SHARE_FLOOR = 1e-6

WordFeatures = dict[str, float]


def syllables(word: str) -> int:
    return max(1, len(_VOWEL_GROUP.findall(word.upper())))


def _span(word: WordSpan, limit: int) -> slice:
    start = min(limit - 1, max(0, round(word.start / HOP_SECONDS)))
    return slice(start, min(limit, max(start + 1, round(word.end / HOP_SECONDS))))


def _spread(values: FloatArray) -> float:
    if len(values) < MIN_VOICED_FRAMES:
        return 0.0
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def _drift(values: FloatArray, parts: int) -> float:
    """Mean of the last 1/parts of the values minus the mean of the first 1/parts."""
    if len(values) < MIN_SLOPE_FRAMES:
        return 0.0
    edge = max(1, len(values) // parts)
    return float(values[-edge:].mean() - values[:edge].mean())


def _balance_db(high_share: FloatArray) -> FloatArray:
    """High-band energy relative to low-band energy, in dB."""
    share = np.clip(high_share.astype(np.float64), _SHARE_FLOOR, 1.0 - _SHARE_FLOOR)
    return np.asarray(10.0 * np.log10(share / (1.0 - share)), dtype=np.float32)


def _zscore(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / max(float(array.std()), _STD_FLOOR)


def _nuclei(energy: FloatArray) -> int:
    """Syllable nuclei: loudness peaks that stand clear of their surroundings."""
    padded = np.concatenate(([energy.min()], energy, [energy.min()]))
    peaks, _ = find_peaks(padded, prominence=NUCLEUS_PROMINENCE_DB)
    return max(1, len(peaks))


def _add_context_features(features: list[WordFeatures]) -> None:
    durations = np.array([f["log_duration"] for f in features])
    cues = [
        _zscore([f[name] for f in features])
        for name in ("f0_peak_st", "energy_mean_db", "log_duration")
    ]
    prominence = np.mean(cues, axis=0)
    for index, feature in enumerate(features):
        window = slice(max(0, index - CONTEXT_WORDS), index + CONTEXT_WORDS + 1)
        neighbours = np.delete(durations[window], index - window.start)
        feature["prominence"] = float(prominence[index])
        feature["rate_variability"] = float(durations[window].std())
        feature["relative_length"] = (
            float(durations[index] - neighbours.mean()) if len(neighbours) else 0.0
        )


def word_features(words: list[WordSpan], tracks: Tracks) -> list[WordFeatures]:
    """Speaker-normalised measurements per word, in transcript order."""
    pitch = semitones(tracks.f0_hz)
    energy = relative_energy(tracks.energy_db)
    speech = speech_mask(tracks.energy_db)
    balance = _balance_db(tracks.hf_ratio)
    balance = balance - (np.median(balance[speech]) if speech.any() else 0.0)
    limit = len(tracks)
    features: list[WordFeatures] = []
    for index, word in enumerate(words):
        own = _span(word, limit)
        first = words[max(0, index - CONTEXT_WORDS)]
        last = words[min(len(words) - 1, index + CONTEXT_WORDS)]
        context = slice(_span(first, limit).start, _span(last, limit).stop)
        own_pitch = pitch[own][~np.isnan(pitch[own])]
        context_pitch = pitch[context][~np.isnan(pitch[context])]
        context_energy = energy[context][speech[context]]
        following = words[index + 1].start if index + 1 < len(words) else word.end
        count = syllables(word.word)
        features.append(
            {
                "log_duration": float(np.log(max(word.end - word.start, HOP_SECONDS) / count)),
                "pause_after": max(0.0, following - word.end),
                "f0_mean_st": float(own_pitch.mean()) if len(own_pitch) else 0.0,
                "f0_peak_st": float(own_pitch.max()) if len(own_pitch) else 0.0,
                "f0_range_st": _spread(context_pitch),
                "f0_slope_st": _drift(own_pitch, 3),
                "fry_fraction": (
                    float(np.mean(own_pitch < -FRY_SEMITONES_BELOW_MEDIAN))
                    if len(own_pitch)
                    else 0.0
                ),
                "energy_mean_db": float(energy[own].mean()),
                "energy_range_db": _spread(context_energy),
                "energy_trend_db": _drift(context_energy, 2),
                "hf_balance_db": float(balance[own].mean()),
                "cpp_db": float(tracks.cpp_db[own].mean()),
                "nuclei_per_syllable": _nuclei(energy[own]) / count,
            }
        )
    if features:
        _add_context_features(features)
    return features
