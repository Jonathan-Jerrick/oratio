"""Word-level features built from frame tracks and word timings."""

import re

import numpy as np

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


def _balance_db(high_share: float) -> float:
    """High-band energy relative to low-band energy, in dB."""
    share = min(max(high_share, _SHARE_FLOOR), 1.0 - _SHARE_FLOOR)
    return float(10.0 * np.log10(share / (1.0 - share)))


def _zscore(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / max(float(array.std()), _STD_FLOOR)


def word_features(words: list[WordSpan], tracks: Tracks) -> list[WordFeatures]:
    """Speaker-normalised measurements per word, in transcript order."""
    pitch = semitones(tracks.f0_hz)
    energy = relative_energy(tracks.energy_db)
    speech = speech_mask(tracks.energy_db)
    limit = len(tracks)
    features: list[WordFeatures] = []
    for index, word in enumerate(words):
        own = _span(word, limit)
        first = words[max(0, index - CONTEXT_WORDS)]
        last = words[min(len(words) - 1, index + CONTEXT_WORDS)]
        context = slice(_span(first, limit).start, _span(last, limit).stop)
        own_pitch = pitch[own][~np.isnan(pitch[own])]
        context_pitch = pitch[context][~np.isnan(pitch[context])]
        following = words[index + 1].start if index + 1 < len(words) else word.end
        features.append(
            {
                "log_duration": float(
                    np.log(max(word.end - word.start, HOP_SECONDS) / syllables(word.word))
                ),
                "pause_after": max(0.0, following - word.end),
                "f0_mean_st": float(own_pitch.mean()) if len(own_pitch) else 0.0,
                "f0_peak_st": float(own_pitch.max()) if len(own_pitch) else 0.0,
                "f0_range_st": _spread(context_pitch),
                "energy_mean_db": float(energy[own].mean()),
                "energy_range_db": _spread(energy[context][speech[context]]),
                "hf_balance_db": _balance_db(float(tracks.hf_ratio[own].mean())),
                "cpp_db": float(tracks.cpp_db[own].mean()),
            }
        )
    if features:
        cues = [
            _zscore([f[name] for f in features])
            for name in ("f0_peak_st", "energy_mean_db", "log_duration")
        ]
        for feature, prominence in zip(features, np.mean(cues, axis=0), strict=True):
            feature["prominence"] = float(prominence)
    return features
