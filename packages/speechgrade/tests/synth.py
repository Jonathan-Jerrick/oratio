"""Synthetic speech for tests: voiced words with exact timings, pitch and level."""

from dataclasses import dataclass, replace

import numpy as np

from speechgrade.align import WordSpan
from speechgrade.audio import SAMPLE_RATE, Audio

FADE_SECONDS = 0.02
LEAD_SECONDS = 0.3
NOISE_LEVEL = 1e-4


@dataclass(frozen=True)
class Word:
    text: str
    seconds: float = 0.35
    f0: float = 150.0
    level: float = 0.3
    pause: float = 0.08
    brightness: float = 1.0


def _voiced(word: Word) -> np.ndarray:
    n = int(word.seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    harmonics = sum(np.sin(2 * np.pi * k * word.f0 * t) / k**word.brightness for k in range(1, 20))
    fade = int(FADE_SECONDS * SAMPLE_RATE)
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    return word.level * envelope * harmonics / np.max(np.abs(harmonics))


def render(words: list[Word], seed: int = 0) -> tuple[Audio, list[WordSpan]]:
    rng = np.random.default_rng(seed)
    pieces = [np.zeros(int(LEAD_SECONDS * SAMPLE_RATE))]
    spans: list[WordSpan] = []
    cursor = LEAD_SECONDS
    for word in words:
        pieces.append(_voiced(word))
        spans.append(WordSpan(word.text.upper(), cursor, cursor + word.seconds, -0.1))
        pieces.append(np.zeros(int(word.pause * SAMPLE_RATE)))
        cursor += word.seconds + word.pause
    pieces.append(np.zeros(int(LEAD_SECONDS * SAMPLE_RATE)))
    signal = np.concatenate(pieces)
    signal = signal + NOISE_LEVEL * rng.standard_normal(len(signal))
    return Audio(signal.astype(np.float32), SAMPLE_RATE), spans


def passage() -> list[Word]:
    """Twelve words with a moving pitch and level contour, like an expressive reading."""
    texts = [
        "we",
        "shall",
        "pay",
        "any",
        "price",
        "bear",
        "any",
        "burden",
        "meet",
        "any",
        "hardship",
        "today",
    ]
    pitches = [130, 150, 175, 160, 190, 140, 165, 185, 135, 170, 195, 120]
    levels = [0.25, 0.30, 0.38, 0.28, 0.40, 0.26, 0.33, 0.39, 0.25, 0.31, 0.40, 0.24]
    pauses = [0.06, 0.06, 0.06, 0.06, 0.40, 0.06, 0.06, 0.40, 0.06, 0.06, 0.40, 0.06]
    return [
        Word(text, f0=f0, level=level, pause=pause)
        for text, f0, level, pause in zip(texts, pitches, levels, pauses, strict=True)
    ]


def edit(words: list[Word], indices: range, **changes: float) -> list[Word]:
    return [replace(word, **changes) if i in indices else word for i, word in enumerate(words)]
