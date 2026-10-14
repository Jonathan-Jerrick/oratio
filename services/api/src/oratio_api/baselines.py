"""Reference passages shipped in the image: audio plus aligned words, matched by transcript."""

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from speechgrade.align import WordSpan, normalise_words
from speechgrade.audio import Audio

MIN_MATCH = 0.6


class BaselineError(LookupError):
    """No reference passage fits this request."""


@dataclass(frozen=True, slots=True)
class Baseline:
    baseline_id: str
    title: str
    audio: Audio
    words: tuple[WordSpan, ...]

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


@lru_cache(maxsize=1)
def load_baselines(directory: Path) -> dict[str, Baseline]:
    found: dict[str, Baseline] = {}
    for meta in sorted(directory.glob("*.json")):
        record = json.loads(meta.read_text())
        samples, rate = sf.read(directory / f"{meta.stem}.flac", dtype="float32")
        words = tuple(WordSpan(w["text"], w["start"], w["end"], 0.0) for w in record["words"])
        found[meta.stem] = Baseline(
            meta.stem, record.get("title", meta.stem), Audio(samples, rate), words
        )
    return found


def choose(
    baselines: dict[str, Baseline], transcript: str, baseline_id: str | None
) -> tuple[Baseline, float]:
    """The requested baseline, or the one whose text best matches the transcript."""
    if baseline_id:
        if baseline_id not in baselines:
            raise BaselineError(f"unknown baseline {baseline_id!r}")
        chosen = baselines[baseline_id]
        return chosen, _similarity(chosen, transcript)
    if not baselines:
        raise BaselineError("no baselines are installed")
    scored = sorted(
        ((_similarity(b, transcript), b.baseline_id) for b in baselines.values()), reverse=True
    )
    score, best = scored[0]
    if score < MIN_MATCH:
        raise BaselineError("the transcript does not match any reference passage closely enough")
    return baselines[best], score


def _similarity(baseline: Baseline, transcript: str) -> float:
    return SequenceMatcher(
        a=[w.word for w in baseline.words], b=normalise_words(transcript), autojunk=False
    ).ratio()
