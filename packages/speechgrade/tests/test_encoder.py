from itertools import pairwise

import numpy as np
import pytest

from speechgrade.align import AlignmentError
from speechgrade.audio import SAMPLE_RATE
from speechgrade.encoder import BLANK, CHUNK_SECONDS, VOCAB, CtcEncoder

FRAMES_PER_SECOND = 50


def _spelling_runner(text: str, total_seconds: float):  # type: ignore[no-untyped-def]
    """A fake acoustic model that 'hears' text spread evenly across the whole clip."""
    tokens = [VOCAB[c] for c in "|".join(text.upper().split())]
    frames = int(total_seconds * FRAMES_PER_SECOND)
    plan = np.full(frames, BLANK)
    slots = np.linspace(0, frames, len(tokens) + 1).astype(int)
    for token, start, end in zip(tokens, slots[:-1], slots[1:], strict=True):
        plan[start + (end - start) // 3 : start + 2 * (end - start) // 3] = token
    position = {"next": 0}

    def run(window: np.ndarray) -> np.ndarray:
        assert abs(float(window.mean())) < 1e-3
        count = len(window) // (SAMPLE_RATE // FRAMES_PER_SECOND)
        chosen = plan[position["next"] : position["next"] + count]
        position["next"] += count
        logits = np.full((len(chosen), len(VOCAB)), -5.0, dtype=np.float32)
        logits[np.arange(len(chosen)), chosen] = 5.0
        return logits

    return run


def test_log_probs_are_normalised_and_stitched_across_windows() -> None:
    seconds = CHUNK_SECONDS * 2 + 3
    encoder = CtcEncoder(_spelling_runner("we shall pay any price", seconds))
    audio = np.random.default_rng(0).standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)

    log_probs = encoder.log_probs(audio * 0.1 + 0.3)

    assert log_probs.shape == (int(seconds * FRAMES_PER_SECOND), len(VOCAB))
    assert np.allclose(np.exp(log_probs).sum(axis=1), 1.0, atol=1e-5)


def test_align_returns_ordered_words_inside_the_clip() -> None:
    seconds = 30.0
    encoder = CtcEncoder(_spelling_runner("we shall pay any price", seconds))
    audio = np.random.default_rng(1).standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)

    words = encoder.align(audio, "We shall pay any price.")

    assert [w.word for w in words] == ["WE", "SHALL", "PAY", "ANY", "PRICE"]
    assert all(0 <= a.start < a.end <= b.start for a, b in pairwise(words))
    assert words[-1].end <= seconds


def test_empty_audio_cannot_be_aligned() -> None:
    encoder = CtcEncoder(lambda window: np.zeros((1, len(VOCAB)), dtype=np.float32))

    assert encoder.log_probs(np.zeros(100, dtype=np.float32)).shape == (0, len(VOCAB))
    with pytest.raises(AlignmentError):
        encoder.align(np.zeros(100, dtype=np.float32), "hello")


def test_vocabulary_matches_the_pinned_model() -> None:
    assert len(VOCAB) == 32
    assert BLANK == 0
    assert VOCAB["|"] == 4
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'") <= set(VOCAB)
