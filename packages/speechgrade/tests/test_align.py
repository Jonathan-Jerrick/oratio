import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from speechgrade.align import (
    AlignmentError,
    align_words,
    normalise_words,
    token_spans,
    viterbi_states,
)

BLANK = 0
VOCAB = {"|": 1, **{chr(ord("A") + i): i + 2 for i in range(26)}, "'": 28}
CLASSES = 29


def _peaked(frame_tokens: list[int], classes: int = CLASSES) -> np.ndarray:
    """Log-probabilities that put almost all mass on the given token in each frame."""
    probs = np.full((len(frame_tokens), classes), 1e-4)
    probs[np.arange(len(frame_tokens)), frame_tokens] = 1.0
    return np.log(probs / probs.sum(axis=1, keepdims=True))


def _collapse(frame_tokens: list[int]) -> list[int]:
    return [t for t, _ in itertools.groupby(frame_tokens) if t != BLANK]


def test_recovers_exact_frame_spans() -> None:
    frames = [0, 0, 5, 5, 5, 0, 7, 7, 0, 0]

    spans = token_spans(_peaked(frames), [5, 7], BLANK)

    assert [(s.token, s.start, s.end) for s in spans] == [(5, 2, 5), (7, 6, 8)]
    assert all(s.score > -0.01 for s in spans)


def test_repeated_token_needs_a_blank_between() -> None:
    frames = [4, 4, 0, 4, 4]

    spans = token_spans(_peaked(frames), [4, 4], BLANK)

    assert [(s.start, s.end) for s in spans] == [(0, 2), (3, 5)]


def test_too_few_frames_is_an_error() -> None:
    with pytest.raises(AlignmentError, match="cannot hold"):
        viterbi_states(_peaked([4, 4]), [4, 4], BLANK)
    with pytest.raises(AlignmentError, match="nothing to align"):
        viterbi_states(_peaked([0, 0]), [], BLANK)


def test_best_path_matches_brute_force_on_small_inputs() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        tokens = [int(t) for t in rng.integers(1, 4, size=rng.integers(1, 4))]
        frames = int(rng.integers(len(tokens) * 2, 7))
        log_probs = np.log(rng.dirichlet(np.ones(4), size=frames))

        path = viterbi_states(log_probs, tokens, BLANK)
        extended = [BLANK if i % 2 == 0 else tokens[i // 2] for i in range(2 * len(tokens) + 1)]
        found = sum(log_probs[f, extended[s]] for f, s in enumerate(path))
        best = max(
            sum(log_probs[f, c] for f, c in enumerate(candidate))
            for candidate in itertools.product(range(4), repeat=frames)
            if _collapse(list(candidate)) == tokens
        )

        assert found == pytest.approx(best)


@settings(max_examples=60, deadline=None)
@given(
    runs=st.lists(
        st.tuples(st.integers(1, 5), st.integers(1, 4), st.integers(1, 3)), min_size=1, max_size=8
    )
)
def test_spans_are_ordered_and_recover_any_peaked_path(runs: list[tuple[int, int, int]]) -> None:
    frames: list[int] = []
    expected: list[tuple[int, int, int]] = []
    for token, length, gap in runs:
        frames.extend([BLANK] * gap)
        expected.append((token, len(frames), len(frames) + length))
        frames.extend([token] * length)

    spans = token_spans(_peaked(frames, classes=6), [token for token, _, _ in runs], BLANK)

    assert [(s.token, s.start, s.end) for s in spans] == expected
    assert all(a.end <= b.start for a, b in itertools.pairwise(spans))


def test_normalise_words_keeps_only_spoken_characters() -> None:
    assert normalise_words("We shall pay any price — bear any burden, 1961!") == [
        "WE", "SHALL", "PAY", "ANY", "PRICE", "BEAR", "ANY", "BURDEN",
    ]  # fmt: skip
    assert normalise_words(f"don{chr(0x2019)}t 'quote'") == ["DON'T", "QUOTE"]


def test_align_words_returns_times_in_seconds() -> None:
    frames = [0, VOCAB["H"], VOCAB["I"], 0, VOCAB["|"], VOCAB["Y"], VOCAB["O"], VOCAB["U"], 0]

    words = align_words(_peaked(frames), "Hi, you.", VOCAB, BLANK)

    assert [(w.word, round(w.start, 2), round(w.end, 2)) for w in words] == [
        ("HI", 0.02, 0.06),
        ("YOU", 0.10, 0.16),
    ]


@pytest.mark.parametrize(
    ("transcript", "message"),
    [("...", "no alignable words"), ("", "no alignable words")],
)
def test_align_words_rejects_empty_transcripts(transcript: str, message: str) -> None:
    with pytest.raises(AlignmentError, match=message):
        align_words(_peaked([0, 0, 0]), transcript, VOCAB, BLANK)


def test_align_words_rejects_characters_outside_the_vocabulary() -> None:
    with pytest.raises(AlignmentError, match="missing from the vocabulary: Z"):
        align_words(_peaked([0, 0, 0]), "zoo", {"|": 1, "O": 2}, BLANK)
