"""CTC forced alignment in numpy: frame-level Viterbi, then word spans."""

import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FRAME_SECONDS = 0.02
WORD_DELIMITER = "|"
_NEG_INF = -1.0e30
_CURLY_APOSTROPHE = chr(0x2019)
_NON_SPOKEN = re.compile(r"[^A-Z' ]+")

LogProbs = npt.NDArray[np.floating]


class AlignmentError(ValueError):
    """The transcript cannot be aligned to this audio."""


@dataclass(frozen=True, slots=True)
class TokenSpan:
    token: int
    start: int
    end: int
    score: float


@dataclass(frozen=True, slots=True)
class WordSpan:
    word: str
    start: float
    end: float
    score: float


def normalise_words(transcript: str) -> list[str]:
    """Upper-case and strip everything a character CTC vocabulary cannot emit."""
    cleaned = _NON_SPOKEN.sub(" ", transcript.upper().replace(_CURLY_APOSTROPHE, "'"))
    return [word for word in (w.strip("'") for w in cleaned.split()) if word]


def viterbi_states(log_probs: LogProbs, tokens: list[int], blank: int) -> npt.NDArray[np.intp]:
    """Best CTC path as an index into the blank-interleaved token sequence, per frame."""
    if not tokens:
        raise AlignmentError("nothing to align")
    frames = log_probs.shape[0]
    extended = np.full(2 * len(tokens) + 1, blank, dtype=np.intp)
    extended[1::2] = tokens
    states = len(extended)
    repeats = int(np.sum(extended[3::2] == extended[1:-2:2]))
    if frames < len(tokens) + repeats:
        raise AlignmentError(f"{frames} frames cannot hold {len(tokens)} tokens")

    can_skip = np.zeros(states, dtype=bool)
    can_skip[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])

    emit = np.asarray(log_probs, dtype=np.float64)[:, extended]
    score = np.full(states, _NEG_INF)
    score[0] = emit[0, 0]
    score[1] = emit[0, 1]
    back = np.zeros((frames, states), dtype=np.int8)
    for frame in range(1, frames):
        stay = score
        step = np.concatenate(([_NEG_INF], score[:-1]))
        skip = np.where(can_skip, np.concatenate(([_NEG_INF, _NEG_INF], score[:-2])), _NEG_INF)
        choices = np.stack((stay, step, skip))
        best = np.argmax(choices, axis=0)
        back[frame] = best
        score = choices[best, np.arange(states)] + emit[frame]

    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    if score[state] <= _NEG_INF / 2:
        raise AlignmentError("no valid alignment path")
    path = np.empty(frames, dtype=np.intp)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    return path


def token_spans(log_probs: LogProbs, tokens: list[int], blank: int) -> list[TokenSpan]:
    """Frame span and mean log-probability of every token, in transcript order."""
    path = viterbi_states(log_probs, tokens, blank)
    spans: list[TokenSpan] = []
    for index, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * index + 1)
        start, end = int(frames[0]), int(frames[-1]) + 1
        score = float(np.mean(log_probs[start:end, token]))
        spans.append(TokenSpan(token=token, start=start, end=end, score=score))
    return spans


def align_words(
    log_probs: LogProbs,
    transcript: str,
    vocab: dict[str, int],
    blank: int,
    frame_seconds: float = FRAME_SECONDS,
) -> list[WordSpan]:
    """Align a transcript to CTC log-probabilities and return word timings in seconds."""
    words = normalise_words(transcript)
    unknown = sorted({char for word in words for char in word if char not in vocab})
    if unknown:
        raise AlignmentError(f"characters missing from the vocabulary: {''.join(unknown)}")
    if not words:
        raise AlignmentError("transcript has no alignable words")

    tokens: list[int] = []
    bounds: list[tuple[int, int]] = []
    for position, word in enumerate(words):
        if position:
            tokens.append(vocab[WORD_DELIMITER])
        bounds.append((len(tokens), len(tokens) + len(word)))
        tokens.extend(vocab[char] for char in word)

    spans = token_spans(log_probs, tokens, blank)
    return [
        WordSpan(
            word=word,
            start=spans[first].start * frame_seconds,
            end=spans[last - 1].end * frame_seconds,
            score=float(np.mean([span.score for span in spans[first:last]])),
        )
        for word, (first, last) in zip(words, bounds, strict=True)
    ]
