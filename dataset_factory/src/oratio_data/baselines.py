"""Cut candidate baseline passages from transcribed speech and screen them."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from oratio_data.dsp import SAMPLE_RATE, Samples

SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
MIN_PASSAGE_S = 45.0
TARGET_PASSAGE_S = 60.0
MAX_PASSAGE_S = 90.0
SENTENCE_GAP_S = 0.25
FORCED_BREAK_GAP_S = 0.7
CONTEXT_PAD_S = 0.3
LOW_CONFIDENCE = 0.5
MAX_LOW_CONFIDENCE_SHARE = 0.05
MIN_WORDS_PER_S = 1.5
MAX_WORDS_PER_S = 4.5
SNR_FRAME_S = 0.025
QA_MARKERS = (
    "question-and-answer",
    "question and answer",
    "first question",
    "open the line",
    "open up the line",
    "q&a",
)
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    probability: float
    speaker: str = ""


@dataclass(frozen=True, slots=True)
class Passage:
    first: int
    last: int
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def sentence_ends(words: list[Word]) -> list[int]:
    """Indices of words that end a sentence: closing punctuation with a pause, or a long pause."""
    ends = []
    for index, word in enumerate(words):
        gap = words[index + 1].start - word.end if index + 1 < len(words) else float("inf")
        punctuated = bool(SENTENCE_END.search(word.text.strip()))
        if (punctuated and gap >= SENTENCE_GAP_S) or gap >= FORCED_BREAK_GAP_S:
            ends.append(index)
    return ends


def cut_passages(words: list[Word], max_passages: int) -> list[Passage]:
    """Consecutive, non-overlapping passages of 45 to 90 s that start and end on sentences."""
    passages: list[Passage] = []
    first = 0
    ends = sentence_ends(words)
    position = 0
    while position < len(ends) and len(passages) < max_passages:
        end = ends[position]
        span = words[end].end - words[first].start
        if span > MAX_PASSAGE_S:
            first, position = end + 1, position + 1
            continue
        following = ends[position + 1] if position + 1 < len(ends) else None
        longer = words[following].end - words[first].start if following is not None else None
        stop_here = span >= TARGET_PASSAGE_S or longer is None or longer > MAX_PASSAGE_S
        if span >= MIN_PASSAGE_S and stop_here:
            before = words[first - 1].end if first else 0.0
            after = words[end + 1].start if end + 1 < len(words) else words[end].end + CONTEXT_PAD_S
            start = max(before, words[first].start - CONTEXT_PAD_S, 0.0)
            stop = min(after, words[end].end + CONTEXT_PAD_S)
            passages.append(Passage(first, end, start, stop))
            first = end + 1
        position += 1
    return passages


def energy_snr_db(samples: Samples) -> float:
    """Speech-to-floor ratio: 90th versus 10th percentile of 25 ms frame energy, in dB."""
    frame = round(SNR_FRAME_S * SAMPLE_RATE)
    count = len(samples) // frame
    if count < 10:
        return 0.0
    blocks = samples[: count * frame].astype(np.float64).reshape(count, frame)
    level = 10.0 * np.log10(np.mean(blocks**2, axis=1) + 1e-12)
    return float(np.percentile(level, 90) - np.percentile(level, 10))


def screen(words: list[Word], passage: Passage) -> list[str]:
    """Reasons a passage is unfit as a baseline; empty when it passes these text-level checks."""
    inside = words[passage.first : passage.last + 1]
    reasons = []
    low = sum(word.probability < LOW_CONFIDENCE for word in inside) / len(inside)
    if low > MAX_LOW_CONFIDENCE_SHARE:
        reasons.append("low_asr_confidence")
    rate = len(inside) / passage.seconds
    if not MIN_WORDS_PER_S <= rate <= MAX_WORDS_PER_S:
        reasons.append("speech_rate")
    if len({word.speaker for word in inside}) > 1:
        reasons.append("multiple_speakers")
    return reasons


def prepared_remarks_end(words: list[Word]) -> int:
    """Index of the first word of the Q&A session, or len(words) if none is announced."""
    tokens: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(words):
        for token in normalise(word.text).split():
            tokens.append(token)
            owner.append(index)
    found = len(words)
    for marker in (normalise(m).split() for m in QA_MARKERS):
        for start in range(len(tokens) - len(marker) + 1):
            if tokens[start : start + len(marker)] == marker:
                found = min(found, owner[start])
                break
    return found


def assign_speakers(words: list[Word], tokens: list[tuple[str, str]]) -> list[Word]:
    """Label ASR words with reference-transcript speakers by aligning the two word sequences."""
    asr = [normalise(w.text) for w in words]
    reference = [normalise(token) for token, _ in tokens]
    matcher = SequenceMatcher(a=asr, b=reference, autojunk=False)
    speaker: list[str | None] = [None] * len(words)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            speaker[block.a + offset] = tokens[block.b + offset][1]
    known = [i for i, label in enumerate(speaker) if label is not None]
    if not known:
        return [Word(w.text, w.start, w.end, w.probability, "") for w in words]
    nearest = np.searchsorted(known, np.arange(len(words)))
    labelled = []
    for index, word in enumerate(words):
        label = speaker[index]
        if label is None:
            left = known[max(0, nearest[index] - 1)]
            right = known[min(len(known) - 1, nearest[index])]
            label = speaker[left if index - left <= right - index else right]
        labelled.append(Word(word.text, word.start, word.end, word.probability, label or ""))
    return labelled


def read_nlp(text: str) -> list[tuple[str, str]]:
    """(token, speaker) pairs from a Rev .nlp transcript."""
    lines = text.splitlines()[1:]
    return [(parts[0], parts[1]) for parts in (line.split("|") for line in lines) if len(parts) > 1]
