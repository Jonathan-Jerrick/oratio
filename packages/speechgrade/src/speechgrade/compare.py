"""Pair participant words with baseline words and measure deviations."""

from dataclasses import dataclass
from difflib import SequenceMatcher

from speechgrade.words import WordFeatures


@dataclass(frozen=True, slots=True)
class Pair:
    """One step of the word pairing; a side is None where the other has an extra word."""

    participant: int | None
    baseline: int | None


def pair_words(participant: list[str], baseline: list[str]) -> list[Pair]:
    pairs: list[Pair] = []
    matcher = SequenceMatcher(a=participant, b=baseline, autojunk=False)
    for tag, p0, p1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(Pair(p0 + k, b0 + k) for k in range(p1 - p0))
            continue
        pairs.extend(Pair(index, None) for index in range(p0, p1))
        pairs.extend(Pair(None, index) for index in range(b0, b1))
    return pairs


def deviations(
    participant: list[WordFeatures],
    baseline: list[WordFeatures],
    pairs: list[Pair],
    sigma_good: dict[str, float],
) -> dict[int, dict[str, float]]:
    """z-score of each participant word against its baseline word, per feature."""
    result: dict[int, dict[str, float]] = {}
    for pair in pairs:
        if pair.participant is None or pair.baseline is None:
            continue
        ours, theirs = participant[pair.participant], baseline[pair.baseline]
        result[pair.participant] = {
            name: (ours[name] - theirs[name]) / sigma for name, sigma in sigma_good.items()
        }
    return result
