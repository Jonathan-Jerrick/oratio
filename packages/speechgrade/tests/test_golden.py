"""Pinned report hashes: any change to measured output must be deliberate."""

import pytest
from synth import Word, edit, passage, render

from speechgrade.pipeline import analyze

FIXTURES: dict[str, list[Word]] = {
    "clean": passage(),
    "quiet_phrase": edit(passage(), range(3, 6), level=0.06),
    "rushed_and_flat": [
        Word(w.text, 0.14 if 6 <= i < 9 else w.seconds, 150.0, w.level, w.pause)
        for i, w in enumerate(passage())
    ],
}
GOLDEN = {
    "clean": "d2199f907084274f9ae3045cdcc120f31074267e83053d568a32727d120ba357",
    "quiet_phrase": "07758040fa0dac4ab7e7a8d60d423e97210a45b8d81afb72116d90628ae52701",
    "rushed_and_flat": "eba41a73516c8bbd94f0ed7226f9319fe3b490a461ee2f52c244587e5f3d72ac",
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_report_hash_matches_golden(name: str) -> None:
    audio, spans = render(FIXTURES[name], seed=1)
    reference, reference_spans = render(passage(), seed=2)

    report = analyze(audio, spans, reference, reference_spans)

    assert report.provenance.report_sha256 == GOLDEN[name]
