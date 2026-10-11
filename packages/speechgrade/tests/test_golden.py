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
    "clean": "1cef6e0905d142a62bd6ee8e923459531f97619dfcf904e7755d14bdb2e739b8",
    "quiet_phrase": "7054ac30a104fa18940c4d6c7632d862919e1c3139bb0b859323cc557b1a52e9",
    "rushed_and_flat": "6d7e8fafbf8ab5a22d77c7ab46866198096eafc5f16a54c3f53625350d6cb6dd",
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_report_hash_matches_golden(name: str) -> None:
    audio, spans = render(FIXTURES[name], seed=1)
    reference, reference_spans = render(passage(), seed=2)

    report = analyze(audio, spans, reference, reference_spans)

    assert report.provenance.report_sha256 == GOLDEN[name]
