"""Analyse a participant recording against a baseline and build the report."""

import base64

import numpy as np

from speechgrade import __version__
from speechgrade.align import WordSpan
from speechgrade.audio import Audio
from speechgrade.compare import Pair, deviations, pair_words
from speechgrade.features import (
    HOP_SECONDS,
    FloatArray,
    Tracks,
    extract_tracks,
    relative_energy,
    semitones,
)
from speechgrade.ground import find_regions
from speechgrade.hashing import content_hash
from speechgrade.rubric import Rubric, load_rubric
from speechgrade.schema import SCHEMA_VERSION, Provenance, Report, WordRecord
from speechgrade.score import score_regions
from speechgrade.words import word_features

TRACK_DECIMALS = 2
_UNHASHED = "0" * 64


def _encode(values: FloatArray) -> str:
    quantised = np.round(values.astype(np.float64), TRACK_DECIMALS).astype("<f4")
    return base64.b64encode(quantised.tobytes()).decode("ascii")


def _encode_tracks(tracks: Tracks) -> dict[str, str]:
    return {
        "hop_seconds": str(HOP_SECONDS),
        "f0_st": _encode(semitones(tracks.f0_hz)),
        "energy_db": _encode(relative_energy(tracks.energy_db)),
        "hf_ratio": _encode(tracks.hf_ratio),
        "cpp_db": _encode(tracks.cpp_db),
    }


def _records(words: list[WordSpan], partner: dict[int, int]) -> tuple[WordRecord, ...]:
    return tuple(
        WordRecord(
            word=word.word,
            start=round(word.start, 4),
            end=round(word.end, 4),
            score=round(word.score, 4),
            baseline_index=partner.get(index),
        )
        for index, word in enumerate(words)
    )


def _matched(pairs: list[Pair]) -> dict[int, int]:
    return {
        p.participant: p.baseline
        for p in pairs
        if p.participant is not None and p.baseline is not None
    }


def analyze(
    participant: Audio,
    participant_words: list[WordSpan],
    baseline: Audio,
    baseline_words: list[WordSpan],
    rubric: Rubric | None = None,
) -> Report:
    rubric = rubric or load_rubric()
    tracks = extract_tracks(participant.samples)
    baseline_tracks = extract_tracks(baseline.samples)
    features = word_features(participant_words, tracks)
    baseline_features = word_features(baseline_words, baseline_tracks)
    pairs = pair_words([w.word for w in participant_words], [w.word for w in baseline_words])
    z_scores = deviations(features, baseline_features, pairs, rubric.sigma_good)
    regions = find_regions(
        participant_words, baseline_words, features, baseline_features, pairs, z_scores, rubric
    )

    def build(report_sha256: str) -> Report:
        return Report(
            schema_version=SCHEMA_VERSION,
            participant_seconds=round(participant.seconds, 4),
            baseline_seconds=round(baseline.seconds, 4),
            words=_records(participant_words, _matched(pairs)),
            baseline_words=_records(baseline_words, {}),
            regions=tuple(regions),
            scores=score_regions(regions, rubric),
            tracks={
                "participant": _encode_tracks(tracks),
                "baseline": _encode_tracks(baseline_tracks),
            },
            provenance=Provenance(
                speechgrade_version=__version__,
                rubric_version=rubric.version,
                rubric_sha256=rubric.sha256,
                report_sha256=report_sha256,
            ),
        )

    return build(content_hash(build(_UNHASHED).model_dump(mode="json")))
