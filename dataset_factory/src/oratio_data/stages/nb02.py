"""NB02: select baselines and generate OCDD v1 (controls, levels, sweeps) with exact labels."""

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from oratio_data.config import load_config
from oratio_data.gates import check_clip, longest_zero_run_s
from oratio_data.injectors import PlanError, Result, apply_plan
from oratio_data.licenses import LicenseError, require_publishable
from oratio_data.plan import Placement, coverage, plan_level, plan_sweep
from speechgrade.align import AlignmentError, WordSpan, normalise_words
from speechgrade.encoder import CtcEncoder

QUOTAS = {"S-LV": 15, "S-LV-ORATIONS": 10, "S-PD-COMMONS": 10, "S-EARN": 12}
VARIANTS = 2
SWEEP_EVERY = 4
SWEEP_CODES = ("PAC-RUSH", "PAU-LONG", "INT-MONO", "ENE-LOW", "CLA-MUFF", "FLU-PROL")
MAX_DRIFT_MS = 30.0
ATTEMPTS = 3
SPLIT_SHARES = (("train", 70), ("val", 15), ("test", 15))
METADATA_FIELDS = (
    "clip_id", "baseline_id", "source_id", "split", "kind", "level", "variant", "code",
    "magnitude", "seconds", "regions", "coverage", "drift_ms_median", "drift_ms_p95",
    "license_class", "speaker", "evidence_url",
)  # fmt: skip


def split_of(item: str) -> str:
    """Deterministic grouped split: every passage of one recording lands in the same split."""
    bucket = int(hashlib.sha256(item.encode()).hexdigest(), 16) % 100
    for name, share in SPLIT_SHARES:
        if bucket < share:
            return name
        bucket -= share
    return SPLIT_SHARES[-1][0]


def select_baselines(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Best passages per source by SNR, one per recording, only redistributable licences."""
    chosen: list[dict[str, str]] = []
    for source, quota in QUOTAS.items():
        pool = [r for r in rows if r["source_id"] == source and not r["reasons"]]
        pool.sort(key=lambda r: (-float(r["snr_db"]), r["candidate_id"]))
        items: set[str] = set()
        for row in pool:
            if len([c for c in chosen if c["source_id"] == source]) == quota:
                break
            if row["item"] in items:
                continue
            try:
                require_publishable(row["license_class"], row["candidate_id"])
            except LicenseError:
                continue
            items.add(row["item"])
            chosen.append({**row, "split": split_of(row["item"])})
    return chosen


def baseline_words(transcript: dict[str, Any]) -> list[WordSpan]:
    """Transcript words as alignable single tokens (punctuation and spacing removed)."""
    words = []
    for word in transcript["words"]:
        token = "".join(normalise_words(word["text"]))
        if token:
            words.append(WordSpan(token, float(word["start"]), float(word["end"]), 0.0))
    return words


def drift_ms(
    encoder: CtcEncoder,
    result: Result,
    reference: list[WordSpan],
    kept: list[int],
    checked: list[int],
) -> tuple[float, float]:
    """Re-align the clip and compare word starts with the warped reference timing.

    Only words outside every edited span are scored: they test that the warp carried the labels
    correctly, while words inside heavy edits mostly measure the aligner's own confusion.
    """
    expected = [reference[i] for i in kept]
    found = encoder.align(result.samples, " ".join(w.word for w in expected))
    position = {index: n for n, index in enumerate(kept)}
    rows = [position[i] for i in checked] or list(range(len(kept)))
    starts = np.array([found[n].start for n in rows])
    target = result.warp.apply([expected[n].start for n in rows])
    error = np.abs(starts - target) * 1000.0
    return float(np.median(error)), float(np.percentile(error, 95))


def _plans(
    baseline: dict[str, str], words: list[WordSpan], attempt: int
) -> list[tuple[str, str, int, int, str, float, list[Placement]]]:
    base_seed = int(hashlib.sha256(baseline["candidate_id"].encode()).hexdigest(), 16) % 10**8
    seed = base_seed + 1000 * attempt
    cid = baseline["candidate_id"]
    plans: list[tuple[str, str, int, int, str, float, list[Placement]]] = [
        (f"{cid}__L0p", "control", 0, 0, "", 0.0, [])
    ]
    for level in range(1, 6):
        for variant in range(VARIANTS):
            plan = plan_level(words, f"L{level}", seed + 10 * level + variant)
            plans.append((f"{cid}__L{level}__v{variant}", "graded", level, variant, "", 0.0, plan))
    if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % SWEEP_EVERY == 0:
        for code in SWEEP_CODES:
            try:
                sweep = plan_sweep(words, code, seed)
            except PlanError:
                continue
            for placement in sweep:
                magnitude = placement.flaw.magnitude
                clip = f"{cid}__SW__{code}__m{round(magnitude * 100):03d}"
                plans.append((clip, "sweep", 0, 0, code, magnitude, [placement]))
    return plans


def generate(
    root: Path, runner: Callable[[np.ndarray], np.ndarray], limit: int | None = None
) -> dict[str, int]:
    """Generate every clip for every selected baseline; resumes by skipping finished clips."""
    candidates = root / "candidates"
    out = root / "ocdd"
    for sub in ("labels",):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with (candidates / "candidates.csv").open(newline="") as handle:
        baselines = select_baselines(list(csv.DictReader(handle)))[:limit]
    with (out / "baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0]))
        writer.writeheader()
        writer.writerows(baselines)
    encoder = CtcEncoder(runner)
    counts = {"clips": 0, "skipped": 0, "gate_failed": 0, "plan_failed": 0}
    for number, baseline in enumerate(baselines, start=1):
        cid = baseline["candidate_id"]
        samples, _ = sf.read(candidates / "audio" / f"{cid}.flac", dtype="float32")
        transcript = json.loads((candidates / "transcripts" / f"{cid}.json").read_text())
        words = baseline_words(transcript)
        control = apply_plan(samples, [], seed=0)
        try:
            reference = encoder.align(control.samples, " ".join(w.word for w in words))
        except AlignmentError as error:
            print(f"{number}/{len(baselines)} {cid}: reference alignment failed: {error}")
            counts["plan_failed"] += 1
            continue
        for attempt in range(ATTEMPTS):
            try:
                plans = _plans(baseline, words, attempt)
                break
            except PlanError:
                plans = []
        made = 0
        for clip_id, kind, level, variant, code, magnitude, plan in plans:
            label_path = out / "labels" / f"{clip_id}.json"
            if label_path.exists():
                counts["skipped"] += 1
                continue
            record = _make_clip(out, baseline, words, reference, encoder, samples,
                                clip_id, kind, level, variant, code, magnitude, plan)  # fmt: skip
            if record is None:
                counts["gate_failed"] += 1
                continue
            label_path.write_text(json.dumps(record))
            counts["clips"] += 1
            made += 1
        print(f"{number}/{len(baselines)} {cid} [{baseline['split']}]: {made} clips", flush=True)
    write_metadata(root)
    return counts


def _make_clip(
    out: Path, baseline: dict[str, str], words: list[WordSpan], reference: list[WordSpan],
    encoder: CtcEncoder, samples: np.ndarray, clip_id: str, kind: str, level: int,
    variant: int, code: str, magnitude: float, plan: list[Placement],
) -> dict[str, Any] | None:  # fmt: skip
    config = load_config()
    flaws = [p.flaw for p in plan]
    result = apply_plan(
        samples, flaws, seed=int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % 10**8
    )
    failures = check_clip(result, config.gates, longest_zero_run_s(samples))
    omitted = {i for i, w in enumerate(words) for f in flaws
               if f.code == "FLU-OMIT" and f.start <= w.start and w.end <= f.end}  # fmt: skip
    kept = [i for i in range(len(words)) if i not in omitted]
    untouched = [
        i for i in kept if not any(f.start < words[i].end and words[i].start < f.end for f in flaws)
    ]
    try:
        median, p95 = drift_ms(encoder, result, reference, kept, untouched)
    except AlignmentError:
        median, p95 = float("inf"), float("inf")
    if median > MAX_DRIFT_MS:
        failures.append("drift")
    if failures:
        print(f"  gate failed {clip_id}: {failures} drift {median:.0f} ms", flush=True)
        return None
    audio_path = out / "audio" / baseline["split"] / f"{clip_id}.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, result.samples, 16_000, subtype="PCM_16")
    warped = [
        {"text": words[i].word, "start": round(float(s), 4), "end": round(float(e), 4)}
        for i in kept
        for s, e in [result.warp.span(words[i].start, words[i].end)]
    ]
    return {
        "clip_id": clip_id, "baseline_id": f"{baseline['candidate_id']}__L0p",
        "passage_id": baseline["candidate_id"], "source_id": baseline["source_id"],
        "split": baseline["split"], "kind": kind, "level": level, "variant": variant,
        "code": code, "magnitude": round(magnitude, 4), "seconds": len(result.samples) / 16_000,
        "license_class": baseline["license_class"], "evidence_url": baseline["evidence_url"],
        "speaker": baseline["speaker"], "synthetic_modification": kind != "control",
        "generator": "psola", "impl_family": "A",
        "coverage": round(coverage(plan, len(samples) / 16_000), 4),
        "drift_ms_median": round(median, 2), "drift_ms_p95": round(p95, 2),
        "regions": [{**asdict(label), "start": round(label.start, 4), "end": round(label.end, 4),
                     "source_start": round(label.source_start, 4),
                     "source_end": round(label.source_end, 4)} for label in result.labels],
        "words": warped,
        "audio": str(audio_path.relative_to(out)),
    }  # fmt: skip


def write_metadata(root: Path) -> int:
    out = root / "ocdd"
    rows = []
    for path in sorted((out / "labels").glob("*.json")):
        record = json.loads(path.read_text())
        rows.append({
            **{k: record[k] for k in METADATA_FIELDS if k in record},
            "regions": len(record["regions"]),
        })  # fmt: skip
    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
