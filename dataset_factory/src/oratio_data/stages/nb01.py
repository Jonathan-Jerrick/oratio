"""NB01: transcribe source speeches and cut screened 45 to 90 s candidate passages."""

import csv
import json
import shutil
import subprocess
import time
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from oratio_data import asr
from oratio_data.baselines import (
    Passage,
    Word,
    assign_speakers,
    bitrate_rank,
    cut_by_speaker,
    energy_snr_db,
    prepared_remarks_end,
    read_nlp,
    recording_key,
    screen,
)
from oratio_data.dsp import SAMPLE_RATE

TARGET_LUFS = -23.0
MIN_SNR_DB = 15.0
ENGLISH_MIN_PROBABILITY = 0.9
ORATION_SAMPLE = 120
CAP_SECONDS = {"S-PD-COMMONS": 600.0, "S-LV": 600.0, "S-LV-ORATIONS": 600.0, "S-EARN": 720.0}
MAX_PASSAGES = {"S-PD-COMMONS": 3, "S-LV": 2, "S-LV-ORATIONS": 2, "S-EARN": 2}
CANDIDATE_FIELDS = (
    "candidate_id",
    "source_id",
    "item",
    "source_file",
    "start",
    "end",
    "seconds",
    "words",
    "words_per_s",
    "snr_db",
    "loudness_lufs",
    "speaker",
    "license_class",
    "evidence_url",
    "language_probability",
    "reasons",
)


@dataclass(frozen=True, slots=True)
class Item:
    key: str
    source_id: str
    path: Path
    license_class: str
    evidence_url: str
    nlp: Path | None = None
    shard: Path | None = None
    row: int = 0


def _register(root: Path) -> list[dict[str, str]]:
    with (root / "manifests" / "rights_register.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _audio_items(root: Path, seed: int) -> Iterator[Item]:
    rows = _register(root)
    orations = [r for r in rows if r["source_id"] == "S-LV-ORATIONS"]
    picked = np.random.default_rng(seed).permutation(len(orations))[:ORATION_SAMPLE]
    chosen = {orations[i]["file"] for i in picked}
    best: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: bitrate_rank(r["file"])):
        best.setdefault(recording_key(row["file"]), row)
    for row in sorted(best.values(), key=lambda r: r["file"]):
        if row["source_id"] in ("S-PD-COMMONS", "S-LV") or row["file"] in chosen:
            key = Path(row["file"]).with_suffix("").as_posix().replace("/", "__")
            yield Item(
                key, row["source_id"], root / row["file"], row["license_class"], row["evidence_url"]
            )


def _earnings_items(root: Path, scratch: Path) -> Iterator[Item]:
    import pyarrow.parquet as pq

    rev = root / "raw" / "earnings" / "rev_speech_datasets"
    for corpus in ("earnings22", "earnings21"):
        for shard in sorted((root / "raw" / "hf" / f"{corpus}_full").rglob("*.parquet")):
            ids = pq.read_table(shard, columns=["file_id"]).column("file_id").to_pylist()
            for row, file_id in enumerate(ids):
                nlp = next(iter(sorted((rev / corpus).rglob(f"{file_id}.nlp"))), None)
                url = f"https://huggingface.co/datasets/distil-whisper/{corpus}"
                path = scratch / f"{corpus}__{file_id}.audio"
                yield Item(f"{corpus}__{file_id}", "S-EARN", path, "cc_by_sa", url, nlp, shard, row)


@lru_cache(maxsize=1)
def _shard_audio(shard: Path) -> list[bytes]:
    import pyarrow.parquet as pq

    return [a["bytes"] for a in pq.read_table(shard, columns=["audio"]).column("audio").to_pylist()]


def _materialise(item: Item) -> None:
    if item.shard is not None and not item.path.exists():
        item.path.write_bytes(_shard_audio(item.shard)[item.row])


def transcribe(
    root: Path, scratch: Path, per_source: int | None = None, seed: int = 0
) -> dict[str, int]:
    """Language-check and transcribe every source; skips items already done.

    per_source caps how many items of each source are attempted, for smoke runs.
    """
    asr.check()
    out = root / "asr"
    out.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "not_english": 0, "failed": 0}
    items = chain(_audio_items(root, seed), _earnings_items(root, scratch))
    seen: dict[str, int] = {}
    for item in items:
        seen[item.source_id] = seen.get(item.source_id, 0) + 1
        if per_source is not None and seen[item.source_id] > per_source:
            continue
        target = out / f"{item.key}.json"
        if target.exists():
            counts["skipped"] += 1
            continue
        try:
            _materialise(item)
            audio = asr.decode(item.path, seconds=CAP_SECONDS[item.source_id])
            lang, probability = asr.language(audio)
            words = (
                asr.transcribe(audio)
                if lang == "en" and probability >= ENGLISH_MIN_PROBABILITY
                else []
            )
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
            print(f"FAILED {item.key}: {error!r}"[:300], flush=True)
            counts["failed"] += 1
            continue
        counts["done" if words else "not_english"] += 1
        record = {
            **asdict(item),
            "path": str(item.path),
            "shard": str(item.shard) if item.shard else None,
            "nlp": str(item.nlp) if item.nlp else None,
            "language": lang,
            "language_probability": probability,
            "seconds": len(audio) / SAMPLE_RATE,
            "words": [asdict(w) for w in words],
        }
        target.write_text(json.dumps(record))
        print(f"{item.key}: {lang} {probability:.2f}, {len(words)} words", flush=True)
    return counts


def _loudness_normalise(audio: np.ndarray) -> tuple[np.ndarray, float]:
    import pyloudnorm

    meter = pyloudnorm.Meter(SAMPLE_RATE)
    loudness = float(meter.integrated_loudness(audio.astype(np.float64)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Possible clipped samples")
        gained = pyloudnorm.normalize.loudness(audio.astype(np.float64), loudness, TARGET_LUFS)
    peak = float(np.max(np.abs(gained)))
    if peak > 0.89:
        gained = gained * (0.89 / peak)
    return gained.astype(np.float32), loudness


def _cut_one(record: dict[str, Any], out: Path) -> list[dict[str, str]]:
    words = [Word(**w) for w in record["words"]]
    if record["source_id"] == "S-EARN":
        if not record["nlp"]:
            return []
        words = assign_speakers(words, read_nlp(Path(record["nlp"]).read_text()))
        words = words[: prepared_remarks_end(words)]
    source = Path(record["path"])
    if record.get("shard") and not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(_shard_audio(Path(record["shard"]))[record["row"]])
    rows = []
    passages: list[Passage] = cut_by_speaker(words, MAX_PASSAGES[record["source_id"]])
    for number, passage in enumerate(passages):
        reasons = screen(words, passage)
        audio = asr.decode(source, passage.start, passage.seconds)
        snr = energy_snr_db(audio)
        if snr < MIN_SNR_DB:
            reasons.append("snr")
        normalised, loudness = _loudness_normalise(audio)
        candidate = f"{record['key']}__p{number}"
        if not reasons:
            sf.write(out / "audio" / f"{candidate}.flac", normalised, SAMPLE_RATE)
            inside = words[passage.first : passage.last + 1]
            transcript = {
                "candidate_id": candidate,
                "text": " ".join(w.text for w in inside),
                "words": [
                    {
                        "text": w.text,
                        "start": round(w.start - passage.start, 3),
                        "end": round(w.end - passage.start, 3),
                    }
                    for w in inside
                ],
            }
            (out / "transcripts" / f"{candidate}.json").write_text(json.dumps(transcript))
        rows.append(
            {
                "candidate_id": candidate,
                "source_id": record["source_id"],
                "item": record["key"],
                "source_file": record["path"],
                "start": f"{passage.start:.3f}",
                "end": f"{passage.end:.3f}",
                "seconds": f"{passage.seconds:.2f}",
                "words": str(passage.last - passage.first + 1),
                "words_per_s": f"{(passage.last - passage.first + 1) / passage.seconds:.2f}",
                "snr_db": f"{snr:.1f}",
                "loudness_lufs": f"{loudness:.1f}",
                "speaker": words[passage.first].speaker,
                "license_class": record["license_class"],
                "evidence_url": record["evidence_url"],
                "language_probability": f"{record['language_probability']:.3f}",
                "reasons": ";".join(reasons),
            }
        )
    return rows


def cut(root: Path) -> dict[str, int]:
    """Cut, screen and loudness-normalise passages; writes candidates.csv from scratch."""
    out = root / "candidates"
    for sub in ("audio", "transcripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    records = [json.loads(p.read_text()) for p in sorted((root / "asr").glob("*.json"))]
    best: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: bitrate_rank(r["path"])):
        best.setdefault(recording_key(record["path"]), record)
    for record in sorted(best.values(), key=lambda r: r["key"]):
        if record["words"]:
            rows.extend(_cut_one(record, out))
    with (out / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = [r for r in rows if not r["reasons"]]
    return {
        "candidates": len(rows),
        "passed": len(passed),
        **{s: sum(r["source_id"] == s for r in passed) for s in MAX_PASSAGES},
    }


def write_manifest(root: Path, name: str, payload: dict[str, Any]) -> None:
    sha = subprocess.run(
        ["git", "-C", "/content/oratio", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gpu = shutil.which("nvidia-smi") is not None
    manifest = {
        "stage": name,
        "git_sha": sha,
        "gpu": gpu,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    (root / "stage_manifests" / f"{name}.json").write_text(json.dumps(manifest, indent=1))
