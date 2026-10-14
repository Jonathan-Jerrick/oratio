"""Demo assets: Gettysburg baselines for the API and precomputed dashboard samples.

Built locally from public-domain LibriVox readings so the live demo does not depend on Colab.
"""

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from oratio_data.injectors import apply_plan
from oratio_data.plan import plan_level
from speechgrade.align import WordSpan
from speechgrade.audio import Audio
from speechgrade.encoder import BLANK, FRAME_SECONDS, VOCAB, CtcEncoder
from speechgrade.pipeline import analyze

ITEM = "gettysburgaddress_150anniv_1311_librivox"
READERS = ("amb", "rk", "wt", "jn", "mk", "dl", "pc")
WANTED_READERS = 3
LEVELS = (2, 4)
PAD_S = 0.4
SAMPLE_RATE = 16_000
_LETTERS = {index: char for char, index in VOCAB.items() if len(char) == 1}


def _decode(path: Path) -> np.ndarray:
    wav = path.with_suffix(".wav")
    if not wav.exists():
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{SAMPLE_RATE}",
                "-c",
                "1",
                str(path),
                str(wav),
            ],
            check=True,
        )
    samples, _ = sf.read(wav, dtype="float32")
    return np.asarray(samples, dtype=np.float32)


def _greedy(log_probs: np.ndarray) -> tuple[str, list[int]]:
    """Greedy CTC transcript with the frame index of every character."""
    best = log_probs.argmax(axis=1)
    chars, frames, previous = [], [], BLANK
    for frame, token in enumerate(best):
        if token != previous and token != BLANK and int(token) in _LETTERS:
            chars.append(_LETTERS[int(token)])
            frames.append(frame)
        previous = int(token)
    return "".join(chars), frames


def _address(encoder: CtcEncoder, samples: np.ndarray) -> tuple[float, float, str]:
    """Start, end and spoken text of the Address inside a LibriVox track (intro and outro cut).

    Readers use different versions of the text, so the passage keeps what this reader said.
    """
    text, frames = _greedy(encoder.log_probs(samples))
    score = text.find("SCORE")
    start = text.rfind("|", 0, score) + 1
    if text[max(0, start - 5) : start] == "FOUR|":
        start -= 5
    end = text.rfind("EARTH")
    if score < 0 or end < 0:
        raise ValueError("could not locate the Address in the reading")
    end += len("EARTH")
    spoken = text[start:end].replace("|", " ")
    return frames[start] * FRAME_SECONDS - PAD_S, frames[end - 1] * FRAME_SECONDS + PAD_S, spoken


def build(work: Path, web: Path, baselines: Path, encoder: CtcEncoder) -> list[dict[str, Any]]:
    work.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    baselines.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    used = 0
    for reader in READERS:
        if used == WANTED_READERS:
            break
        name = f"gettysburgaddress_lincoln_{reader}_64kb.mp3"
        mp3 = work / name
        if not mp3.exists():
            urllib.request.urlretrieve(f"https://archive.org/download/{ITEM}/{name}", mp3)
        full = _decode(mp3)
        try:
            start, end, spoken = _address(encoder, full)
        except ValueError:
            print(f"skipping reader {reader}: Address not found in the transcript")
            continue
        used += 1
        passage = full[max(0, round(start * SAMPLE_RATE)) : round(end * SAMPLE_RATE)]
        control = apply_plan(passage, [], seed=0)
        words = encoder.align(control.samples, spoken)
        baseline_id = f"gettysburg_{reader}__L0p"
        sf.write(baselines / f"{baseline_id}.flac", control.samples, SAMPLE_RATE, subtype="PCM_16")
        record = {
            "title": f"Gettysburg Address (LibriVox reader {reader})",
            "source": "S-LV",
            "license": "public_domain",
            "evidence_url": f"https://archive.org/details/{ITEM}",
            "words": [{"text": w.word, "start": w.start, "end": w.end} for w in words],
        }
        (baselines / f"{baseline_id}.json").write_text(json.dumps(record))
        _m4a(baselines / f"{baseline_id}.flac", web / f"{baseline_id}.m4a")
        for level in LEVELS:
            plan = plan_level(words, f"L{level}", seed=1000 * level + used)
            result = apply_plan(control.samples, [p.flaw for p in plan], seed=level)
            warped = [
                WordSpan(w.word, *result.warp.span(w.start, w.end), w.score)
                for w in words
                if not any(
                    p.flaw.code == "FLU-OMIT" and p.flaw.start <= w.start and w.end <= p.flaw.end
                    for p in plan
                )
            ]
            report = analyze(
                Audio(result.samples, SAMPLE_RATE),
                warped,
                Audio(control.samples, SAMPLE_RATE),
                list(words),
            )
            sample_id = f"gettysburg_{reader}__L{level}"
            flac = work / f"{sample_id}.flac"
            sf.write(flac, result.samples, SAMPLE_RATE, subtype="PCM_16")
            _m4a(flac, web / f"{sample_id}.m4a")
            (web / f"{sample_id}.json").write_text(report.model_dump_json())
            truth = [
                {
                    "code": label.code,
                    "dimension": label.dimension,
                    "magnitude": round(label.magnitude, 3),
                    "start": round(label.start, 3),
                    "end": round(label.end, 3),
                }
                for label in result.labels
            ]
            (web / f"{sample_id}.labels.json").write_text(json.dumps(truth))
            index.append(
                {
                    "id": sample_id,
                    "title": "Gettysburg Address",
                    "speaker": f"LibriVox reader {reader}",
                    "source": "S-LV",
                    "license": "public_domain",
                    "level": level,
                    "baseline_id": baseline_id,
                    "report": f"{sample_id}.json",
                    "audio": f"{sample_id}.m4a",
                    "baseline_audio": f"{baseline_id}.m4a",
                    "overall": report.scores.overall,
                    "truth": f"{sample_id}.labels.json",
                }
            )
            print(
                sample_id,
                report.scores.overall,
                len(report.regions),
                "regions",
                len(truth),
                "injected",
            )
    (web / "index.json").write_text(json.dumps(index, indent=1))
    return index


def _m4a(source: Path, target: Path) -> None:
    """AAC copy for the browser; afconvert reads WAV, not FLAC."""
    wav = target.with_suffix(".tmp.wav")
    samples, rate = sf.read(source, dtype="float32")
    sf.write(wav, samples, rate, subtype="PCM_16")
    try:
        subprocess.run(
            ["afconvert", "-f", "m4af", "-d", "aac", "-b", "32000", str(wav), str(target)],
            check=True,
        )
    finally:
        wav.unlink(missing_ok=True)
