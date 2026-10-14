"""Decoding and Whisper transcription with word timings (Colab, GPU)."""

import ctypes
import importlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from oratio_data.baselines import Word
from oratio_data.dsp import SAMPLE_RATE, Samples

WHISPER_MODEL = "large-v3-turbo"
LID_SECONDS = 30.0
BATCH_SIZE = 16


def decode(path: Path, start: float = 0.0, seconds: float | None = None) -> Samples:
    """Any audio file to 16 kHz mono float32 through ffmpeg."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}"]
    if seconds is not None:
        command += ["-t", f"{seconds:.3f}"]
    command += ["-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout  # noqa: S603
    return np.frombuffer(raw, dtype=np.float32).copy()


CUDA_PACKAGES = ("nvidia.cublas.lib", "nvidia.cudnn.lib")
CHECK_SECONDS = 1.0


def _preload_cuda_libraries() -> None:
    """Load the pip-installed CUDA 12 libraries CTranslate2 links against.

    Colab's system CUDA can be newer than the one CTranslate2 was built for, and the loader
    path cannot change after the process starts, so the libraries are loaded globally here.
    """
    pending: list[Path] = []
    for name in CUDA_PACKAGES:
        try:
            package = importlib.import_module(name)
        except ImportError:
            continue
        for directory in package.__path__:
            pending.extend(sorted(Path(directory).glob("lib*.so*")))
    for _ in range(len(pending)):
        failed = []
        for library in pending:
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(library)
        if len(failed) in (0, len(pending)):
            return
        pending = failed


@lru_cache(maxsize=1)
def _models() -> tuple[Any, Any]:
    _preload_cuda_libraries()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    return model, BatchedInferencePipeline(model=model)


def language(audio: Samples) -> tuple[str, float]:
    model, _ = _models()
    _, info = model.transcribe(audio[: round(LID_SECONDS * SAMPLE_RATE)], beam_size=1)
    return str(info.language), float(info.language_probability)


def transcribe(audio: Samples) -> list[Word]:
    _, pipeline = _models()
    segments, _ = pipeline.transcribe(
        audio, language="en", batch_size=BATCH_SIZE, word_timestamps=True
    )
    return [
        Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
        for segment in segments
        for w in segment.words or []
    ]


def check() -> None:
    """Load the model and transcribe a second of silence; raises if the GPU stack is broken."""
    transcribe(np.zeros(round(CHECK_SECONDS * SAMPLE_RATE), dtype=np.float32))
