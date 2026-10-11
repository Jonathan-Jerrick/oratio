"""Time each planned serving stage on synthetic speech-like audio."""

import os
import platform
import resource
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

SAMPLE_RATE = 16_000
PITCH_STEP_S = 0.01
MIN_WINDOW_SAMPLES = 400
LGBM_WORDS = 200
LGBM_FEATURES = 60
LGBM_ROUNDS = 200

Variant = Literal["fp32", "int8"]
FloatArray = npt.NDArray[np.float32]


class BenchResult(BaseModel):
    seconds: float
    variant: Variant
    chunk_seconds: float | None
    arch: str
    cpu_count: int
    threads: int
    load_ms: dict[str, float]
    stage_ms: dict[str, float]
    total_ms: float
    max_rss_mb: float


def synth_speech(seconds: float, seed: int = 0) -> FloatArray:
    """Harmonic source with a moving F0 and syllable-rate envelope, so pitch tracking has work."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 140.0 + 40.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voiced = sum(np.sin(k * phase) / k for k in range(1, 9))
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    signal = voiced * envelope + 0.01 * rng.standard_normal(n)
    return np.asarray(0.5 * signal / np.max(np.abs(signal)), dtype=np.float32)


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


@lru_cache(maxsize=4)
def _session(path: str, threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _booster() -> Any:
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    features = rng.standard_normal((2_000, LGBM_FEATURES))
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.int32)
    params = {"objective": "binary", "verbose": -1, "num_threads": 1, "seed": 0}
    return lgb.train(params, lgb.Dataset(features, labels), num_boost_round=LGBM_ROUNDS)


def _encode(session: Any, audio: FloatArray, chunk_seconds: float | None) -> int:
    """Run the encoder over the clip, whole or in fixed windows; returns the frame count."""
    name = session.get_inputs()[0].name
    step = len(audio) if chunk_seconds is None else int(chunk_seconds * SAMPLE_RATE)
    frames = 0
    for start in range(0, len(audio), step):
        window = audio[start : start + step]
        if len(window) < MIN_WINDOW_SAMPLES:
            break
        frames += session.run(None, {name: window[np.newaxis, :]})[0].shape[1]
    return frames


def _prosody(audio: FloatArray) -> tuple[Any, Any]:
    import parselmouth

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    return sound.to_pitch(time_step=PITCH_STEP_S), sound.to_intensity(time_step=PITCH_STEP_S)


def _max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0**2 if sys.platform == "darwin" else 1024.0)


def run_bench(
    model_dir: Path, seconds: float, variant: Variant, chunk_seconds: float | None = None
) -> BenchResult:
    threads = os.cpu_count() or 1
    audio = synth_speech(seconds)
    word_features = np.random.default_rng(1).standard_normal((LGBM_WORDS, LGBM_FEATURES))

    ctc, ctc_load = _timed(
        lambda: _session(str(model_dir / f"wav2vec2_base_{variant}.onnx"), threads)
    )
    ssl, ssl_load = _timed(
        lambda: _session(str(model_dir / f"wavlm_base_plus_{variant}.onnx"), threads)
    )
    booster, lgbm_load = _timed(_booster)

    _, ctc_ms = _timed(lambda: _encode(ctc, audio, chunk_seconds))
    _, ssl_ms = _timed(lambda: _encode(ssl, audio, chunk_seconds))
    _, prosody_ms = _timed(lambda: _prosody(audio))
    _, shap_ms = _timed(lambda: booster.predict(word_features, pred_contrib=True))

    stage_ms = {
        "encoder_ctc": ctc_ms,
        "encoder_ssl": ssl_ms,
        "prosody": prosody_ms,
        "lgbm_shap": shap_ms,
    }
    return BenchResult(
        seconds=seconds,
        variant=variant,
        chunk_seconds=chunk_seconds,
        arch=platform.machine(),
        cpu_count=threads,
        threads=threads,
        load_ms={"encoder_ctc": ctc_load, "encoder_ssl": ssl_load, "lgbm": lgbm_load},
        stage_ms=stage_ms,
        total_ms=sum(stage_ms.values()),
        max_rss_mb=_max_rss_mb(),
    )
