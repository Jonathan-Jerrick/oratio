"""Automated signal gates every generated clip must pass."""

import numpy as np

from oratio_data.config import Gates
from oratio_data.dsp import SAMPLE_RATE, Samples
from oratio_data.injectors import Result


def _longest_zero_run(samples: Samples) -> int:
    zero = np.concatenate(([False], samples == 0.0, [False]))
    edges = np.flatnonzero(np.diff(zero.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def longest_zero_run_s(samples: Samples) -> float:
    return _longest_zero_run(samples) / SAMPLE_RATE


def check_clip(result: Result, gates: Gates, source_silence_s: float = 0.0) -> list[str]:
    """Names of the gates this clip fails; empty when it passes.

    Digital silence already present in the source recording is not held against the clip.
    """
    samples = result.samples
    failures: list[str] = []
    if np.mean(np.abs(samples) >= gates.clip_level) > gates.max_clipped_share:
        failures.append("clipping")
    if longest_zero_run_s(samples) > max(gates.max_digital_silence_s, source_silence_s + 0.001):
        failures.append("digital_silence")
    seconds = len(samples) / SAMPLE_RATE
    if abs(seconds - result.warp.output_duration) > gates.duration_tolerance_s:
        failures.append("duration")
    if any(not 0.0 <= label.start <= label.end <= seconds + 1e-6 for label in result.labels):
        failures.append("label_bounds")
    return failures
