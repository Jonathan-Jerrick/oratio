"""Decode and validate uploaded audio."""

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import soundfile as sf

SAMPLE_RATE = 16_000
MAX_BYTES = 5_500_000
MAX_SECONDS = 170.0
MIN_SECONDS = 1.0

FloatArray = npt.NDArray[np.float32]


class AudioError(ValueError):
    """The upload is not audio this service accepts."""


@dataclass(frozen=True, slots=True)
class Audio:
    samples: FloatArray
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def decode_wav(data: bytes) -> Audio:
    """Parse a 16 kHz WAV upload into mono float32, enforcing the size and length limits."""
    if len(data) > MAX_BYTES:
        raise AudioError(f"file is {len(data)} bytes; the limit is {MAX_BYTES}")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a WAV file")
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as error:
        raise AudioError("WAV data could not be decoded") from error
    if sample_rate != SAMPLE_RATE:
        raise AudioError(f"sample rate is {sample_rate} Hz; expected {SAMPLE_RATE}")
    mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
    if not np.isfinite(mono).all():
        raise AudioError("audio contains non-finite samples")
    seconds = len(mono) / SAMPLE_RATE
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise AudioError(
            f"audio is {seconds:.1f} s; allowed {MIN_SECONDS:.0f} to {MAX_SECONDS:.0f} s"
        )
    return Audio(samples=mono, sample_rate=SAMPLE_RATE)
