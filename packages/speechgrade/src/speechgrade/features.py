"""Frame-level acoustic tracks and speaker normalisation."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import parselmouth
from scipy.fft import dct

from speechgrade.audio import SAMPLE_RATE

HOP_SECONDS = 0.01
HOP = int(HOP_SECONDS * SAMPLE_RATE)
WINDOW = 400
N_FFT = 512
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 500.0
HF_CUTOFF_HZ = 2000.0
N_MELS = 40
N_MFCC = 13
SPEECH_FLOOR_DB = 35.0
_EPS = 1e-10

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Tracks:
    """Acoustic measurements on a common 10 ms grid; f0_hz is 0 where unvoiced."""

    f0_hz: FloatArray
    energy_db: FloatArray
    hf_ratio: FloatArray
    cpp_db: FloatArray
    mfcc: FloatArray

    def __len__(self) -> int:
        return len(self.energy_db)


def _frames(samples: FloatArray) -> npt.NDArray[np.float64]:
    padded = np.pad(samples.astype(np.float64), WINDOW // 2, mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP]


def _pitch(samples: FloatArray, count: int) -> FloatArray:
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    pitch = sound.to_pitch(
        time_step=HOP_SECONDS, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    values = pitch.selected_array["frequency"]
    index = np.rint((np.arange(count) * HOP_SECONDS - pitch.xs()[0]) / pitch.dx).astype(np.intp)
    inside = (index >= 0) & (index < len(values))
    out = np.zeros(count, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def _mel_filterbank() -> npt.NDArray[np.float64]:
    def to_mel(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    edges_mel = np.linspace(0.0, to_mel(np.array([SAMPLE_RATE / 2.0]))[0], N_MELS + 2)
    edges_hz = 700.0 * (10.0 ** (edges_mel / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    rising = (bins[None, :] - edges_hz[:-2, None]) / np.diff(edges_hz)[:-1, None]
    falling = (edges_hz[2:, None] - bins[None, :]) / np.diff(edges_hz)[1:, None]
    return np.maximum(0.0, np.minimum(rising, falling))


def _cepstral_peak_prominence(power: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Height of the cepstral peak in the pitch range above its regression line, in dB."""
    cepstrum = 10.0 * np.log10(np.fft.irfft(np.log(power + _EPS), axis=1) ** 2 + _EPS)
    low, high = int(SAMPLE_RATE / PITCH_CEILING_HZ), int(SAMPLE_RATE / PITCH_FLOOR_HZ)
    quefrency = np.arange(low, min(high, N_FFT // 2))
    band = cepstrum[:, quefrency]
    centred = quefrency - quefrency.mean()
    slope = (band - band.mean(axis=1, keepdims=True)) @ centred / np.sum(centred**2)
    peak = np.argmax(band, axis=1)
    rows = np.arange(len(band))
    trend = band.mean(axis=1) + slope * centred[peak]
    return np.asarray(band[rows, peak] - trend, dtype=np.float64)


def extract_tracks(samples: FloatArray) -> Tracks:
    frames = _frames(samples)
    energy_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + _EPS)
    power = np.abs(np.fft.rfft(frames * np.hanning(WINDOW), n=N_FFT, axis=1)) ** 2
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    hf_ratio = power[:, bins >= HF_CUTOFF_HZ].sum(axis=1) / (power.sum(axis=1) + _EPS)
    log_mel = np.log(power @ _mel_filterbank().T + _EPS)
    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, :N_MFCC]
    return Tracks(
        f0_hz=_pitch(samples, len(frames)),
        energy_db=energy_db.astype(np.float32),
        hf_ratio=hf_ratio.astype(np.float32),
        cpp_db=_cepstral_peak_prominence(power).astype(np.float32),
        mfcc=mfcc.astype(np.float32),
    )


def speech_mask(energy_db: FloatArray) -> BoolArray:
    """Frames within SPEECH_FLOOR_DB of the speaker's loud level."""
    return np.asarray(energy_db > np.percentile(energy_db, 95) - SPEECH_FLOOR_DB)


def semitones(f0_hz: FloatArray) -> FloatArray:
    """Pitch in semitones around the speaker's own median; NaN where unvoiced."""
    voiced = f0_hz > 0
    out = np.full(len(f0_hz), np.nan, dtype=np.float32)
    if voiced.any():
        out[voiced] = 12.0 * np.log2(f0_hz[voiced] / np.median(f0_hz[voiced]))
    return out


def relative_energy(energy_db: FloatArray) -> FloatArray:
    """Loudness in dB around the speaker's own median speech level."""
    mask = speech_mask(energy_db)
    reference = np.median(energy_db[mask]) if mask.any() else 0.0
    return np.asarray(energy_db - reference, dtype=np.float32)
