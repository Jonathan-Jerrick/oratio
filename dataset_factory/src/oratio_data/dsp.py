"""Signal edits used to inject delivery flaws: PSOLA resynthesis, gain, filtering, splicing."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, sosfiltfilt

from oratio_data.warp import WarpMap

SAMPLE_RATE = 16_000
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 500.0
ANALYSIS_STEP_S = 0.01
TIER_RAMP_S = 0.001
EDGE_S = 0.015
CROSSFADE_S = 0.012
SLIVER_S = 0.01
SEAM_S = 0.004
ROOM_TONE_WINDOW_S = 0.15
ROOM_TONE_FLOOR = 10.0 ** (-70.0 / 20.0)
COMPRESS_FRAME_S = 0.02
PEAK_LIMIT = 0.98
FILTER_ORDER = 4
_EPS = 1e-10

Samples = npt.NDArray[np.float32]
Span = tuple[float, float]


class DspError(ValueError):
    """An edit cannot be applied to this audio."""


@dataclass(frozen=True, slots=True)
class DurationEdit:
    span: Span
    factor: float


@dataclass(frozen=True, slots=True)
class PitchEdit:
    """Scale pitch excursions around the span median, then shift by a semitone ramp."""

    span: Span
    excursion: float = 1.0
    shift_start_st: float = 0.0
    shift_end_st: float = 0.0


def _index(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def _median_f0(sound: parselmouth.Sound, span: Span) -> float | None:
    pitch = sound.to_pitch(
        time_step=ANALYSIS_STEP_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    times, values = pitch.xs(), pitch.selected_array["frequency"]
    voiced = values[(times >= span[0]) & (times <= span[1]) & (values > 0)]
    return float(np.median(voiced)) if len(voiced) else None


def resynthesize(
    samples: Samples, durations: list[DurationEdit], pitches: list[PitchEdit]
) -> tuple[Samples, WarpMap]:
    """One PSOLA pass applying every duration and pitch edit; with no edits it is the control."""
    total = len(samples) / SAMPLE_RATE
    warp = WarpMap.from_segments(total, [(*edit.span, edit.factor) for edit in durations])
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    manipulation = call(sound, "To Manipulation", ANALYSIS_STEP_S, PITCH_FLOOR_HZ, PITCH_CEILING_HZ)

    if pitches:
        tier = call(manipulation, "Extract pitch tier")
        for edit in pitches:
            median = _median_f0(sound, edit.span)
            if median is None:
                continue
            start, end = edit.span
            slope = (edit.shift_end_st - edit.shift_start_st) / (end - start)
            ramp = f"({edit.shift_start_st} + {slope} * (x - {start}))"
            formula = (
                f"if x >= {start} and x <= {end} then "
                f"{median} * (self / {median}) ^ {edit.excursion} * 2 ^ ({ramp} / 12) else self fi"
            )
            call(tier, "Formula", formula)
        call([manipulation, tier], "Replace pitch tier")

    if durations:
        tier = call("Create DurationTier", "duration", 0.0, total)
        for stretch in sorted(durations, key=lambda item: item.span):
            start, end = stretch.span
            call(tier, "Add point", max(0.0, start - TIER_RAMP_S), 1.0)
            call(tier, "Add point", start, stretch.factor)
            call(tier, "Add point", end, stretch.factor)
            call(tier, "Add point", min(total, end + TIER_RAMP_S), 1.0)
        call([manipulation, tier], "Replace duration tier")

    output = call(manipulation, "Get resynthesis (overlap-add)")
    audio = np.asarray(output.values[0], dtype=np.float32)
    return fit_length(audio, warp.output_duration), warp


def _window(length: int, span: Span) -> npt.NDArray[np.float64]:
    """1 inside the span, 0 outside, with raised-cosine edges centred on the boundaries."""
    start, end = _index(span[0]), _index(span[1])
    edge = max(1, min(_index(EDGE_S), (end - start) // 2))
    position = np.arange(length)
    rise = np.clip((position - (start - edge / 2)) / edge, 0.0, 1.0)
    fall = np.clip(((end + edge / 2) - position) / edge, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall))


def gain(samples: Samples, span: Span, start_db: float, end_db: float) -> Samples:
    """Apply a gain that moves linearly in dB from start_db to end_db across the span."""
    start, end = _index(span[0]), _index(span[1])
    progress = np.clip((np.arange(len(samples)) - start) / max(1, end - start), 0.0, 1.0)
    curve_db = (start_db + (end_db - start_db) * progress) * _window(len(samples), span)
    return np.asarray(samples * 10.0 ** (curve_db / 20.0), dtype=np.float32)


def lowpass(samples: Samples, span: Span, cutoff_hz: float) -> Samples:
    sos = butter(FILTER_ORDER, cutoff_hz, btype="low", fs=SAMPLE_RATE, output="sos")
    mix = _window(len(samples), span)
    filtered = sosfiltfilt(sos, samples.astype(np.float64))
    return np.asarray(samples * (1.0 - mix) + filtered * mix, dtype=np.float32)


def compress(samples: Samples, span: Span, amount: float) -> Samples:
    """Pull frame levels inside the span toward their median; amount 1 removes all contrast."""
    frame = _index(COMPRESS_FRAME_S)
    start, end = _index(span[0]), _index(span[1])
    count = (end - start) // frame
    if count < 2:
        return samples.copy()
    blocks = samples[start : start + count * frame].astype(np.float64).reshape(count, frame)
    level_db = 20.0 * np.log10(np.sqrt(np.mean(blocks**2, axis=1)) + _EPS)
    audible = level_db > level_db.max() - 40.0
    change_db = np.where(audible, -amount * (level_db - np.median(level_db[audible])), 0.0)
    centres = start + (np.arange(count) + 0.5) * frame
    curve_db = np.interp(np.arange(len(samples)), centres, change_db) * _window(len(samples), span)
    return np.asarray(samples * 10.0 ** (curve_db / 20.0), dtype=np.float32)


def tremor(samples: Samples, span: Span, rate_hz: float, depth: float) -> Samples:
    """Pulse the amplitude at a low rate, as in creaky voice."""
    phase = 2.0 * np.pi * rate_hz * np.arange(len(samples)) / SAMPLE_RATE
    modulation = 1.0 - depth * (0.5 + 0.5 * np.sin(phase)) * _window(len(samples), span)
    return np.asarray(samples * modulation, dtype=np.float32)


def limit_peak(samples: Samples) -> Samples:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return samples if peak <= PEAK_LIMIT else np.asarray(samples * (PEAK_LIMIT / peak), np.float32)


def _join(left: Samples, right: Samples) -> Samples:
    """Concatenate with an equal-power crossfade that overlaps the two pieces."""
    overlap = min(_index(CROSSFADE_S), len(left), len(right))
    if overlap == 0:
        return np.concatenate((left, right))
    angle = np.linspace(0.0, np.pi / 2.0, overlap)
    blended = left[-overlap:] * np.cos(angle) + right[:overlap] * np.sin(angle)
    return np.concatenate((left[:-overlap], blended.astype(np.float32), right[overlap:]))


def room_tone(samples: Samples, seconds: float, rng: np.random.Generator) -> Samples:
    """Background noise of the requested length, taken from the quietest stretch of the clip."""
    window = _index(ROOM_TONE_WINDOW_S)
    need = _index(seconds)
    if len(samples) < 2 * window:
        raise DspError("clip is too short to sample room tone from")
    power = np.convolve(samples.astype(np.float64) ** 2, np.ones(window), mode="valid")
    start = int(np.argmin(power))
    quiet = samples[start : start + window]
    tone = quiet
    while len(tone) < need + window:
        tone = _join(tone, quiet[::-1] if rng.random() < 0.5 else quiet)
    tone = tone[:need] + ROOM_TONE_FLOOR * rng.standard_normal(need).astype(np.float32)
    return np.asarray(tone, dtype=np.float32)


def _splice_warp(total: float, added_samples: int, sliver: Span) -> WarpMap:
    factor = max(0.0, 1.0 + added_samples / SAMPLE_RATE / (sliver[1] - sliver[0]))
    return WarpMap.from_segments(total, [(sliver[0], min(sliver[1], total), factor)])


def fit_length(samples: Samples, seconds: float) -> Samples:
    """Trim or zero-extend by a few samples so the audio is exactly as long as its warp says."""
    target = _index(seconds)
    if len(samples) >= target:
        return samples[:target]
    return np.concatenate((samples, np.zeros(target - len(samples), dtype=np.float32)))


def cut(samples: Samples, span: Span, total: float) -> tuple[Samples, WarpMap]:
    """Remove the span; the warp collapses it to a point."""
    start, end = _index(span[0]), _index(span[1])
    if end <= start:
        return samples, WarpMap.identity(total)
    overlap = min(_index(CROSSFADE_S), start, end - start)
    output = _join(samples[:start], samples[end - overlap :])
    exact = (start / SAMPLE_RATE, end / SAMPLE_RATE)
    return output, _splice_warp(total, len(output) - len(samples), exact)


def _faded(piece: Samples, seconds: float, head: bool, tail: bool) -> Samples:
    length = min(_index(seconds), len(piece) // 2)
    out = piece.astype(np.float32, copy=True)
    if length:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, length))).astype(np.float32)
        if head:
            out[:length] *= ramp
        if tail:
            out[-length:] *= ramp[::-1]
    return out


def insert(
    samples: Samples, at: float, payload: Samples, before: bool, total: float
) -> tuple[Samples, WarpMap]:
    """Insert audio at a time.

    With before=True the insert belongs to what follows it: audio from `at` onward shifts and
    the sliver just before `at` stretches. With before=False the sliver after `at` stretches.
    """
    if len(payload) == 0:
        return samples, WarpMap.identity(total)
    position = _index(at)
    output = np.concatenate(
        (
            _faded(samples[:position], SEAM_S, head=False, tail=True),
            _faded(payload, CROSSFADE_S, head=True, tail=True),
            _faded(samples[position:], SEAM_S, head=True, tail=False),
        )
    )
    exact = position / SAMPLE_RATE
    use_before = (before and exact >= SLIVER_S) or exact + SLIVER_S > total
    start = exact - SLIVER_S if use_before else exact
    return output, _splice_warp(total, len(payload), (start, start + SLIVER_S))
