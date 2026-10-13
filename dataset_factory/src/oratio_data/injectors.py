"""Compile flaw codes into signal edits and apply a plan with exact output-time labels."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from oratio_data import dsp
from oratio_data.dsp import SLIVER_S, DurationEdit, PitchEdit, Samples, Span
from oratio_data.warp import WarpError, WarpMap

DIMENSIONS = {
    "PAC": "pacing",
    "PAU": "pausing",
    "INT": "intonation",
    "ENE": "vocal_energy",
    "EMP": "emphasis",
    "FLU": "fluency",
    "CLA": "clarity",
    "VQ": "clarity",
}
REPEAT_GAP_S = 0.08
FALSE_START_SHARE = 0.45
FALSE_START_GAP_S = 0.15
MAX_ADDED_PAUSE_S = 1.2
MUFFLE_OPEN_HZ = 6000.0
MUFFLE_CLOSED_HZ = 1200.0
FRY_RATE_HZ = 50.0
FRY_DROP_ST = 14.0


class PlanError(ValueError):
    """The plan asks for something the injectors cannot do."""


@dataclass(frozen=True, slots=True)
class Flaw:
    """One flaw to inject: a taxonomy code over a source-time span at magnitude 0 to 1."""

    code: str
    start: float
    end: float
    magnitude: float
    word_spans: tuple[Span, ...] = ()


@dataclass(frozen=True, slots=True)
class Label:
    code: str
    dimension: str
    magnitude: float
    start: float
    end: float
    source_start: float
    source_end: float


@dataclass(frozen=True, slots=True)
class Result:
    samples: Samples
    warp: WarpMap
    labels: tuple[Label, ...]


TimeEdit = Callable[[Samples, Span], Samples]
SpliceEdit = Callable[[Samples, Span, np.random.Generator, float], tuple[Samples, WarpMap]]


@dataclass(slots=True)
class _Recipe:
    durations: list[DurationEdit] = field(default_factory=list)
    pitches: list[PitchEdit] = field(default_factory=list)
    time_edits: list[TimeEdit] = field(default_factory=list)
    splices: list[SpliceEdit] = field(default_factory=list)
    inserts_before: bool = False
    inserts_after: bool = False


def _middle(span: Span, share: float) -> Span:
    half = (span[1] - span[0]) * share / 2.0
    centre = (span[0] + span[1]) / 2.0
    return centre - half, centre + half


def _add_pause(m: float) -> SpliceEdit:
    def edit(
        x: Samples, span: Span, rng: np.random.Generator, total: float
    ) -> tuple[Samples, WarpMap]:
        return dsp.insert(
            x, span[1], dsp.room_tone(x, m * MAX_ADDED_PAUSE_S, rng), before=False, total=total
        )

    return edit


def _repeat(share: float, gap_s: float, before: bool) -> SpliceEdit:
    def edit(
        x: Samples, span: Span, rng: np.random.Generator, total: float
    ) -> tuple[Samples, WarpMap]:
        end = span[0] + (span[1] - span[0]) * share
        copy = x[round(span[0] * dsp.SAMPLE_RATE) : round(end * dsp.SAMPLE_RATE)]
        tone = dsp.room_tone(x, gap_s, rng)
        payload = np.concatenate((copy, tone)) if before else np.concatenate((tone, copy))
        return dsp.insert(
            x, span[0] if before else span[1], payload.astype(np.float32), before, total
        )

    return edit


def _filler(filler: Samples) -> SpliceEdit:
    def edit(
        x: Samples, span: Span, rng: np.random.Generator, total: float
    ) -> tuple[Samples, WarpMap]:
        payload = np.concatenate((filler, dsp.room_tone(x, REPEAT_GAP_S, rng)))
        return dsp.insert(x, span[0], payload.astype(np.float32), before=True, total=total)

    return edit


def _cut(share: float) -> SpliceEdit:
    return lambda x, span, rng, total: dsp.cut(x, _middle(span, share), total)


def _compile(flaw: Flaw, filler: Samples | None) -> _Recipe:
    m, span = flaw.magnitude, (flaw.start, flaw.end)
    recipe = _Recipe()
    match flaw.code:
        case "PAC-RUSH":
            recipe.durations.append(DurationEdit(span, 1.0 - 0.5 * m))
        case "PAC-DRAG":
            recipe.durations.append(DurationEdit(span, 1.0 + m))
        case "PAC-ERR":
            if len(flaw.word_spans) < 2:
                raise PlanError("PAC-ERR needs the word spans it alternates over")
            for index, word in enumerate(flaw.word_spans):
                recipe.durations.append(
                    DurationEdit(word, float(np.exp((0.7 if index % 2 else -0.7) * m)))
                )
        case "PAU-DROP":
            recipe.splices.append(_cut(0.9 * m))
        case "PAU-LONG" | "PAU-MID":
            recipe.splices.append(_add_pause(m))
            recipe.inserts_after = True
        case "INT-MONO":
            recipe.pitches.append(PitchEdit(span, excursion=1.0 - m))
        case "INT-EXAG":
            recipe.pitches.append(PitchEdit(span, excursion=1.0 + 1.5 * m))
        case "INT-UP":
            tail = (span[1] - 0.4 * (span[1] - span[0]), span[1])
            recipe.pitches.append(PitchEdit(tail, shift_end_st=6.0 * m))
        case "ENE-LOW":
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, -15.0 * m, -15.0 * m))
        case "ENE-SPIKE":
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, 9.0 * m, 9.0 * m))
        case "ENE-TRAIL":
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, 0.0, -18.0 * m))
        case "ENE-FLAT":
            recipe.time_edits.append(lambda x, s: dsp.compress(x, s, m))
        case "EMP-DROP":
            recipe.pitches.append(PitchEdit(span, excursion=1.0 - m))
            recipe.durations.append(DurationEdit(span, 1.0 - 0.25 * m))
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, -6.0 * m, -6.0 * m))
        case "EMP-MISP":
            recipe.pitches.append(PitchEdit(span, shift_start_st=4.0 * m, shift_end_st=4.0 * m))
            recipe.durations.append(DurationEdit(span, 1.0 + 0.4 * m))
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, 6.0 * m, 6.0 * m))
        case "FLU-REP":
            recipe.splices.append(_repeat(1.0, REPEAT_GAP_S, before=False))
            recipe.inserts_after = True
        case "FLU-FALSE":
            recipe.splices.append(_repeat(FALSE_START_SHARE, FALSE_START_GAP_S, before=True))
            recipe.inserts_before = True
        case "FLU-FILL":
            if filler is None:
                raise PlanError("FLU-FILL needs filler audio from the same voice")
            recipe.splices.append(_filler(filler))
            recipe.inserts_before = True
        case "FLU-PROL":
            recipe.durations.append(DurationEdit(_middle(span, 0.5), 1.0 + 3.0 * m))
        case "FLU-OMIT":
            recipe.splices.append(_cut(1.0))
        case "CLA-MUFF":
            cutoff = MUFFLE_OPEN_HZ * (MUFFLE_CLOSED_HZ / MUFFLE_OPEN_HZ) ** m
            recipe.time_edits.append(lambda x, s: dsp.lowpass(x, s, cutoff))
        case "CLA-SWAL":
            recipe.time_edits.append(lambda x, s: dsp.gain(x, s, -6.0 * m, -6.0 * m))
            recipe.splices.append(_cut(0.45 * m))
        case "VQ-FRY":
            recipe.pitches.append(
                PitchEdit(span, shift_start_st=-FRY_DROP_ST * m, shift_end_st=-FRY_DROP_ST * m)
            )
            recipe.time_edits.append(lambda x, s: dsp.tremor(x, s, FRY_RATE_HZ, 0.6 * m))
        case _:
            raise PlanError(f"no injector for {flaw.code!r}")
    return recipe


def _validate(flaws: list[Flaw], duration: float) -> None:
    for flaw in flaws:
        if not 0.0 <= flaw.magnitude <= 1.0:
            raise PlanError(f"{flaw.code}: magnitude {flaw.magnitude} is outside 0 to 1")
        if not 0.0 <= flaw.start < flaw.end <= duration:
            raise PlanError(f"{flaw.code}: span {flaw.start}-{flaw.end} is outside the clip")


def apply_plan(
    samples: Samples, flaws: list[Flaw], seed: int, filler: Samples | None = None
) -> Result:
    """Inject every flaw and return the audio, the source-to-output warp and the labels.

    An empty plan still passes through resynthesis, which makes it the processing-matched control.
    """
    _validate(flaws, len(samples) / dsp.SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    recipes = [_compile(flaw, filler) for flaw in flaws]
    try:
        audio, warp = dsp.resynthesize(
            samples,
            [edit for recipe in recipes for edit in recipe.durations],
            [edit for recipe in recipes for edit in recipe.pitches],
        )
    except WarpError as error:
        raise PlanError(f"duration edits conflict: {error}") from error

    for flaw, recipe in zip(flaws, recipes, strict=True):
        for time_edit in recipe.time_edits:
            audio = time_edit(audio, warp.span(flaw.start, flaw.end))
    audio = dsp.limit_peak(audio)

    for flaw, recipe in zip(flaws, recipes, strict=True):
        for splice in recipe.splices:
            audio, step = splice(audio, warp.span(flaw.start, flaw.end), rng, warp.output_duration)
            warp = warp.then(step)

    labels = []
    for flaw, recipe in zip(flaws, recipes, strict=True):
        lead = SLIVER_S if recipe.inserts_before else 0.0
        trail = SLIVER_S if recipe.inserts_after else 0.0
        start, end = warp.span(max(0.0, flaw.start - lead), flaw.end + trail)
        labels.append(
            Label(
                code=flaw.code,
                dimension=DIMENSIONS[flaw.code.split("-")[0]],
                magnitude=flaw.magnitude,
                start=start,
                end=end,
                source_start=flaw.start,
                source_end=flaw.end,
            )
        )
    return Result(samples=audio, warp=warp, labels=tuple(labels))
