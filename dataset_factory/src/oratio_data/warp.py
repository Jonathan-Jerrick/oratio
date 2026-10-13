"""Monotone piecewise-linear time maps that carry labels through duration edits."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

_TOLERANCE = 1e-9


class WarpError(ValueError):
    """The knots do not describe a valid time map."""


@dataclass(frozen=True, slots=True)
class WarpMap:
    """Maps source time to output time through (source, output) knots."""

    source: tuple[float, ...]
    output: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.source) != len(self.output) or len(self.source) < 2:
            raise WarpError("a warp needs at least two matching knots")
        if self.source[0] != 0.0 or self.output[0] != 0.0:
            raise WarpError("a warp starts at time zero")
        if np.any(np.diff(self.source) <= 0):
            raise WarpError("source knots must strictly increase")
        if np.any(np.diff(self.output) < 0):
            raise WarpError("output knots must not decrease")

    @classmethod
    def identity(cls, duration: float) -> Self:
        return cls((0.0, duration), (0.0, duration))

    @classmethod
    def from_segments(cls, duration: float, edits: list[tuple[float, float, float]]) -> Self:
        """Build from (start, end, factor) edits; factor scales that span's duration."""
        source, output = [0.0], [0.0]
        for start, end, factor in sorted(edits):
            if start < source[-1] - _TOLERANCE or end > duration + _TOLERANCE or end <= start:
                raise WarpError(f"edit {start}-{end} overlaps another or leaves the clip")
            if factor < 0:
                raise WarpError("a duration factor cannot be negative")
            if start > source[-1] + _TOLERANCE:
                output.append(output[-1] + start - source[-1])
                source.append(start)
            output.append(output[-1] + (end - start) * factor)
            source.append(end)
        if duration > source[-1] + _TOLERANCE:
            output.append(output[-1] + duration - source[-1])
            source.append(duration)
        return cls(tuple(source), tuple(output))

    @property
    def source_duration(self) -> float:
        return self.source[-1]

    @property
    def output_duration(self) -> float:
        return self.output[-1]

    def apply(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Output time of each source time; times outside the clip are clamped to it."""
        return np.interp(np.asarray(times, dtype=np.float64), self.source, self.output)

    def span(self, start: float, end: float) -> tuple[float, float]:
        mapped = self.apply([start, end])
        return float(mapped[0]), float(mapped[1])

    def _sources_of(self, outputs: tuple[float, ...]) -> list[float]:
        """Source times that land on the given output times, skipping collapsed spans."""
        source, output = np.asarray(self.source), np.asarray(self.output)
        found: list[float] = []
        for value in outputs:
            for side in ("left", "right"):
                upper = int(np.clip(np.searchsorted(output, value, side=side), 1, len(output) - 1))
                rise = output[upper] - output[upper - 1]
                if rise > _TOLERANCE and output[upper - 1] <= value <= output[upper]:
                    share = (value - output[upper - 1]) / rise
                    found.append(
                        float(source[upper - 1] + share * (source[upper] - source[upper - 1]))
                    )
        return found

    def then(self, later: Self) -> Self:
        """The map equal to applying this warp and then a later one to its output."""
        if abs(later.source_duration - self.output_duration) > 1e-6:
            raise WarpError("the later warp must start from this warp's output duration")
        knots = np.unique(np.concatenate((self.source, self._sources_of(later.source))))
        final = later.apply(self.apply(knots))
        return type(self)(tuple(float(k) for k in knots), tuple(float(v) for v in final))
