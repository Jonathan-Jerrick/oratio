"""Deterministic rubric scoring: every lost point traces to a region."""

import math

from speechgrade.rubric import Rubric, Scoring
from speechgrade.schema import Region, Scores

MAX_DIMENSION_SCORE = 10.0


def deduction(
    z: float, z_threshold: float, weight: float, seconds: float, scoring: Scoring
) -> float:
    """Points lost for one region: excess deviation, capped, scaled by how long it lasts."""
    excess = min(max(abs(z) - z_threshold, 0.0), scoring.z_cap)
    duration_weight = min(1.0, math.sqrt(max(seconds, 0.0) / scoring.duration_reference_s))
    return weight * excess * max(duration_weight, scoring.duration_weight_floor)


def score_regions(regions: list[Region], rubric: Rubric) -> Scores:
    lost = dict.fromkeys(rubric.dimensions, 0.0)
    for region in regions:
        lost[region.dimension] += region.deduction
    dimensions = {
        name: round(min(MAX_DIMENSION_SCORE, max(0.0, MAX_DIMENSION_SCORE - points)), 4)
        for name, points in lost.items()
    }
    overall = round(10.0 * sum(dimensions.values()) / len(dimensions), 4)
    return Scores(dimensions=dimensions, overall=overall)
