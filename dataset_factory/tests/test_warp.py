import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oratio_data.warp import WarpError, WarpMap


def test_identity_leaves_times_unchanged() -> None:
    warp = WarpMap.identity(10.0)

    assert warp.apply([0.0, 3.3, 10.0]).tolist() == [0.0, 3.3, 10.0]
    assert warp.output_duration == 10.0


def test_segment_edits_shift_everything_after_them() -> None:
    warp = WarpMap.from_segments(10.0, [(2.0, 4.0, 0.5), (6.0, 7.0, 3.0)])

    assert warp.span(2.0, 4.0) == (2.0, 3.0)
    assert warp.span(6.0, 7.0) == (5.0, 8.0)
    assert warp.output_duration == pytest.approx(11.0)
    assert warp.apply(3.0) == pytest.approx(2.5)


def test_a_removed_span_collapses_to_a_point() -> None:
    warp = WarpMap.from_segments(5.0, [(1.0, 2.0, 0.0)])

    assert warp.span(1.0, 2.0) == (1.0, 1.0)
    assert warp.span(3.0, 4.0) == (2.0, 3.0)


@pytest.mark.parametrize(
    "edits",
    [[(1.0, 3.0, 1.0), (2.0, 4.0, 1.0)], [(4.0, 6.0, 1.0)], [(2.0, 2.0, 1.0)], [(1.0, 2.0, -1.0)]],
)
def test_invalid_edits_are_rejected(edits: list[tuple[float, float, float]]) -> None:
    with pytest.raises(WarpError):
        WarpMap.from_segments(5.0, edits)


@pytest.mark.parametrize(
    ("source", "output"),
    [((0.0,), (0.0,)), ((1.0, 2.0), (0.0, 1.0)), ((0.0, 1.0, 1.0), (0.0, 1.0, 2.0)),
     ((0.0, 1.0, 2.0), (0.0, 2.0, 1.0)), ((0.0, 1.0), (0.0, 1.0, 2.0))],
)  # fmt: skip
def test_invalid_knots_are_rejected(source: tuple[float, ...], output: tuple[float, ...]) -> None:
    with pytest.raises(WarpError):
        WarpMap(source, output)


def test_composition_requires_matching_durations() -> None:
    with pytest.raises(WarpError, match="output duration"):
        WarpMap.identity(5.0).then(WarpMap.identity(6.0))


_edit = st.tuples(st.integers(0, 9), st.floats(0.0, 3.0))


def _warp(duration: float, picks: list[tuple[int, float]]) -> WarpMap:
    step = duration / 10.0
    unique = {slot: factor for slot, factor in picks}
    return WarpMap.from_segments(
        duration, [(slot * step, (slot + 1) * step, factor) for slot, factor in unique.items()]
    )


@settings(max_examples=150, deadline=None)
@given(first=st.lists(_edit, max_size=5), second=st.lists(_edit, max_size=5))
def test_composition_equals_applying_both_warps_in_turn(
    first: list[tuple[int, float]], second: list[tuple[int, float]]
) -> None:
    a = _warp(10.0, first)
    if a.output_duration == 0.0:
        return
    b = _warp(a.output_duration, second)
    times = np.linspace(0.0, 10.0, 201)

    combined = a.then(b)

    assert np.allclose(combined.apply(times), b.apply(a.apply(times)), atol=1e-9)
    assert combined.output_duration == pytest.approx(b.output_duration)
    assert np.all(np.diff(combined.apply(times)) >= -1e-12)
