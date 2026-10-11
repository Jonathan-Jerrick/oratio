import math

import pytest

from speechgrade.hashing import canonical_json, content_hash


def test_hash_ignores_key_order() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})


def test_hash_ignores_float_noise_below_precision() -> None:
    assert content_hash({"z": 0.1 + 0.2}) == content_hash({"z": 0.3})
    assert content_hash({"z": -0.0}) == content_hash({"z": 0.0})


def test_hash_changes_with_content() -> None:
    assert content_hash({"z": 0.3}) != content_hash({"z": 0.300002})
    assert content_hash([1, 2]) != content_hash([2, 1])


def test_canonical_json_is_compact_utf8() -> None:
    assert canonical_json({"b": "é", "a": (1, 2.5)}) == '{"a":[1,2.5],"b":"é"}'.encode()


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_numbers_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json({"z": bad})


def test_unsupported_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="set"):
        canonical_json({"z": {1, 2}})
