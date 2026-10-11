"""Canonical JSON and stable SHA-256 hashes for reports."""

import hashlib
import json
from typing import Any

FLOAT_DECIMALS = 6


def _normalise(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    raise TypeError(f"not canonicalisable: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Serialise so equal content always yields equal bytes, across runs and platforms."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
