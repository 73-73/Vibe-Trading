"""Deterministic JSON canonicalization for request/config hashing (§4.3).

Request hashes and ``config_sha256`` are computed over RFC 8785-style canonical
JSON UTF-8 bytes. This module implements a pragmatic canonical form: object keys
sorted by code point, minimal number serialization, and finite-only values.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize *value* into a deterministic canonical JSON string."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the hex SHA-256 over canonical JSON UTF-8 bytes of *value*."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite numbers are forbidden in canonical JSON")
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
