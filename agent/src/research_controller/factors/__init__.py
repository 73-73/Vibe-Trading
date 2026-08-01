"""Authoritative VIBE factor inventory and compute/upload orchestration."""

from .authority import (
    A_SHARE_NATIVE_FACTOR_COUNT,
    A_SHARE_NATIVE_SCOPE,
    build_authoritative_manifest,
    validate_authoritative_manifest,
)

__all__ = [
    "A_SHARE_NATIVE_FACTOR_COUNT",
    "A_SHARE_NATIVE_SCOPE",
    "build_authoritative_manifest",
    "validate_authoritative_manifest",
]
