"""VIBE-owned 357-factor authority manifest (§11.1 / §11.10).

The implementation deliberately derives every entry from ``src.factors.Registry``
and the actual Python source file.  DSA consumes the exported immutable JSON; it
must never synthesize a parallel factor namespace or provisional hash set.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.factors.registry import Registry, get_default_registry
from src.research_controller.store.canonical import canonical_sha256

A_SHARE_NATIVE_SCOPE = "a_share_native_357"
A_SHARE_NATIVE_FACTOR_COUNT = 357
AUTHORITY_SCHEMA_VERSION = "factor_authority_manifest.v1"
_A_SHARE_NATIVE_ZOOS = ("gtja191", "qlib158", "academic")


class FactorAuthorityError(ValueError):
    """The source registry cannot produce the frozen research inventory."""


def _formula_payload(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula_latex": meta.get("formula_latex") or "",
        "columns_required": list(meta.get("columns_required") or []),
        "extras_required": list(meta.get("extras_required") or []),
        "requires_sector": bool(meta.get("requires_sector")),
        "min_warmup_bars": int(meta.get("min_warmup_bars") or 0),
    }


def build_authoritative_manifest(registry: Registry | None = None) -> dict[str, Any]:
    """Build the deterministic, source-hashed CN research factor manifest."""
    source = registry or get_default_registry()
    factor_ids = sorted(
        factor_id
        for zoo in _A_SHARE_NATIVE_ZOOS
        for factor_id in source.list(zoo=zoo)
    )
    if len(factor_ids) != A_SHARE_NATIVE_FACTOR_COUNT or len(set(factor_ids)) != len(factor_ids):
        raise FactorAuthorityError(
            f"{A_SHARE_NATIVE_SCOPE} requires exactly {A_SHARE_NATIVE_FACTOR_COUNT} unique factors, "
            f"found {len(factor_ids)}"
        )
    factors: list[dict[str, Any]] = []
    for factor_id in factor_ids:
        alpha = source.get(factor_id)
        metadata = dict(alpha.meta)
        source_code = source.get_source(factor_id)
        factors.append(
            {
                "factor_id": factor_id,
                "library": alpha.zoo,
                "module_path": alpha.module_path,
                "metadata": metadata,
                "formula_sha256": canonical_sha256(_formula_payload(metadata)),
                "source_sha256": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "scope_id": A_SHARE_NATIVE_SCOPE,
        "factor_count": len(factors),
        "libraries": {
            zoo: sum(1 for item in factors if item["library"] == zoo)
            for zoo in _A_SHARE_NATIVE_ZOOS
        },
        "factors": factors,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_authoritative_manifest(manifest)
    return manifest


def validate_authoritative_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on count, IDs, libraries, hashes, or manifest drift."""
    if manifest.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise FactorAuthorityError("unsupported factor authority schema")
    if manifest.get("scope_id") != A_SHARE_NATIVE_SCOPE:
        raise FactorAuthorityError("unexpected factor authority scope")
    factors = manifest.get("factors")
    if not isinstance(factors, list) or len(factors) != A_SHARE_NATIVE_FACTOR_COUNT:
        raise FactorAuthorityError("factor authority must contain exactly 357 entries")
    ids = [item.get("factor_id") for item in factors if isinstance(item, dict)]
    if len(ids) != len(factors) or len(set(ids)) != len(ids) or ids != sorted(ids):
        raise FactorAuthorityError("factor authority IDs must be unique and sorted")
    if any(item.get("library") not in _A_SHARE_NATIVE_ZOOS for item in factors):
        raise FactorAuthorityError("factor authority contains an unapproved library")
    for item in factors:
        for field in ("formula_sha256", "source_sha256"):
            value = item.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise FactorAuthorityError(f"{item.get('factor_id')}: invalid {field}")
        if (item.get("metadata") or {}).get("id") != item.get("factor_id"):
            raise FactorAuthorityError(f"{item.get('factor_id')}: metadata ID mismatch")
    claimed = manifest.get("manifest_sha256")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if claimed != canonical_sha256(semantic):
        raise FactorAuthorityError("factor authority manifest_sha256 mismatch")
