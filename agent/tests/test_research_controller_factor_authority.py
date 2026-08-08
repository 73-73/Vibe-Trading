from __future__ import annotations

from copy import deepcopy

import pytest

from src.factors.registry import Registry
from src.research_controller.factors.authority import (
    FactorAuthorityError,
    build_authoritative_manifest,
    validate_authoritative_manifest,
)


def test_authority_is_exactly_registry_owned_357_with_real_ids_and_hashes() -> None:
    registry = Registry()
    manifest = build_authoritative_manifest(registry)
    factors = manifest["factors"]
    ids = {item["factor_id"] for item in factors}
    assert manifest["factor_count"] == 357
    assert manifest["libraries"] == {"gtja191": 191, "qlib158": 154, "academic": 12}
    assert "gtja191_099" in ids
    assert "qlib158_rank20" in ids
    assert "gtja191_alpha099" not in ids
    assert "qlib158_RANK_20" not in ids
    for item in factors:
        assert item["source_sha256"] != item["formula_sha256"]
        assert len(item["source_sha256"]) == len(item["formula_sha256"]) == 64


def test_authority_build_is_byte_semantically_deterministic() -> None:
    assert build_authoritative_manifest() == build_authoritative_manifest()


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "hash"])
def test_authority_rejects_unknown_duplicate_and_hash_drift(mutation: str) -> None:
    manifest = deepcopy(build_authoritative_manifest())
    if mutation == "unknown":
        manifest["factors"][0]["library"] = "untrusted"
    elif mutation == "duplicate":
        manifest["factors"][1]["factor_id"] = manifest["factors"][0]["factor_id"]
    else:
        manifest["factors"][0]["source_sha256"] = "0" * 64
    with pytest.raises(FactorAuthorityError):
        validate_authoritative_manifest(manifest)
