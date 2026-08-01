"""Deterministic panel -> Registry.compute -> upload/register factor pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.factors.registry import Registry, RegistryError, SkipAlpha, get_default_registry
from src.research_controller.factors.authority import build_authoritative_manifest
from src.research_controller.store.canonical import canonical_sha256

_PANEL_REQUIRED = {
    "trade_date",
    "available_at",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
}
_OUTPUT_COLUMNS = ["as_of_date", "available_at", "symbol", "factor_id", "score", "rank_pct"]


class FactorPipelineError(RuntimeError):
    """The deterministic factor pipeline cannot safely publish a snapshot."""


@dataclass(frozen=True)
class FactorTerminalRecord:
    factor_id: str
    status: str
    reason: str
    rows: int = 0
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "status": self.status,
            "reason": self.reason,
            "rows": self.rows,
            "chunk_index": self.chunk_index,
        }


def load_market_panel(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load one unpartitioned DSA long CSV panel into Registry wide frames."""
    frame = pd.read_csv(path)
    missing = sorted(_PANEL_REQUIRED.difference(frame.columns))
    if missing:
        raise FactorPipelineError(f"market panel missing columns: {', '.join(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    if frame[["trade_date", "available_at", "symbol"]].isna().any().any():
        raise FactorPipelineError("market panel has null/invalid PIT identity columns")
    if (frame["available_at"] > frame["trade_date"]).any():
        raise FactorPipelineError("market panel contains data not visible as of trade_date")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise FactorPipelineError("market panel has duplicate (trade_date, symbol) rows")
    frame = frame.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    panel: dict[str, pd.DataFrame] = {}
    for column in sorted(_PANEL_REQUIRED - {"trade_date", "available_at", "symbol"}):
        panel[column] = frame.pivot(index="trade_date", columns="symbol", values=column).sort_index()
    return panel, frame


def _long_factor(factor_id: str, values: pd.DataFrame) -> pd.DataFrame:
    named = values.copy()
    named.index = pd.to_datetime(named.index)
    named.index.name = "as_of_date"
    named.columns.name = "symbol"
    score = named.stack(future_stack=True).rename("score").reset_index()
    score = score.loc[score["score"].notna()].copy()
    if score.empty:
        raise RegistryError(f"{factor_id}: no finite rows after warmup")
    numeric = pd.to_numeric(score["score"], errors="coerce")
    if numeric.isna().any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise RegistryError(f"{factor_id}: non-finite score")
    score["score"] = numeric.astype(float)
    score["rank_pct"] = score.groupby("as_of_date", sort=False)["score"].rank(
        method="average", pct=True
    )
    score["factor_id"] = factor_id
    # Daily OHLCV factors are available at the close of their own trade date.
    # This is an explicit source-availability policy, not a missing-column fill.
    score["available_at"] = score["as_of_date"]
    score["as_of_date"] = score["as_of_date"].dt.date.astype(str)
    score["available_at"] = score["available_at"].dt.date.astype(str)
    return score[_OUTPUT_COLUMNS]


def compute_factor_chunks(
    panel_path: Path,
    *,
    registry: Registry | None = None,
    factor_ids: list[str] | None = None,
    chunk_size: int = 25,
) -> tuple[pd.DataFrame, list[FactorTerminalRecord]]:
    """Compute all requested factors in stable chunks with a terminal record each."""
    if chunk_size < 1 or chunk_size > 25:
        raise FactorPipelineError("factor chunk_size must be within 1..25")
    source = registry or get_default_registry()
    authority = build_authoritative_manifest(source)
    authority_ids = [item["factor_id"] for item in authority["factors"]]
    selected = authority_ids if factor_ids is None else list(factor_ids)
    if len(selected) != len(set(selected)) or any(value not in set(authority_ids) for value in selected):
        raise FactorPipelineError("factor_ids must be unique members of the VIBE authority manifest")
    panel, _ = load_market_panel(panel_path)
    frames: list[pd.DataFrame] = []
    records: list[FactorTerminalRecord] = []
    for offset in range(0, len(selected), chunk_size):
        chunk_index = offset // chunk_size + 1
        for factor_id in selected[offset : offset + chunk_size]:
            try:
                long = _long_factor(factor_id, source.compute(factor_id, panel))
            except SkipAlpha as exc:
                records.append(FactorTerminalRecord(factor_id, "skipped", str(exc), 0, chunk_index))
            except (RegistryError, ValueError, TypeError) as exc:
                records.append(FactorTerminalRecord(factor_id, "failed", str(exc), 0, chunk_index))
            else:
                frames.append(long)
                records.append(FactorTerminalRecord(factor_id, "completed", "", len(long), chunk_index))
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["as_of_date", "symbol", "factor_id"], kind="stable")
        .reset_index(drop=True)
        if frames
        else pd.DataFrame(columns=_OUTPUT_COLUMNS)
    )
    return combined, records


def publish_factor_snapshot(
    *,
    client: Any,
    panel_path: Path,
    panel_manifest: dict[str, Any],
    output_path: Path,
    registry: Registry | None = None,
    factor_ids: list[str] | None = None,
    chunk_size: int = 25,
) -> dict[str, Any]:
    """Compute, upload, complete, and register one immutable factor snapshot."""
    source = registry or get_default_registry()
    frame, records = compute_factor_chunks(
        panel_path, registry=source, factor_ids=factor_ids, chunk_size=chunk_size
    )
    if frame.empty:
        raise FactorPipelineError("no factor completed; refusing to publish an empty snapshot")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    content = output_path.read_bytes()
    content_sha = hashlib.sha256(content).hexdigest()
    completed_ids = sorted(frame["factor_id"].unique().tolist())
    authority = {item["factor_id"]: item for item in build_authoritative_manifest(source)["factors"]}
    snapshot_id = f"factor_snapshot_{canonical_sha256({'panel': panel_manifest['panel_id'], 'sha256': content_sha})[:20]}"
    create = client.create_artifact_upload(
        {
            "kind": "factor_values",
            "media_type": "text/csv",
            "size_bytes": len(content),
            "sha256": content_sha,
        }
    )
    if create.get("status") != "ok":
        raise FactorPipelineError(f"artifact upload create failed: {create.get('error')}")
    upload_id = (create.get("data") or {}).get("upload_id")
    uploaded = client.upload_artifact_content(upload_id, output_path)
    if uploaded.get("status") != "ok":
        raise FactorPipelineError(f"artifact content upload failed: {uploaded.get('error')}")
    completed = client.complete_artifact_upload(upload_id)
    if completed.get("status") != "ok":
        raise FactorPipelineError(f"artifact complete failed: {completed.get('error')}")
    artifact_id = (completed.get("data") or {}).get("artifact_id")
    manifest = {
        "schema_version": "factor_snapshot.v1",
        "snapshot_id": snapshot_id,
        "data_snapshot_id": panel_manifest["data_snapshot_id"],
        "universe_snapshot_id": panel_manifest["universe_snapshot_id"],
        "factor_ids": completed_ids,
        "formula_versions": {factor_id: authority[factor_id]["formula_sha256"] for factor_id in completed_ids},
        "source_versions": {factor_id: authority[factor_id]["source_sha256"] for factor_id in completed_ids},
        "row_count": len(frame),
        "artifact_id": artifact_id,
        "sha256": content_sha,
    }
    registered = client.register_factor_snapshot({"manifest": manifest})
    if registered.get("status") != "ok":
        raise FactorPipelineError(f"factor snapshot registration failed: {registered.get('error')}")
    return {
        "factor_snapshot_id": snapshot_id,
        "manifest": manifest,
        "inventory": {
            "target": len(records),
            "completed": sum(item.status == "completed" for item in records),
            "skipped": sum(item.status == "skipped" for item in records),
            "failed": sum(item.status == "failed" for item in records),
        },
        "records": [item.to_dict() for item in records],
        "chunk_manifest_sha256": canonical_sha256(
            {"records": [item.to_dict() for item in records], "factor_snapshot": manifest}
        ),
    }
