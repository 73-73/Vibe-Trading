from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.factors.registry import Registry
from src.research_controller.factors.pipeline import compute_factor_chunks, publish_factor_snapshot


def _panel(path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for date_index, trade_date in enumerate(pd.bdate_range("2020-01-02", periods=80), start=1):
        for symbol_index, symbol in enumerate(("000001.SZ", "600519.SH", "000002.SZ"), start=1):
            close = 10.0 + date_index * symbol_index / 10
            rows.append(
                {
                    "trade_date": trade_date.date(),
                    "available_at": "2010-01-01",
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + date_index * symbol_index,
                    "amount": close * (1000 + date_index * symbol_index),
                    "vwap": close,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class UploadClient:
    def __init__(self) -> None:
        self.content: bytes | None = None
        self.registered: dict[str, Any] | None = None

    def create_artifact_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.create_payload = payload
        return {"status": "ok", "data": {"upload_id": "upload_factor"}}

    def upload_artifact_content(self, upload_id: str, path: Path) -> dict[str, Any]:
        assert upload_id == "upload_factor"
        self.content = path.read_bytes()
        return {"status": "ok", "data": {"received_size_bytes": len(self.content)}}

    def complete_artifact_upload(self, upload_id: str) -> dict[str, Any]:
        assert upload_id == "upload_factor"
        return {"status": "ok", "data": {"artifact_id": "artifact_factor"}}

    def register_factor_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.registered = payload
        return {"status": "ok", "data": {"factor_snapshot_id": payload["manifest"]["snapshot_id"]}}


def test_chunking_matches_single_batch_exactly(tmp_path: Path) -> None:
    path = _panel(tmp_path / "panel.csv")
    registry = Registry()
    factor_ids = ["qlib158_roc5", "qlib158_ma10", "gtja191_099"]
    one, one_records = compute_factor_chunks(path, registry=registry, factor_ids=factor_ids, chunk_size=3)
    chunked, chunked_records = compute_factor_chunks(path, registry=registry, factor_ids=factor_ids, chunk_size=1)
    pd.testing.assert_frame_equal(one, chunked)
    assert [(item.factor_id, item.status, item.rows) for item in one_records] == [
        (item.factor_id, item.status, item.rows) for item in chunked_records
    ]


def test_publish_uses_authority_hashes_and_exact_content(tmp_path: Path) -> None:
    panel = _panel(tmp_path / "panel.csv")
    output = tmp_path / "factor.csv"
    client = UploadClient()
    result = publish_factor_snapshot(
        client=client,
        panel_path=panel,
        panel_manifest={
            "panel_id": "panel_test",
            "data_snapshot_id": "data_test",
            "universe_snapshot_id": "universe_test",
        },
        output_path=output,
        registry=Registry(),
        factor_ids=["qlib158_roc5", "qlib158_ma10"],
        chunk_size=1,
    )
    assert result["inventory"] == {"target": 2, "completed": 2, "skipped": 0, "failed": 0}
    manifest = client.registered["manifest"]  # type: ignore[index]
    assert manifest["row_count"] == len(pd.read_csv(output))
    assert manifest["sha256"] == hashlib.sha256(client.content).hexdigest()  # type: ignore[arg-type]
    assert set(manifest["formula_versions"]) == set(manifest["factor_ids"])
    assert set(manifest["source_versions"]) == set(manifest["factor_ids"])
