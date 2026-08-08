from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.research_controller.state_machine.controller import (
    ResearchCampaignController,
    _state_snapshot_semantic_hash,
)
from src.research_controller.store.campaign_store import CampaignStore


CAMPAIGN_ID = "campaign_state_restore"
EXPERIMENT_ID = "exp_state_restore"
EXECUTION_ID = "exec_state_restore"
SNAPSHOT_ID = "state_exp_state_restore_12"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(store: CampaignStore) -> None:
    now = _now()
    store.create_campaign(
        {
            "campaign_id": CAMPAIGN_ID,
            "research_goal_id": "goal_state_restore",
            "status": "running",
            "current_stage": "BACKTEST",
            "config_sha256": "a" * 64,
            "protocol_bundle_sha256": "b" * 64,
            "data_snapshot_id": "data_dev",
            "universe_snapshot_id": "universe_dev",
            "gate_policy_version": "gate_champion_v1",
            "factor_inventory": {"target": 357, "completed": 0, "skipped": 0, "failed": 0},
            "queue_counts": {"pending": 0, "running": 1, "blocked": 0},
            "budget_usage": {"llm_calls_rolling_24h": 0, "candidate_versions_rolling_24h": 0},
            "last_event_sequences": {EXPERIMENT_ID: 2},
            "config": {},
            "preflight": {},
            "blocked_reason": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    store.upsert_experiment(
        {
            "experiment_id": EXPERIMENT_ID,
            "campaign_id": CAMPAIGN_ID,
            "round_id": "round_state_restore",
            "objective": "restore active execution",
            "hypothesis": {},
            "status": "active",
            "phase": "backtest_running",
            "data_snapshot_id": "data_dev",
            "universe_snapshot_id": "universe_dev",
            "gate_policy_version": "gate_champion_v1",
            "last_decision_action": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    store.upsert_execution(
        {
            "execution_id": EXECUTION_ID,
            "campaign_id": CAMPAIGN_ID,
            "experiment_id": EXPERIMENT_ID,
            "candidate_id": None,
            "candidate_version": None,
            "execution_type": "portfolio_backtest",
            "status": "running",
            "progress": 0.1,
            "current_stage": "queued",
            "created_at": now,
        }
    )


def _snapshot(*, state_hash: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "experiment_state_snapshot.v1",
        "experiment_id": EXPERIMENT_ID,
        "through_sequence": 12,
        "status": "active",
        "current_execution_ids": [EXECUTION_ID],
        "latest_candidate_versions": {},
        "latest_decision_id": None,
        "latest_evidence_ids": [],
        "latest_review_ids": [],
        "queue_summary": {"pending": 0, "running": 1, "blocked": 0},
        "state_sha256": "0" * 64,
        "created_at": "2026-08-01T10:00:00Z",
    }
    value["state_sha256"] = state_hash or _state_snapshot_semantic_hash(value)
    return value


class SnapshotDsa:
    def __init__(self, *, bad_hash: bool = False) -> None:
        self.snapshot = _snapshot(state_hash="f" * 64 if bad_hash else None)
        self.poll_cursors: list[int] = []
        self.start_calls = 0

    def poll_events(self, _experiment_id: str, *, after_sequence: int, page_size: int) -> dict[str, Any]:
        del page_size
        self.poll_cursors.append(after_sequence)
        if after_sequence < 12:
            return {
                "status": "error",
                "retryable": False,
                "error": {
                    "code": "event_cursor_expired",
                    "message": "compacted",
                    "details": {
                        "state_snapshot_id": SNAPSHOT_ID,
                        "state_snapshot_sha256": self.snapshot["state_sha256"],
                        "earliest_available_sequence": 5,
                    },
                },
            }
        return {
            "status": "ok",
            "data": {"items": [], "next_sequence": after_sequence, "has_more": False},
        }

    def get_experiment_state_snapshot(self, _experiment_id: str, _snapshot_id: str) -> dict[str, Any]:
        return {"status": "ok", "data": {"state_snapshot": self.snapshot}}

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "execution": {
                    "execution_id": execution_id,
                    "experiment_id": EXPERIMENT_ID,
                    "execution_type": "portfolio_backtest",
                    "status": "running",
                    "progress": 0.5,
                    "current_stage": "simulate",
                    "created_at": "2026-08-01T10:00:00Z",
                    "started_at": "2026-08-01T10:00:01Z",
                }
            },
        }


def _controller(store: CampaignStore, dsa: SnapshotDsa) -> ResearchCampaignController:
    return ResearchCampaignController(  # type: ignore[arg-type]
        store,
        dsa,
        capabilities_provider=lambda: {"status": "ok", "data": {}},
    )


def test_cursor_410_restores_active_execution_atomically_without_resubmission(tmp_path: Path) -> None:
    db_path = tmp_path / "state_restore.db"
    store = CampaignStore(db_path=db_path)
    _seed(store)
    dsa = SnapshotDsa()
    controller = _controller(store, dsa)

    controller.poll_events_once(CAMPAIGN_ID)

    campaign = store.get_campaign(CAMPAIGN_ID)
    assert campaign is not None
    assert campaign["status"] == "running"
    assert campaign["last_event_sequences"][EXPERIMENT_ID] == 12
    execution = store.get_execution(EXECUTION_ID)
    assert execution is not None
    assert execution["status"] == "running"
    assert execution["progress"] == 0.5
    assert execution["current_stage"] == "simulate"
    assert store.get_experiment_state_snapshot(SNAPSHOT_ID) is not None
    assert dsa.poll_cursors == [2, 12]
    assert dsa.start_calls == 0

    # Crash/restart replay is idempotent and keeps the same remote execution.
    restarted_store = CampaignStore(db_path=db_path)
    restarted = _controller(restarted_store, dsa)
    restarted_store.set_campaign_cursor(CAMPAIGN_ID, EXPERIMENT_ID, 2)
    restarted.poll_events_once(CAMPAIGN_ID)
    assert len(restarted_store.list_executions(CAMPAIGN_ID, EXPERIMENT_ID)) == 1
    assert restarted_store.get_execution(EXECUTION_ID)["status"] == "running"  # type: ignore[index]
    assert dsa.start_calls == 0


def test_cursor_410_bad_snapshot_hash_blocks_without_advancing_cursor(tmp_path: Path) -> None:
    store = CampaignStore(db_path=tmp_path / "bad_hash.db")
    _seed(store)
    controller = _controller(store, SnapshotDsa(bad_hash=True))

    controller.poll_events_once(CAMPAIGN_ID)

    campaign = store.get_campaign(CAMPAIGN_ID)
    assert campaign is not None
    assert campaign["status"] == "blocked"
    assert "state_sha256 mismatch" in campaign["blocked_reason"]
    assert campaign["last_event_sequences"][EXPERIMENT_ID] == 2
    assert store.get_experiment_state_snapshot(SNAPSHOT_ID) is None
