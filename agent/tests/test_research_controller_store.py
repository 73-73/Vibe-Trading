"""Tests for the Research Campaign SQLite store.

Covers crash-safe persistence, the single-transaction bundle (events + cursor +
derived state, §7.2.1), at-least-once dedupe by ``message_id`` (§4.3), immutable
decisions and idempotency records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research_controller.store.campaign_store import CampaignStore
from src.research_controller.store.canonical import canonical_json, canonical_sha256


@pytest.fixture
def store(tmp_path: Path) -> CampaignStore:
    return CampaignStore(db_path=tmp_path / "campaigns.db")


def _campaign_record(campaign_id: str = "campaign_test_001") -> dict:
    return {
        "campaign_id": campaign_id,
        "research_goal_id": "goal_test_001",
        "status": "initializing",
        "current_stage": "WAIT_DATA",
        "config_sha256": "a" * 64,
        "protocol_bundle_sha256": "b" * 64,
        "data_snapshot_id": None,
        "universe_snapshot_id": None,
        "gate_policy_version": "gate_champion_v1",
        "factor_inventory": {"target": 357, "completed": 0, "skipped": 0, "failed": 0},
        "queue_counts": {"pending": 0, "running": 0, "blocked": 0},
        "budget_usage": {"llm_calls_rolling_24h": 0, "candidate_versions_rolling_24h": 0},
        "last_event_sequences": {},
        "config": {"objective": "test"},
        "blocked_reason": None,
        "created_at": "2026-08-01T10:00:00+00:00",
        "updated_at": "2026-08-01T10:00:00+00:00",
    }


def test_campaign_crud(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record())
    campaign = store.get_campaign("campaign_test_001")
    assert campaign["campaign_id"] == "campaign_test_001"
    assert campaign["status"] == "initializing"
    assert campaign["factor_inventory"]["target"] == 357

    store.update_campaign("campaign_test_001", status="running")
    assert store.get_campaign("campaign_test_001")["status"] == "running"
    assert len(store.list_campaigns(status="running")) == 1
    assert store.get_campaign("missing") is None


def test_campaign_rows_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "campaigns.db"
    store = CampaignStore(db_path=path)
    store.create_campaign(_campaign_record("campaign_reopen"))
    # 模拟进程重启：新建 store 实例重新打开同一文件
    store2 = CampaignStore(db_path=path)
    assert store2.get_campaign("campaign_reopen")["research_goal_id"] == "goal_test_001"


def test_apply_bundle_atomic_events_cursor_and_state(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_bundle"))
    event = {
        "message_id": "msg_001",
        "sequence": 1,
        "producer": "dsa_api",
        "event_type": "experiment.created",
        "payload": {"experiment_id": "exp_1"},
        "created_at": "2026-08-01T10:00:01+00:00",
    }
    store.apply_bundle(
        {
            "events": [("campaign_bundle", "exp_1", event)],
            "cursor": ("campaign_bundle", "exp_1", 1),
            "experiments": [("exp_1", {"phase": "created", "status": "active", "updated_at": "x"})],
        }
    )
    events = store.list_events("campaign_bundle", "exp_1")
    assert len(events) == 1
    assert events[0]["message_id"] == "msg_001"
    cursor = store.get_campaign("campaign_bundle")["last_event_sequences"]
    assert cursor["exp_1"] == 1


def test_apply_bundle_dedupes_by_message_id(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_dedupe"))
    event = {
        "message_id": "msg_dup",
        "sequence": 1,
        "producer": "dsa_api",
        "event_type": "experiment.created",
        "payload": {},
        "created_at": "2026-08-01T10:00:01+00:00",
    }
    # 至少一次投递：同一 message_id 重复投递只落库一次
    store.apply_bundle({"events": [("campaign_dedupe", "exp_1", event)], "cursor": ("campaign_dedupe", "exp_1", 1)})
    store.apply_bundle({"events": [("campaign_dedupe", "exp_1", event)], "cursor": ("campaign_dedupe", "exp_1", 1)})
    store.insert_event(
        {"campaign_id": "campaign_dedupe", "experiment_id": "exp_1", "message_id": "msg_dup",
         "sequence": 2, "producer": "x", "event_type": "y", "created_at": "2026-08-01T10:00:02+00:00"}
    )
    events = store.list_events("campaign_dedupe", "exp_1")
    assert len(events) == 1


def test_cursor_only_advances_forward(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_cursor"))
    store.apply_bundle({"cursor": ("campaign_cursor", "exp_1", 10)})
    store.apply_bundle({"cursor": ("campaign_cursor", "exp_1", 5)})
    assert store.get_campaign("campaign_cursor")["last_event_sequences"]["exp_1"] == 10


def test_decisions_are_immutable(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_decision"))
    decision = {
        "decision_id": "decision_001",
        "campaign_id": "campaign_decision",
        "experiment_id": "exp_1",
        "round_id": "round_1",
        "action": "submit_candidate_revision",
        "rationale": "accept reviewer",
        "source_evidence_ids": ["evidence_1"],
        "source_review_id": "review_1",
        "next_candidate_id": "c1",
        "next_candidate_version": 2,
        "created_at": "2026-08-01T10:00:00+00:00",
    }
    assert store.insert_decision(decision) is True
    assert store.insert_decision(decision) is False  # 不可变
    assert len(store.list_decisions("campaign_decision")) == 1


def test_idempotency_record_roundtrip(store: CampaignStore) -> None:
    store.save_idempotency("start_execution", "key_1", "abc", 200, '{"execution_id":"exec_1"}', "exec_1")
    record = store.idempotency_lookup("start_execution", "key_1")
    assert record["resource_id"] == "exec_1"
    assert store.idempotency_lookup("start_execution", "missing") is None


def test_rolling_budget_usage(store: CampaignStore) -> None:
    assert store.rolling_budget_usage("llm_calls") == 0
    store.record_budget_usage("llm_calls")
    store.record_budget_usage("llm_calls")
    assert store.rolling_budget_usage("llm_calls") == 2
    assert store.rolling_budget_usage("candidate_versions") == 0


def test_repairs_and_evidence(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_repair"))
    repair = {
        "candidate_id": "c1",
        "candidate_version": 2,
        "campaign_id": "campaign_repair",
        "experiment_id": "exp_1",
        "parent_version": 1,
        "repair_of_error_id": "err_1",
        "error_fingerprint": "fp",
        "change_summary": "增加零值保护",
        "preserved_constraints": ["hypothesis"],
        "created_at": "2026-08-01T10:00:00+00:00",
    }
    assert store.insert_repair(repair) is True
    assert store.insert_repair(repair) is False  # (candidate_id, version) 唯一
    assert store.list_repairs("campaign_repair")[0]["change_summary"] == "增加零值保护"

    evidence = {
        "evidence_id": "evidence_1",
        "campaign_id": "campaign_repair",
        "experiment_id": "exp_1",
        "execution_id": "exec_1",
        "evidence": {"evidence_id": "evidence_1", "bundle_sha256": "a" * 64},
        "created_at": "2026-08-01T10:00:00+00:00",
    }
    store.upsert_evidence(evidence)
    assert store.get_evidence("evidence_1")["evidence"]["bundle_sha256"] == "a" * 64


def test_reports_saved_and_latest(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_report"))
    store.save_report("campaign_report", "report_1", "# r1")
    store.save_report("campaign_report", "report_2", "# r2")
    latest = store.get_latest_report("campaign_report")
    assert latest["report_id"] == "report_2"


def test_canonical_json_is_deterministic() -> None:
    a = {"b": 1, "a": [3, 2], "c": 1.0}
    b = {"c": 1, "a": [3, 2.0], "b": 1.0}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_sha256(a) == canonical_sha256(b)


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_unknown_update_field_rejected(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_bad"))
    with pytest.raises(ValueError):
        store.update_campaign("campaign_bad", nonexistent_field=1)


def test_list_campaigns_status_filter_and_limit(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("c1"))
    store.create_campaign(_campaign_record("c2"))
    store.update_campaign("c2", status="running")
    assert len(store.list_campaigns()) == 2
    assert len(store.list_campaigns(status="running")) == 1
    assert len(store.list_campaigns(limit=1)) == 1


def test_json_blob_roundtrip_survives_none(store: CampaignStore) -> None:
    store.create_campaign(_campaign_record("campaign_none"))
    campaign = store.get_campaign("campaign_none")
    assert campaign["data_snapshot_id"] is None
    assert campaign["blocked_reason"] is None
    assert json.loads(json.dumps(campaign, ensure_ascii=False))  # JSON 可序列化
