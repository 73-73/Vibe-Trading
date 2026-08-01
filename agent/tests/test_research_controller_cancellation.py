"""Durable Campaign cancellation and restart reconciliation tests (§16)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research_controller.client.dsa_client import DsaProtocolError, DsaUnavailableError
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore
from tests.research_loop_test_helpers import ready_capabilities_provider, sample_campaign_request


class CancellationDsa:
    def __init__(self, status: str = "running") -> None:
        self.status = status
        self.query_failures = 0
        self.query_protocol_failures = 0
        self.cancel_mode = "cancelled"
        self.get_calls = 0
        self.cancel_calls = 0
        self.terminal_fields: dict = {}

    def get_execution(self, execution_id: str) -> dict:
        self.get_calls += 1
        if self.query_failures:
            self.query_failures -= 1
            raise DsaUnavailableError("test query unavailable")
        if self.query_protocol_failures:
            self.query_protocol_failures -= 1
            raise DsaProtocolError("test malformed response")
        return {
            "status": "ok",
            "data": {
                "execution": {
                    "execution_id": execution_id,
                    "status": self.status,
                    "progress": 1.0 if self.status == "completed" else 0.4,
                    **self.terminal_fields,
                }
            },
        }

    def cancel_execution(self, execution_id: str) -> dict:
        self.cancel_calls += 1
        if self.cancel_mode == "unknown_but_cancelled":
            self.status = "cancelled"
            raise DsaUnavailableError("test response lost after commit")
        if self.cancel_mode == "ack_pending":
            return {
                "status": "ok",
                "data": {"execution_id": execution_id, "status": "running"},
            }
        if self.cancel_mode == "forbidden":
            return {
                "status": "error",
                "http_status": 403,
                "retryable": False,
                "error": {"code": "cancel_forbidden", "message": "operator permission required"},
            }
        self.status = "cancelled"
        return {
            "status": "ok",
            "data": {"execution_id": execution_id, "status": "cancelled"},
        }


def _seed(
    tmp_path: Path,
    dsa: CancellationDsa,
    *,
    execution_id: str = "exec_cancel_001",
) -> tuple[ResearchCampaignController, CampaignStore, str]:
    store = CampaignStore(db_path=tmp_path / "campaigns.db")
    controller = ResearchCampaignController(
        store,
        dsa,  # type: ignore[arg-type] - focused transport fake
        capabilities_provider=ready_capabilities_provider,
    )
    campaign_id = controller.create_campaign(sample_campaign_request())["campaign_id"]
    store.update_campaign(campaign_id, status="running", current_stage="PORTFOLIO_BACKTEST")
    store.upsert_execution(
        {
            "execution_id": execution_id,
            "campaign_id": campaign_id,
            "experiment_id": "exp_cancel_001",
            "execution_type": "portfolio_backtest",
            "status": "running",
            "progress": 0.4,
            "current_stage": "running",
            "created_at": "2026-08-01T00:00:00+00:00",
        }
    )
    return controller, store, campaign_id


def test_cancel_query_network_failure_recovers_without_new_work(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    dsa.query_failures = 1
    controller, store, campaign_id = _seed(tmp_path, dsa)

    cancelled = controller.cancel_campaign(campaign_id)
    assert cancelled["status"] == "cancelled"
    assert cancelled["current_stage"] == "CANCELLED"
    assert cancelled["cancellation"]["pending"] == 1
    assert cancelled["cancellation"]["items"][0]["diagnostic_code"] == "execution_query_unavailable"
    assert controller.list_active_campaign_ids() == [campaign_id]

    summary = controller.run_pipeline_once(campaign_id)
    assert summary["cancellations_pending"] == 0
    assert store.get_execution("exec_cancel_001")["status"] == "cancelled"
    assert controller.list_active_campaign_ids() == []


def test_unknown_cancel_outcome_is_queried_after_restart_not_blindly_replayed(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    dsa.cancel_mode = "unknown_but_cancelled"
    controller, store, campaign_id = _seed(tmp_path, dsa)

    first = controller.cancel_campaign(campaign_id)
    assert first["cancellation"]["pending"] == 1
    assert first["cancellation"]["complete"] is False
    assert first["cancellation"]["items"][0]["diagnostic_code"] == "cancel_transport_outcome_unknown"
    assert dsa.cancel_calls == 1

    restarted = ResearchCampaignController(
        CampaignStore(db_path=store.db_path),
        dsa,  # type: ignore[arg-type]
        capabilities_provider=ready_capabilities_provider,
    )
    restarted.run_pipeline_once(campaign_id)

    assert restarted.get_campaign(campaign_id)["cancellation"]["resolved"] == 1
    assert restarted.store.get_execution("exec_cancel_001")["status"] == "cancelled"
    assert dsa.cancel_calls == 1, "terminal query result must prevent blind cancel replay"


def test_cancel_ack_pending_is_idempotently_replayed_after_query(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    dsa.cancel_mode = "ack_pending"
    controller, store, campaign_id = _seed(tmp_path, dsa)

    first = controller.cancel_campaign(campaign_id)
    assert first["cancellation"]["pending"] == 1
    assert first["cancellation"]["complete"] is False
    assert controller.run_pipeline_once(campaign_id)["complete"] is False
    assert dsa.cancel_calls == 2

    dsa.cancel_mode = "cancelled"
    controller.run_pipeline_once(campaign_id)
    assert dsa.get_calls == 3
    assert dsa.cancel_calls == 3
    assert store.get_execution("exec_cancel_001")["status"] == "cancelled"


def test_nonretryable_cancel_diagnostic_blocks_completion_until_operator_retry(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    dsa.cancel_mode = "forbidden"
    controller, _store, campaign_id = _seed(tmp_path, dsa)

    detail = controller.cancel_campaign(campaign_id)
    summary = controller.run_pipeline_once(campaign_id)
    assert detail["cancellation"]["blocked"] == 1
    assert detail["cancellation"]["requires_operator"] is True
    assert detail["cancellation"]["complete"] is False
    assert summary["complete"] is False
    assert summary["cancellations_blocked"] == 1
    assert controller.list_active_campaign_ids() == []

    dsa.cancel_mode = "cancelled"
    recovered = controller.reconcile_campaign_cancellations(campaign_id)
    assert recovered["cancellation"]["complete"] is True
    assert recovered["cancellation"]["resolved"] == 1


def test_unexpected_remote_status_requires_explicit_operator_reconcile(tmp_path: Path) -> None:
    dsa = CancellationDsa("pausing")
    controller, _store, campaign_id = _seed(tmp_path, dsa)
    detail = controller.cancel_campaign(campaign_id)
    assert detail["cancellation"]["blocked"] == 1
    assert detail["cancellation"]["items"][0]["diagnostic_code"] == "unexpected_remote_execution_status"
    assert controller.run_pipeline_once(campaign_id)["complete"] is False

    dsa.status = "running"
    recovered = controller.reconcile_campaign_cancellations(campaign_id)
    assert recovered["cancellation"]["resolved"] == 1


def test_local_execution_missing_is_visible_and_restart_durable(tmp_path: Path) -> None:
    import sqlite3

    dsa = CancellationDsa()
    dsa.query_failures = 1
    controller, store, campaign_id = _seed(tmp_path, dsa)
    controller.cancel_campaign(campaign_id)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM executions WHERE execution_id='exec_cancel_001'")

    restarted = ResearchCampaignController(
        CampaignStore(db_path=store.db_path),
        dsa,  # type: ignore[arg-type]
        capabilities_provider=ready_capabilities_provider,
    )
    summary = restarted.run_pipeline_once(campaign_id)
    detail = restarted.get_campaign(campaign_id)
    assert summary["complete"] is False
    assert detail["cancellation"]["blocked"] == 1
    assert detail["cancellation"]["items"][0]["diagnostic_code"] == "local_execution_missing"


def test_repeated_protocol_errors_reach_threshold_then_operator_can_recover(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    dsa.query_protocol_failures = 3
    controller, _store, campaign_id = _seed(tmp_path, dsa)

    controller.cancel_campaign(campaign_id)
    controller.run_pipeline_once(campaign_id)
    summary = controller.run_pipeline_once(campaign_id)
    detail = controller.get_campaign(campaign_id)
    assert summary["complete"] is False
    assert summary["cancellations_blocked"] == 1
    assert detail["cancellation"]["items"][0]["attempt_count"] == 3
    assert detail["cancellation"]["items"][0]["diagnostic_code"] == "execution_query_protocol_error"

    recovered = controller.reconcile_campaign_cancellations(campaign_id)
    assert recovered["cancellation"]["complete"] is True
    assert recovered["cancellation"]["requires_operator"] is False


def test_repeated_cancel_is_idempotent_after_resolution(tmp_path: Path) -> None:
    dsa = CancellationDsa()
    controller, _store, campaign_id = _seed(tmp_path, dsa)

    first = controller.cancel_campaign(campaign_id)
    second = controller.cancel_campaign(campaign_id)

    assert first["cancellation"]["resolved"] == 1
    assert second["cancellation"]["resolved"] == 1
    assert dsa.cancel_calls == 1
    assert dsa.get_calls == 1


@pytest.mark.parametrize(
    ("remote_status", "terminal_fields", "expected_field", "expected_value"),
    [
        ("completed", {"evidence_id": "evidence_after_cancel"}, "evidence_id", "evidence_after_cancel"),
        ("failed", {"error_id": "error_after_cancel"}, "error_id", "error_after_cancel"),
    ],
)
def test_remote_terminal_wins_and_local_history_is_preserved(
    tmp_path: Path,
    remote_status: str,
    terminal_fields: dict,
    expected_field: str,
    expected_value: str,
) -> None:
    dsa = CancellationDsa(remote_status)
    dsa.terminal_fields = terminal_fields
    controller, store, campaign_id = _seed(tmp_path, dsa)
    store.upsert_evidence(
        {
            "evidence_id": "existing_evidence",
            "campaign_id": campaign_id,
            "experiment_id": "exp_cancel_001",
            "execution_id": "exec_cancel_001",
            "evidence": {"evidence_id": "existing_evidence", "bundle_sha256": "a" * 64},
            "created_at": "2026-08-01T00:00:00+00:00",
        }
    )

    detail = controller.cancel_campaign(campaign_id)

    execution = store.get_execution("exec_cancel_001")
    assert execution["status"] == remote_status
    assert execution[expected_field] == expected_value
    assert store.get_evidence("existing_evidence") is not None
    assert detail["cancellation"]["resolved"] == 1
    assert dsa.cancel_calls == 0
