"""Fail-closed Campaign preflight tests (§17.2 / §20)."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from fastapi import FastAPI

from src.research_controller.client.dsa_client import DsaLoopClient
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore
from tests.research_loop_test_helpers import (
    ready_capabilities_provider,
    sample_campaign_request,
)


def _controller(
    tmp_path: Path,
    provider,
    *,
    require_longbridge: bool = False,
) -> ResearchCampaignController:
    return ResearchCampaignController(
        CampaignStore(db_path=tmp_path / "campaigns.db"),
        DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
        capabilities_provider=provider,
        require_longbridge=require_longbridge,
    )


def _caps() -> dict:
    return deepcopy(ready_capabilities_provider())


def _assembled_controller(monkeypatch, tmp_path: Path, caps: dict) -> ResearchCampaignController:
    import src.research_controller.campaign_api.routes as routes_module
    import src.research_controller.client.dsa_client as client_module
    import src.research_controller.store.campaign_store as store_module
    from src.research_controller.campaign_api.assemble import register_research_campaign

    real_store = CampaignStore(db_path=tmp_path / "assembled.db")

    class _ProductionDsa:
        def get_capabilities(self) -> dict:
            return caps

    class _TestOnlyGenerators:
        def verify_online(self) -> None:
            pass

        def hypothesis_generator(self, config, existing):  # noqa: ANN001,ANN201
            raise AssertionError("generation is outside this preflight test")

        def code_generator(self, experiment, candidate_id, candidate_version, **kwargs):  # noqa: ANN001,ANN201
            raise AssertionError("generation is outside this preflight test")

    monkeypatch.setattr(client_module, "DsaLoopClient", _ProductionDsa)
    monkeypatch.setattr(store_module, "CampaignStore", lambda: real_store)
    app = FastAPI()
    register_research_campaign(
        app,
        require_auth=lambda: None,
        _test_generator_factory=_TestOnlyGenerators,
    )
    return routes_module.get_controller()


def test_create_all_capabilities_passes_preflight_before_running(tmp_path: Path) -> None:
    controller = _controller(tmp_path, ready_capabilities_provider)

    created = controller.create_campaign(sample_campaign_request())

    assert created["status"] == "initializing"
    assert created["current_stage"] == "PREFLIGHT"
    assert created["preflight"]["status"] == "passed"
    assert created["blocking_items"] == []

    summary = controller.run_pipeline_once(created["campaign_id"])
    assert summary["status"] == "running"
    detail = controller.get_campaign(created["campaign_id"])
    assert detail["current_stage"] == "WAIT_DATA"
    assert detail["blocking_items"] == []


def test_create_missing_execution_types_is_blocked_and_not_active(tmp_path: Path) -> None:
    caps = _caps()
    caps["data"]["execution_types"] = ["data_snapshot_build", "market_panel_export"]
    controller = _controller(tmp_path, lambda: caps)

    created = controller.create_campaign(sample_campaign_request())

    assert created["status"] == "blocked"
    assert created["current_stage"] == "PREFLIGHT"
    assert [item["code"] for item in created["blocking_items"]] == ["missing_execution_types"]
    assert "missing_execution_types" in created["blocked_reason"]
    assert controller.list_active_campaign_ids() == []


def test_create_missing_bundle_protocol_and_readiness_lists_each_blocker(tmp_path: Path) -> None:
    caps = _caps()
    payload = caps["data"]
    payload.pop("contract_bundle_sha256")
    payload["protocol_versions"] = []
    payload["readiness"].pop("sandbox")
    payload["readiness"]["reviewer"] = {
        "status": "not_ready",
        "available": False,
        "ready": False,
        "reason": "remote reviewer endpoint is not configured",
    }
    controller = _controller(tmp_path, lambda: caps)

    created = controller.create_campaign(sample_campaign_request())

    codes = {item["code"] for item in created["blocking_items"]}
    assert {
        "contract_bundle_hash_missing",
        "missing_protocol_versions",
        "sandbox_not_ready",
        "reviewer_not_ready",
    } <= codes
    assert "remote reviewer endpoint is not configured" in created["blocked_reason"]


def test_resume_rechecks_and_preflight_block_can_recover(tmp_path: Path) -> None:
    state = {"caps": _caps()}
    controller = _controller(tmp_path, lambda: state["caps"])
    campaign_id = controller.create_campaign(sample_campaign_request())["campaign_id"]
    controller.run_pipeline_once(campaign_id)
    controller.pause_campaign(campaign_id)

    degraded = _caps()
    degraded["data"]["readiness"]["data"] = {
        "status": "not_ready",
        "available": False,
        "ready": False,
        "reason": "development data directory is unavailable",
    }
    state["caps"] = degraded

    blocked = controller.resume_campaign(campaign_id)
    assert blocked["status"] == "blocked"
    assert blocked["current_stage"] == "PREFLIGHT"
    assert [item["code"] for item in blocked["blocking_items"]] == ["data_not_ready"]

    state["caps"] = _caps()
    resumed = controller.resume_campaign(campaign_id)
    assert resumed["status"] == "running"
    assert resumed["current_stage"] == "WAIT_DATA"
    assert resumed["blocked_reason"] is None
    assert resumed["blocking_items"] == []


def test_longbridge_only_blocks_when_explicitly_required(tmp_path: Path) -> None:
    caps = _caps()
    caps["data"]["readiness"]["longbridge"] = {
        "status": "not_ready",
        "available": False,
        "ready": False,
        "reason": "real Longbridge provider is unavailable",
    }

    optional = _controller(tmp_path / "optional", lambda: caps)
    assert optional.create_campaign(sample_campaign_request())["preflight"]["status"] == "passed"

    required = _controller(tmp_path / "required", lambda: caps, require_longbridge=True)
    created = required.create_campaign(sample_campaign_request())
    assert created["status"] == "blocked"
    assert [item["code"] for item in created["blocking_items"]] == ["longbridge_not_ready"]


def test_production_assembly_requires_longbridge_and_redacts_unknown_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    caps = _caps()
    caps["data"]["readiness"]["longbridge"] = {
        "status": "not_ready",
        "available": False,
        "ready": False,
        "reason": "real provider is not configured",
        "credential": "SENTINEL_SECRET_VALUE",
    }
    controller = _assembled_controller(monkeypatch, tmp_path, caps)

    created = controller.create_campaign(sample_campaign_request())
    listing = controller.list_campaigns(status="blocked")

    assert created["status"] == "blocked"
    assert [item["code"] for item in created["blocking_items"]] == ["longbridge_not_ready"]
    assert listing[0]["campaign_id"] == created["campaign_id"]
    assert "SENTINEL_SECRET_VALUE" not in json.dumps(listing, ensure_ascii=False)


def test_production_assembly_all_ready_passes(tmp_path: Path, monkeypatch) -> None:
    controller = _assembled_controller(monkeypatch, tmp_path, _caps())
    created = controller.create_campaign(sample_campaign_request())
    assert created["status"] == "initializing"
    assert created["preflight"]["status"] == "passed"
    assert "longbridge" in created["preflight"]["required_components"]


def test_malformed_capabilities_fail_closed(tmp_path: Path) -> None:
    controller = _controller(tmp_path, lambda: {"status": "ok", "data": None})
    created = controller.create_campaign(sample_campaign_request())
    assert created["status"] == "blocked"
    assert any(item["code"] == "dsa_capabilities_malformed" for item in created["blocking_items"])


def test_unexpected_capabilities_failure_is_persisted_as_blocker(tmp_path: Path) -> None:
    def _broken_provider() -> dict:
        raise RuntimeError("test internal failure")

    created = _controller(tmp_path, _broken_provider).create_campaign(sample_campaign_request())
    assert created["status"] == "blocked"
    assert [item["code"] for item in created["blocking_items"]] == [
        "dsa_capabilities_internal_error"
    ]


def test_artifact_upload_allowlist_missing_or_unknown_fails_closed(tmp_path: Path) -> None:
    missing = _caps()
    missing["data"].pop("artifact_uploads")
    created = _controller(tmp_path / "missing", lambda: missing).create_campaign(
        sample_campaign_request()
    )
    missing_codes = {item["code"] for item in created["blocking_items"]}
    assert "artifact_upload_capabilities_missing" in missing_codes
    assert "artifact_upload_media_type_not_allowed" in missing_codes

    wrong = _caps()
    wrong["data"]["artifact_uploads"]["allowlist"] = {
        "factor_values": ["application/zip"],
    }
    created = _controller(tmp_path / "wrong", lambda: wrong).create_campaign(
        sample_campaign_request()
    )
    assert [item["code"] for item in created["blocking_items"]] == [
        "artifact_upload_media_type_not_allowed"
    ]


def test_restart_rechecks_legacy_running_campaign_before_any_work(tmp_path: Path) -> None:
    db_path = tmp_path / "campaigns.db"
    store = CampaignStore(db_path=db_path)
    first = ResearchCampaignController(
        store,
        DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
        capabilities_provider=ready_capabilities_provider,
    )
    campaign_id = first.create_campaign(sample_campaign_request())["campaign_id"]
    first.run_pipeline_once(campaign_id)
    store.update_campaign(campaign_id, status="running", current_stage="WAIT_DATA", preflight_json="{}")

    degraded = _caps()
    degraded["data"]["execution_types"] = ["data_snapshot_build", "market_panel_export"]
    restarted = ResearchCampaignController(
        CampaignStore(db_path=db_path),
        DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
        capabilities_provider=lambda: degraded,
    )
    summary = restarted.run_pipeline_once(campaign_id)
    detail = restarted.get_campaign(campaign_id)

    assert summary["status"] == "blocked"
    assert detail["current_stage"] == "PREFLIGHT"
    assert [item["code"] for item in detail["blocking_items"]] == ["missing_execution_types"]
