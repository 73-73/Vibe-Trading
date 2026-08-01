"""Tests for the Research Campaign HTTP API (§7.2).

Mounts ``register_research_campaign_routes`` on a fresh FastAPI app with a
controller wired to the Mock DSA server and drives it through ``TestClient``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.research_loop_test_helpers import MockDsaServer, make_controller, sample_campaign_request
from src.research_controller.campaign_api.routes import (
    get_controller,
    register_research_campaign_routes,
    set_controller,
)
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore


@pytest.fixture
def mock() -> MockDsaServer:
    server = MockDsaServer("happy_path")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _app(controller: ResearchCampaignController) -> FastAPI:
    app = FastAPI()
    register_research_campaign_routes(app, require_auth=lambda: None)
    set_controller(controller)
    return app


def _client(tmp_path: Path, mock: MockDsaServer) -> TestClient:
    store = CampaignStore(db_path=tmp_path / "c.db")
    controller = make_controller(store, mock.base_url)
    return TestClient(_app(controller), client=("127.0.0.1", 50000))


def test_create_and_get_campaign(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    response = client.post("/research-campaigns", json=sample_campaign_request())
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "initializing"
    assert body["factor_inventory"]["target"] == 357

    campaign_id = body["campaign_id"]
    detail = client.get(f"/research-campaigns/{campaign_id}")
    assert detail.status_code == 200
    assert detail.json()["campaign_id"] == campaign_id
    assert "factor_inventory" in detail.json()


def test_list_campaigns_filter(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    client.post("/research-campaigns", json=sample_campaign_request())
    response = client.get("/research-campaigns")
    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert response.json()[0]["status"] in (
        "initializing",
        "running",
        "paused",
        "waiting_budget",
        "blocked",
        "completed",
        "cancelled",
    )


def test_create_campaign_rejects_locked_holdout(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    response = client.post(
        "/research-campaigns",
        json=sample_campaign_request(development_window={"start_date": "2010-01-01", "end_date": "2022-06-30"}),
    )
    assert response.status_code == 422
    assert "2022-01-01" in response.json()["detail"]


def test_create_campaign_rejects_open_locked_holdout(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    request = sample_campaign_request()
    request["automation"]["open_locked_holdout"] = True
    response = client.post("/research-campaigns", json=request)
    assert response.status_code == 422


def test_campaign_subresources(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    campaign_id = client.post("/research-campaigns", json=sample_campaign_request()).json()["campaign_id"]
    for path in (
        f"/research-campaigns/{campaign_id}/experiments",
        f"/research-campaigns/{campaign_id}/candidates",
        f"/research-campaigns/{campaign_id}/repairs",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_report_latest_404_before_generation(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    campaign_id = client.post("/research-campaigns", json=sample_campaign_request()).json()["campaign_id"]
    response = client.get(f"/research-campaigns/{campaign_id}/reports/latest")
    assert response.status_code == 404


def test_pause_resume_cancel_endpoints(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    campaign_id = client.post("/research-campaigns", json=sample_campaign_request()).json()["campaign_id"]

    paused = client.post(f"/research-campaigns/{campaign_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"

    resumed = client.post(f"/research-campaigns/{campaign_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "running"

    cancelled = client.post(f"/research-campaigns/{campaign_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    # 终态不能 resume
    assert client.post(f"/research-campaigns/{campaign_id}/resume").status_code == 409


def test_missing_campaign_404(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    assert client.get("/research-campaigns/campaign_missing").status_code == 404
    assert client.get("/research-campaigns/campaign_missing/experiments").status_code == 404


def test_invalid_path_param_rejected(tmp_path: Path, mock: MockDsaServer) -> None:
    client = _client(tmp_path, mock)
    assert client.get("/research-campaigns/bad%2Fid").status_code in (400, 404)


def test_report_generated_through_api(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    controller = make_controller(store, mock.base_url)
    client = TestClient(_app(controller), client=("127.0.0.1", 50000))
    campaign_id = client.post("/research-campaigns", json=sample_campaign_request()).json()["campaign_id"]
    for _ in range(60):
        if controller.run_pipeline_once(campaign_id)["complete"]:
            break
    generated = controller.generate_report(campaign_id)
    response = client.get(f"/research-campaigns/{campaign_id}/reports/latest")
    assert response.status_code == 200
    assert response.json()["report_id"] == generated["report_id"]


def test_controller_not_configured_raises() -> None:
    import src.research_controller.campaign_api.routes as routes_module

    original = routes_module._controller
    routes_module._controller = None
    try:
        with pytest.raises(RuntimeError):
            get_controller()
    finally:
        routes_module._controller = original
