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
    set_controller_factory,
)
from src.research_controller.client.dsa_client import DsaLoopClient
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

    original_controller = routes_module._controller
    original_factory = routes_module._controller_factory
    routes_module._controller = None
    routes_module._controller_factory = None
    try:
        with pytest.raises(RuntimeError):
            get_controller()
    finally:
        routes_module._controller = original_controller
        routes_module._controller_factory = original_factory


def test_factory_swap_uses_new_controller_for_requests(tmp_path: Path) -> None:
    """Replacing the factory must serve subsequent requests from the new controller.

    Regression: the module-level ``_controller`` singleton was built once from
    the factory and cached; swapping the factory kept serving the stale
    instance, leaking state across configs / databases / app instances.
    """
    import src.research_controller.campaign_api.routes as routes_module

    saved_controller = routes_module._controller
    saved_factory = routes_module._controller_factory
    try:
        controller_a = ResearchCampaignController(
            store=CampaignStore(db_path=tmp_path / "a.db"),
            dsa_client=DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
        )
        controller_b = ResearchCampaignController(
            store=CampaignStore(db_path=tmp_path / "b.db"),
            dsa_client=DsaLoopClient(base_url="http://127.0.0.1:10", timeout=0.1),
        )

        app = FastAPI()
        register_research_campaign_routes(app, require_auth=lambda: None)
        routes_module.set_controller_factory(lambda: controller_a)

        client = TestClient(app, client=("127.0.0.1", 50000))
        created = client.post("/research-campaigns", json=sample_campaign_request())
        assert created.status_code == 201
        campaign_id = created.json()["campaign_id"]

        # Swap the factory; the next request must go to controller_b (empty
        # store), not the stale cached controller_a.
        routes_module.set_controller_factory(lambda: controller_b)
        assert client.get(f"/research-campaigns/{campaign_id}").status_code == 404
        assert client.get("/research-campaigns").json() == []
    finally:
        routes_module._controller = saved_controller
        routes_module._controller_factory = saved_factory


def test_set_controller_overrides_leftover_factory(tmp_path: Path) -> None:
    """set_controller(c) must win over a factory registered earlier.

    Regression: api_server registers a factory at import time; a test seam that
    calls set_controller(c) must clear that factory, otherwise get_controller()
    could lazily rebuild a different controller.
    """
    import src.research_controller.campaign_api.routes as routes_module

    saved_controller = routes_module._controller
    saved_factory = routes_module._controller_factory
    try:
        explicit = ResearchCampaignController(
            store=CampaignStore(db_path=tmp_path / "explicit.db"),
            dsa_client=DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
        )
        other = ResearchCampaignController(
            store=CampaignStore(db_path=tmp_path / "other.db"),
            dsa_client=DsaLoopClient(base_url="http://127.0.0.1:11", timeout=0.1),
        )

        routes_module.set_controller_factory(lambda: other)
        routes_module.set_controller(explicit)
        # Explicit instance wins; leftover factory must not rebuild another one.
        assert routes_module.get_controller() is explicit

        # A later set_controller_factory is still authoritative (rebuilds).
        routes_module.set_controller_factory(lambda: other)
        assert routes_module.get_controller() is other
    finally:
        routes_module._controller = saved_controller
        routes_module._controller_factory = saved_factory
