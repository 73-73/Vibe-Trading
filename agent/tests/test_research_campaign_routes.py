"""Research Campaign routes 集成冒烟（M4 接入 api_server 后的路由层验证）。

只验证路由注册 / 校验 / 404 / 创建生命周期；Controller 内部状态机由
``test_research_controller_state_machine*`` 覆盖。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api_server
from src.research_controller.campaign_api import routes
from src.research_controller.client.dsa_client import DsaLoopClient
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore


def _client(tmp_path: Path) -> TestClient:
    # create_campaign 只做校验与持久化，不调用 DSA；stub base_url 永不会连接。
    controller = ResearchCampaignController(
        store=CampaignStore(db_path=tmp_path / "campaigns.db"),
        dsa_client=DsaLoopClient(base_url="http://127.0.0.1:9", timeout=0.1),
    )
    routes.set_controller(controller)
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


def _campaign_payload() -> dict:
    return {
        "objective": "研究 A 股量价因子组合的横截面稳定性",
        "market": "CN",
        "frequency": "1d",
        "development_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
        "factor_scope": "a_share_native_357",
        "universe_policy": "pit_liquid_active_v1",
        "target_universe_size": 1200,
        "automation": {"open_locked_holdout": False},
        "report_language": "zh-CN",
    }


def test_campaign_create_and_get(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/research-campaigns", json=_campaign_payload())
    assert resp.status_code == 201, resp.text
    campaign = resp.json()
    assert campaign["status"] == "initializing"
    assert campaign["gate_policy_version"] == "gate_champion_v1"
    cid = campaign["campaign_id"]
    detail = client.get(f"/research-campaigns/{cid}")
    assert detail.status_code == 200
    assert detail.json()["campaign_id"] == cid


def test_campaign_rejects_locked_holdout_window(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = _campaign_payload()
    payload["development_window"]["end_date"] = "2022-06-30"
    resp = client.post("/research-campaigns", json=payload)
    assert resp.status_code == 422


def test_campaign_rejects_open_locked_holdout_flag(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = _campaign_payload()
    payload["automation"]["open_locked_holdout"] = True
    resp = client.post("/research-campaigns", json=payload)
    assert resp.status_code == 422


def test_campaign_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/research-campaigns/does_not_exist").status_code == 404


def test_campaign_list_and_pause(tmp_path: Path) -> None:
    client = _client(tmp_path)
    cid = client.post("/research-campaigns", json=_campaign_payload()).json()["campaign_id"]
    listing = client.get("/research-campaigns")
    assert listing.status_code == 200
    assert any(c["campaign_id"] == cid for c in listing.json())
    paused = client.post(f"/research-campaigns/{cid}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
