"""M4 end-to-end integration against the Mock DSA server as a real process.

The Mock DSA server is launched with uvicorn as a subprocess (the same way it
runs in the DSA repo), and the Research Controller drives the full loop over
real loopback HTTP (§18.8):

- create campaign -> data snapshot -> experiment -> candidate -> execution ->
  review -> decision -> report (happy_path);
- error/repair round trip (execution_failure): v1 fails -> Reviewer -> decision
  -> v2 submitted -> same-fingerprint stop;
- restart recovery: a fresh controller instance over the same store does not
  resubmit accepted candidates/executions.

Mock success is not integration success with the real DSA; this file only
proves the controller/HTTP layer against the frozen Mock scenarios.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Iterator

import pytest

from tests.research_loop_test_helpers import (
    DSA_BACKTEST_LAB_PATH,
    ready_capabilities_provider,
    sample_campaign_request,
)
from src.research_controller.client.dsa_client import DsaLoopClient
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore

DSA_REPO_ROOT = Path("/Users/pandeng/Projects/daily_stock_analysis")
DSA_VENV_PYTHON = DSA_BACKTEST_LAB_PATH / ".venv" / "bin" / "python"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(base_url: str, timeout: float = 20.0) -> None:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/internal/v1/research-loop/capabilities", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            time.sleep(0.1)
    raise RuntimeError(f"mock DSA server not ready at {base_url}")


class _MockProcess:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.process: subprocess.Popen | None = None
        self.base_url: str | None = None

    def start(self) -> str:
        if not DSA_REPO_ROOT.is_dir():
            pytest.skip(f"DSA repo not available at {DSA_REPO_ROOT}")
        if not DSA_VENV_PYTHON.is_file():
            pytest.skip(f"DSA venv python not available at {DSA_VENV_PYTHON}")
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.process = subprocess.Popen(
            [
                str(DSA_VENV_PYTHON),
                "-m",
                "services.backtest_lab.research_loop.mocks.mock_server",
                "--scenario",
                self.scenario,
                "--port",
                str(port),
            ],
            cwd=str(DSA_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(self.base_url)
        except Exception:
            self.stop()
            raise
        return self.base_url

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self) -> "_MockProcess":
        self.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False


@pytest.fixture
def mock_process() -> Iterator[_MockProcess]:
    proc = _MockProcess("happy_path")
    proc.start()
    try:
        yield proc
    finally:
        proc.stop()


def _drive(ctrl: ResearchCampaignController, campaign_id: str, max_iters: int = 100) -> dict:
    for _ in range(max_iters):
        summary = ctrl.run_pipeline_once(campaign_id)
        if summary["complete"]:
            return summary
    return ctrl.run_pipeline_once(campaign_id)


def test_end_to_end_happy_path_over_real_http(tmp_path: Path, mock_process: _MockProcess) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = ResearchCampaignController(
        store,
        DsaLoopClient(base_url=mock_process.base_url),
        capabilities_provider=ready_capabilities_provider,
    )
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]

    summary = _drive(ctrl, campaign_id)
    assert summary["status"] == "completed"
    assert summary["experiments_total"] == 1

    campaign = ctrl.get_campaign(campaign_id)
    assert campaign["current_stage"] == "COMPLETED"
    assert campaign["data_snapshot_id"] == "data_dev_001"
    assert campaign["gate_policy_version"] == "gate_champion_v1"

    decisions = ctrl.get_decisions(campaign_id)
    assert "accept_result" in [d["action"] for d in decisions]

    report = ctrl.generate_report(campaign_id)
    for section in ("## 1. 研究目标与预注册假设", "## 9. Artifact / 数据 / 公式 / 代码 / 规则哈希"):
        assert section in report["report_md"]


def test_end_to_end_repair_lineage_over_real_http(tmp_path: Path) -> None:
    with _MockProcess("execution_failure") as proc:
        store = CampaignStore(db_path=tmp_path / "c.db")
        ctrl = ResearchCampaignController(
            store,
            DsaLoopClient(base_url=proc.base_url),
            capabilities_provider=ready_capabilities_provider,
        )
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        _drive(ctrl, campaign_id)

        candidates = ctrl.get_candidates(campaign_id)
        assert [c["candidate_version"] for c in candidates] == [1, 2]
        v2 = candidates[-1]
        assert v2["parent_version"] == 1
        assert v2["repair_of_error_id"]

        repairs = ctrl.get_repairs(campaign_id)
        assert repairs and repairs[0]["parent_version"] == 1

        actions = [d["action"] for d in ctrl.get_decisions(campaign_id)]
        assert "submit_candidate_revision" in actions
        assert actions[-1] == "abandon_candidate"  # 相同指纹连续 2 次提前停止

        report = ctrl.generate_report(campaign_id)["report_md"]
        assert "repair of v1" in report or "修复" in report


def test_restart_recovery_no_duplicate_tasks_over_real_http(tmp_path: Path, mock_process: _MockProcess) -> None:
    db_path = tmp_path / "c.db"
    store = CampaignStore(db_path=db_path)
    ctrl = ResearchCampaignController(
        store,
        DsaLoopClient(base_url=mock_process.base_url),
        capabilities_provider=ready_capabilities_provider,
    )
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    for _ in range(8):
        ctrl.run_pipeline_once(campaign_id)
    candidates_before = [(c["candidate_id"], c["candidate_version"]) for c in ctrl.get_candidates(campaign_id)]
    executions_before = [(e["execution_id"], e["execution_type"]) for e in store.list_executions(campaign_id)]
    assert executions_before

    # 模拟 Vibe 重启：新 store + 新 controller，同一 db 与同一 DSA
    store2 = CampaignStore(db_path=db_path)
    ctrl2 = ResearchCampaignController(
        store2,
        DsaLoopClient(base_url=mock_process.base_url),
        capabilities_provider=ready_capabilities_provider,
    )
    _drive(ctrl2, campaign_id)

    candidates_after = [(c["candidate_id"], c["candidate_version"]) for c in ctrl2.get_candidates(campaign_id)]
    executions_after = [(e["execution_id"], e["execution_type"]) for e in store2.list_executions(campaign_id)]
    assert candidates_after == candidates_before
    for execution in executions_before:
        assert execution in executions_after
    assert ctrl2.get_campaign(campaign_id)["status"] == "completed"


def test_locked_holdout_never_reached(tmp_path: Path, mock_process: _MockProcess) -> None:
    """§7.2.5：Campaign 自动队列永远不能携带 locked-holdout 授权。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = ResearchCampaignController(
        store,
        DsaLoopClient(base_url=mock_process.base_url),
        capabilities_provider=ready_capabilities_provider,
    )
    with pytest.raises(Exception):
        ctrl.create_campaign(
            sample_campaign_request(development_window={"start_date": "2022-01-01", "end_date": "2025-12-31"})
        )
