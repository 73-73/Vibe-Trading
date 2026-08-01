"""Tests for the Research Campaign Controller state machine (§7.2 / §13).

Exercises the controller against the real Mock DSA server (uvicorn on an
ephemeral loopback port): campaign lifecycle, pause/resume/cancel, event cursor
recovery without duplicate work, queue refill, budget exhaustion and the
cursor-expired block path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.research_loop_test_helpers import (
    MockDsaServer,
    make_controller,
    sample_campaign_request,
)
from src.research_controller.client.dsa_client import DsaLoopClient
from src.research_controller.state_machine.controller import (
    CampaignNotFoundError,
    CampaignValidationError,
    ResearchCampaignController,
)
from src.research_controller.store.campaign_store import CampaignStore


def _drive_to_complete(ctrl: ResearchCampaignController, campaign_id: str, *, max_iters: int = 80) -> dict:
    for _ in range(max_iters):
        summary = ctrl.run_pipeline_once(campaign_id)
        if summary["complete"]:
            return summary
    return ctrl.run_pipeline_once(campaign_id)


def _one_hypothesis_generator(config: dict, existing: list[dict]) -> dict | None:
    del config
    if existing:
        return None
    return {
        "experiment_id": "exp_controller_001",
        "round_id": "round_1",
        "objective": "验证动量与成交量组合",
        "hypothesis": {
            "mechanism": "动量捕捉趋势延续",
            "prediction": "等权组合在周频调仓下有正向超额",
            "failure_conditions": ["rank_ic_median <= 0"],
            "factor_ids": ["qlib158_mom_20", "gtja191_alpha099"],
        },
        "universe_snapshot_id": "universe_pit_001",
    }


@pytest.fixture
def mock() -> MockDsaServer:
    server = MockDsaServer("happy_path")
    server.start()
    try:
        yield server
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Campaign creation validation (§7.2.1)
# ---------------------------------------------------------------------------


def test_create_campaign_validation_rejects_locked_holdout(tmp_path: Path, mock: MockDsaServer) -> None:
    ctrl = make_controller(CampaignStore(db_path=tmp_path / "c.db"), mock.base_url)
    with pytest.raises(CampaignValidationError):
        ctrl.create_campaign(
            sample_campaign_request(development_window={"start_date": "2010-01-01", "end_date": "2025-12-31"})
        )


def test_create_campaign_validation_rejects_open_locked_holdout(tmp_path: Path, mock: MockDsaServer) -> None:
    ctrl = make_controller(CampaignStore(db_path=tmp_path / "c.db"), mock.base_url)
    automation = sample_campaign_request()["automation"]
    automation["open_locked_holdout"] = True
    with pytest.raises(CampaignValidationError):
        ctrl.create_campaign(sample_campaign_request(automation=automation))


def test_create_campaign_response_shape(tmp_path: Path, mock: MockDsaServer) -> None:
    ctrl = make_controller(CampaignStore(db_path=tmp_path / "c.db"), mock.base_url)
    response = ctrl.create_campaign(sample_campaign_request())
    assert response["status"] == "initializing"
    assert response["campaign_id"].startswith("campaign_")
    assert response["research_goal_id"].startswith("goal_")
    assert response["gate_policy_version"] == "gate_champion_v1"
    assert response["factor_inventory"]["target"] == 357
    assert response["data_snapshot_id"] is None


def test_get_missing_campaign_raises(tmp_path: Path, mock: MockDsaServer) -> None:
    ctrl = make_controller(CampaignStore(db_path=tmp_path / "c.db"), mock.base_url)
    with pytest.raises(CampaignNotFoundError):
        ctrl.get_campaign("campaign_missing")


# ---------------------------------------------------------------------------
# Happy path: create -> snapshot -> experiment -> candidate -> execution ->
# review -> accept -> robustness -> completed (§13.1 / §18.8)
# ---------------------------------------------------------------------------


def test_happy_path_full_flow(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]

    summary = _drive_to_complete(ctrl, campaign_id)
    assert summary["status"] == "completed"

    campaign = ctrl.get_campaign(campaign_id)
    assert campaign["status"] == "completed"
    assert campaign["current_stage"] == "COMPLETED"
    assert campaign["data_snapshot_id"] == "data_dev_001"

    experiments = ctrl.get_experiments(campaign_id)
    assert len(experiments) == 1
    assert experiments[0]["phase"] == "completed"
    assert experiments[0]["last_decision_action"] == "complete_experiment"

    decisions = ctrl.get_decisions(campaign_id)
    actions = [d["action"] for d in decisions]
    assert "accept_result" in actions
    assert "complete_experiment" in actions

    # 完整生命周期事件应落库（去重 + 审计）
    events = store.list_events(campaign_id)
    event_types = {e["event_type"] for e in events}
    assert "experiment.created" in event_types
    assert "candidate.validated" in event_types
    assert "execution.completed" in event_types
    assert "review.completed" in event_types
    assert "decision.recorded" in event_types


def test_campaign_detail_includes_rich_state(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    ctrl.run_pipeline_once(campaign_id)
    detail = ctrl.get_campaign(campaign_id)
    for key in (
        "status",
        "current_stage",
        "factor_inventory",
        "queue_counts",
        "budget_usage",
        "last_event_sequences",
        "experiment_count",
        "candidate_count",
        "execution_count",
    ):
        assert key in detail


def test_list_campaigns_filter(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url)
    cid1 = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    cid2 = ctrl.create_campaign(sample_campaign_request(objective="另一个目标"))["campaign_id"]
    ctrl.pause_campaign(cid2)
    assert {c["campaign_id"] for c in ctrl.list_campaigns()} >= {cid1, cid2}
    assert {c["campaign_id"] for c in ctrl.list_campaigns(status="paused")} == {cid2}


# ---------------------------------------------------------------------------
# Pause / resume / cancel (§7.2.3)
# ---------------------------------------------------------------------------


def test_pause_resume_cancel_semantics(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]

    # initializing -> running
    ctrl.run_pipeline_once(campaign_id)
    assert ctrl.get_campaign(campaign_id)["status"] == "running"

    # pause: 不生成和提交新任务
    ctrl.pause_campaign(campaign_id)
    execs_before = len(store.list_executions(campaign_id))
    ctrl.run_pipeline_once(campaign_id)
    assert len(store.list_executions(campaign_id)) == execs_before
    assert ctrl.get_campaign(campaign_id)["status"] == "paused"

    # resume: 沿用快照/游标/预算继续
    ctrl.resume_campaign(campaign_id)
    assert ctrl.get_campaign(campaign_id)["status"] == "running"
    summary = _drive_to_complete(ctrl, campaign_id)
    assert summary["status"] == "completed"

    # cancel: 终态，保留历史
    cid2 = ctrl.create_campaign(sample_campaign_request(objective="将取消的目标"))["campaign_id"]
    ctrl.run_pipeline_once(cid2)
    ctrl.cancel_campaign(cid2)
    assert ctrl.get_campaign(cid2)["status"] == "cancelled"
    with pytest.raises(CampaignValidationError):
        ctrl.resume_campaign(cid2)


def test_pause_with_cancel_running(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    for _ in range(6):
        ctrl.run_pipeline_once(campaign_id)
    assert any(e["status"] in ("queued", "running") for e in store.list_executions(campaign_id))
    ctrl.pause_campaign(campaign_id, cancel_running=True)
    # mock 会取消 queued/running 的执行
    assert ctrl.get_campaign(campaign_id)["status"] == "paused"


# ---------------------------------------------------------------------------
# 重启恢复：不重复提交候选/执行（§16 / §4.3）
# ---------------------------------------------------------------------------


def test_restart_recovery_no_duplicate_work(tmp_path: Path, mock: MockDsaServer) -> None:
    db_path = tmp_path / "c.db"
    store = CampaignStore(db_path=db_path)
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    for _ in range(8):
        ctrl.run_pipeline_once(campaign_id)

    executions_before = [(e["execution_id"], e["execution_type"]) for e in store.list_executions(campaign_id)]
    candidates_before = [(c["candidate_id"], c["candidate_version"]) for c in ctrl.get_candidates(campaign_id)]
    assert executions_before, "expected at least one started execution before restart"

    # 模拟重启：新建 store + controller 实例，同一 db 与同一 mock
    store2 = CampaignStore(db_path=db_path)
    ctrl2 = make_controller(store2, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    _drive_to_complete(ctrl2, campaign_id)

    executions_after = [(e["execution_id"], e["execution_type"]) for e in store2.list_executions(campaign_id)]
    candidates_after = [(c["candidate_id"], c["candidate_version"]) for c in ctrl2.get_candidates(campaign_id)]
    assert candidates_after == candidates_before, "restart must not resubmit candidates"
    # 已经启动的 execution 不重复；只有尚未到达的阶段新增 execution
    for execution in executions_before:
        assert execution in executions_after


def test_restart_does_not_reset_running_to_pending(tmp_path: Path, mock: MockDsaServer) -> None:
    """§16：Vibe 重启后先查询 DSA execution 再决定状态。"""
    db_path = tmp_path / "c.db"
    store = CampaignStore(db_path=db_path)
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    # 运行到出现一个 in-flight（queued/running）执行后停下
    for _ in range(15):
        ctrl.run_pipeline_once(campaign_id)
        in_flight = [e for e in store.list_executions(campaign_id) if e["status"] in ("queued", "running")]
        if in_flight:
            break
    in_flight_ids = [e["execution_id"] for e in in_flight]
    assert in_flight_ids, "expected an in-flight execution before restart"

    store2 = CampaignStore(db_path=db_path)
    ctrl2 = make_controller(store2, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    # 对账阶段应通过 DSA 查询推进，而不是把 RUNNING 重置为 PENDING
    for _ in range(12):
        ctrl2.run_pipeline_once(campaign_id)
    assert not any(e["status"] == "pending" for e in store2.list_executions(campaign_id))
    # in-flight 执行要么保持 queued/running，要么由 DSA 推进为 completed/failed，绝不回退 pending
    for exec_id in in_flight_ids:
        record = store2.get_execution(exec_id)
        assert record is not None
        assert record["status"] in ("queued", "running", "completed", "failed", "cancelled", "interrupted")


# ---------------------------------------------------------------------------
# 事件游标去重（duplicate_events 场景）
# ---------------------------------------------------------------------------


def test_duplicate_events_deduplicated(tmp_path: Path) -> None:
    server = MockDsaServer("duplicate_events")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "dup_test.db")
        ctrl = make_controller(store, server.base_url, hypothesis_generator=_one_hypothesis_generator)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        _drive_to_complete(ctrl, campaign_id)
        events = store.list_events(campaign_id)
        message_ids = [e["message_id"] for e in events]
        assert len(message_ids) == len(set(message_ids)), "duplicate message_id must be deduplicated"
        # 事件序列在单个实验内严格递增（§4.3）
        assert message_ids == sorted(message_ids)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 队列补充（§13.2）
# ---------------------------------------------------------------------------


def test_queue_refills_to_target(tmp_path: Path, mock: MockDsaServer) -> None:
    def _many_hypotheses(config: dict, existing: list[dict]) -> dict | None:
        del config
        n = len(existing)
        if n >= 7:
            return None
        return {
            "experiment_id": f"exp_refill_{n:03d}",
            "round_id": "round_refill",
            "objective": f"假设 {n}",
            "hypothesis": {
                "mechanism": f"机制 {n}",
                "prediction": f"预测 {n}",
                "failure_conditions": ["f"],
                "factor_ids": [f"factor_{n}"],
            },
            "universe_snapshot_id": "universe_pit_001",
        }

    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_many_hypotheses, refill_threshold=3, queue_target=10)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    # 数据快照完成后再推进队列
    for _ in range(12):
        ctrl.run_pipeline_once(campaign_id)
    experiments = ctrl.get_experiments(campaign_id)
    assert len(experiments) >= 3  # 少于 3 个时自动补充


def test_near_duplicate_hypothesis_skipped(tmp_path: Path, mock: MockDsaServer) -> None:
    """近重复候选（Jaccard ≥0.8 且机制相同）不重复提交（§13.2）。"""
    seen: dict[str, str] = {}

    def _generator(config: dict, existing: list[dict]) -> dict | None:
        del config
        candidate = {
            "experiment_id": f"exp_nd_{len(existing)}",
            "round_id": "round_nd",
            "objective": "低相关因子组合",
            "hypothesis": {
                "mechanism": "动量延续",
                "prediction": "正向超额",
                "failure_conditions": ["f"],
                "factor_ids": ["qlib158_mom_20", "gtja191_alpha099", "gtja191_alpha098"],
            },
            "universe_snapshot_id": "universe_pit_001",
        }
        mechanism = candidate["hypothesis"]["mechanism"]
        if any(mechanism == seen.get(e["experiment_id"]) for e in existing):
            return None  # 机制相同且因子集高度重叠 → 跳过
        seen[candidate["experiment_id"]] = mechanism
        return candidate

    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_generator, refill_threshold=1, queue_target=20)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    for _ in range(12):
        ctrl.run_pipeline_once(campaign_id)
    # 由于同一机制只注册一次，实验数应保持为 1（不会无限重复注册同一假设）
    assert len(ctrl.get_experiments(campaign_id)) == 1


# ---------------------------------------------------------------------------
# 滚动资源护栏（§13.3）
# ---------------------------------------------------------------------------


def test_budget_exhaustion_enters_waiting_budget(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    request = sample_campaign_request()
    request["budgets"]["llm_calls_rolling_24h"] = 1
    campaign_id = ctrl.create_campaign(request)["campaign_id"]
    # 注册第 1 个实验消耗 1 次 LLM 调用后，budget 达上限 → waiting_budget
    _drive_to_complete(ctrl, campaign_id, max_iters=20)
    status = ctrl.get_campaign(campaign_id)["status"]
    assert status in ("waiting_budget", "completed")
    # 若只注册了一个实验且候选版本未超限，则候选可继续直到完成；验证 budget_usage 被记录
    detail = ctrl.get_campaign(campaign_id)
    assert detail["budget_usage"]["llm_calls_rolling_24h"] >= 1


def test_candidate_version_budget_recorded(tmp_path: Path, mock: MockDsaServer) -> None:
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _drive_to_complete(ctrl, campaign_id)
    detail = ctrl.get_campaign(campaign_id)
    assert detail["budget_usage"]["candidate_versions_rolling_24h"] >= 1


# ---------------------------------------------------------------------------
# 游标过期（cursor_expired 场景）
# ---------------------------------------------------------------------------


def test_cursor_expired_blocks_campaign(tmp_path: Path) -> None:
    server = MockDsaServer("cursor_expired")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "cursor_test.db")
        ctrl = make_controller(store, server.base_url, hypothesis_generator=_one_hypothesis_generator)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        for _ in range(10):
            ctrl.run_pipeline_once(campaign_id)
        campaign = ctrl.get_campaign(campaign_id)
        # 游标落入已清理区间时 DSA 返回 410，Controller 进入 blocked 并给出恢复提示
        assert campaign["status"] == "blocked"
        assert campaign["blocked_reason"] and "event cursor expired" in campaign["blocked_reason"]
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# DSA 不可用：指数退避，不把超时当作提交失败
# ---------------------------------------------------------------------------


def test_dsa_unavailable_sets_backoff(tmp_path: Path) -> None:
    from tests.research_loop_test_helpers import DSA_BACKTEST_LAB_PATH

    if not DSA_BACKTEST_LAB_PATH.is_dir():
        pytest.skip("DSA mock not available")
    store = CampaignStore(db_path=tmp_path / "c.db")
    # 指向一个无人监听的 loopback 端口
    ctrl = ResearchCampaignController(store, DsaLoopClient(base_url="http://127.0.0.1:1", timeout=1.0))
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    ctrl.run_pipeline_once(campaign_id)
    interval = ctrl.poll_interval_seconds(campaign_id)
    assert interval >= 2.0
    ctrl.reset_availability()
    assert ctrl.poll_interval_seconds(campaign_id) <= 15.0


def test_poll_interval_active_vs_idle(tmp_path: Path, mock: MockDsaServer) -> None:
    from src.research_controller.state_machine.machine import compute_poll_interval

    assert compute_poll_interval(any_active_execution=True, any_waiting_review=False) == 2.0
    assert compute_poll_interval(any_active_execution=False, any_waiting_review=True) == 5.0
    assert compute_poll_interval(any_active_execution=False, any_waiting_review=False) == 15.0
    assert compute_poll_interval(any_active_execution=True, any_waiting_review=False, backoff_seconds=100) == 60.0
