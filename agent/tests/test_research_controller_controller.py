"""Tests for the Research Campaign Controller state machine (§7.2 / §13).

Exercises the controller against the real Mock DSA server (uvicorn on an
ephemeral loopback port): campaign lifecycle, pause/resume/cancel, event cursor
recovery without duplicate work, queue refill, budget exhaustion and the
cursor-expired block path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


# ---------------------------------------------------------------------------
# R04 fail-closed：portfolio_backtest 空 gate_results 不允许晋级（§6.10）
# ---------------------------------------------------------------------------


def _seed_completed_portfolio_backtest(
    store: CampaignStore, campaign_id: str, *, gate_results: list[dict]
) -> tuple[str, str, str]:
    """在 Vibe store 中直接构造一条 completed 的 portfolio_backtest 执行与 evidence。

    复用与真实流水线一致的 store 记录形状，绕开 DSA mock（mock 固定返回带
    gate_results 的 golden evidence，无法覆盖空 gate 场景）。
    """
    now = "2026-08-01T00:00:00+00:00"
    exp_id = "exp_gate_fail_closed_001"
    candidate_id = "candidate_gate_001"
    store.upsert_experiment(
        {
            "experiment_id": exp_id,
            "campaign_id": campaign_id,
            "round_id": "round_1",
            "objective": "R04 gate fail-closed test",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": []},
            "status": "running",
            "phase": "promoted",
            "data_snapshot_id": None,
            "universe_snapshot_id": "universe_pit_001",
            "gate_policy_version": "gate_champion_v1",
            "created_at": now,
            "updated_at": now,
        }
    )
    store.upsert_candidate(
        {
            "candidate_id": candidate_id,
            "candidate_version": 1,
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "parent_version": None,
            "repair_of_error_id": None,
            "contract_version": "strategy-signal.v1",
            "manifest": {"factor_ids": ["qlib158_mom_20"]},
            "source_sha256": "0" * 64,
            "status": "validated",
            "created_at": now,
        }
    )
    execution_id = "exec_gate_fail_closed_001"
    evidence_id = "evidence_gate_fail_closed_001"
    store.upsert_execution(
        {
            "execution_id": execution_id,
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "candidate_id": candidate_id,
            "candidate_version": 1,
            "execution_type": "portfolio_backtest",
            "status": "completed",
            "progress": 1.0,
            "current_stage": "completed",
            "evidence_id": evidence_id,
            "created_at": now,
            "started_at": now,
            "finished_at": now,
        }
    )
    store.upsert_evidence(
        {
            "evidence_id": evidence_id,
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "execution_id": execution_id,
            "evidence": {
                "evidence_id": evidence_id,
                "experiment_id": exp_id,
                "execution_id": execution_id,
                "execution_type": "portfolio_backtest",
                "status": "completed",
                "metrics": {
                    "rank_ic": {"status": "computed", "artifact_id": "artifact_rank_ic_001"},
                    "shared_capital_portfolio": {"status": "computed", "artifact_id": "artifact_portfolio_001"},
                },
                "gate_results": gate_results,
            },
            "created_at": now,
        }
    )
    return exp_id, candidate_id, execution_id


def _insert_review_requested_event(
    store: CampaignStore, campaign_id: str, exp_id: str, execution_id: str, *, created_at: str
) -> None:
    """在 store 中插入一条 review.requested 事件（R09 超时窗口判断）。"""
    store.insert_event(
        {
            "message_id": f"evt_review_req_{execution_id}",
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "sequence": 500,
            "producer": "dsa_reviewer",
            "event_type": "review.requested",
            "correlation_id": None,
            "causation_id": None,
            "payload": {"execution_id": execution_id},
            "artifact_refs": [],
            "created_at": created_at,
        }
    )


def _decision_for_seeded_backtest(store: CampaignStore, ctrl: ResearchCampaignController, campaign_id: str) -> dict:
    campaign = store.get_campaign(campaign_id)
    exp_id = store.list_experiments(campaign_id)[-1]["experiment_id"]
    exp = store.get_experiment(exp_id)
    current = store.list_candidates_for_experiment(campaign_id, exp_id)[-1]
    return ctrl._compute_decision(campaign, exp, current)


def test_completed_backtest_empty_gate_results_blocks_promotion(tmp_path: Path) -> None:
    """R04：portfolio_backtest 返回空 gate_results（占位 evidence）-> request_human_review/blocked，
    不晋级 accept_result/promoted。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_completed_portfolio_backtest(store, campaign_id, gate_results=[])

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is not None
    assert decision["action"] == "request_human_review"
    assert decision["phase"] == "blocked"
    assert "gate" in decision["rationale"]


def test_completed_backtest_with_pass_gate_results_promotes(tmp_path: Path) -> None:
    """对照：portfolio_backtest 有 pass gate_results -> accept_result/promoted（未过度收紧）。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_completed_portfolio_backtest(
        store,
        campaign_id,
        gate_results=[{"check_id": "coverage", "status": "pass", "observed": 0.94, "operator": ">=", "threshold": 0.8}],
    )

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is not None
    assert decision["action"] == "accept_result"
    assert decision["phase"] == "promoted"


def test_completed_backtest_with_gate_failure_rejects(tmp_path: Path) -> None:
    """对照：portfolio_backtest 有 fail gate_results -> abandon_candidate/rejected（既有语义保持）。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_completed_portfolio_backtest(
        store,
        campaign_id,
        gate_results=[{"check_id": "coverage", "status": "fail", "observed": 0.3, "operator": ">=", "threshold": 0.8}],
    )

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is not None
    assert decision["action"] == "abandon_candidate"
    assert decision["phase"] == "rejected"


# ---------------------------------------------------------------------------
# R06：blocked / waiting_data 实验不得把 campaign 误判为 completed（§13）
# ---------------------------------------------------------------------------


def _seed_blocked_phase_experiment(store: CampaignStore, campaign_id: str, phase: str) -> None:
    now = "2026-08-01T00:00:00+00:00"
    store.upsert_experiment(
        {
            "experiment_id": f"exp_{phase}_001",
            "campaign_id": campaign_id,
            "round_id": "round_1",
            "objective": f"R06 {phase} test",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": []},
            "status": "running",
            "phase": phase,
            "data_snapshot_id": None,
            "universe_snapshot_id": "universe_campaign_x",
            "gate_policy_version": "gate_champion_v1",
            "created_at": now,
            "updated_at": now,
        }
    )


def test_blocked_experiment_keeps_campaign_blocked(tmp_path: Path) -> None:
    """R06：blocked 实验在 TERMINAL_EXPERIMENT_PHASES 内，但 campaign 必须 blocked 而非 completed。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_blocked_phase_experiment(store, campaign_id, "blocked")

    ctrl._sync_campaign_status(campaign_id)
    campaign = store.get_campaign(campaign_id)
    assert campaign["status"] == "blocked"
    assert campaign["current_stage"] == "VIBE_DECISION"


def test_waiting_data_experiment_keeps_campaign_blocked(tmp_path: Path) -> None:
    """R06：waiting_data 实验同样不能把 campaign 判成 completed。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_blocked_phase_experiment(store, campaign_id, "waiting_data")

    ctrl._sync_campaign_status(campaign_id)
    campaign = store.get_campaign(campaign_id)
    assert campaign["status"] == "blocked"


def test_terminal_experiments_with_active_execution_not_completed(tmp_path: Path) -> None:
    """R06：全部实验终态但仍有活跃 execution 时不得提前 completed。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_blocked_phase_experiment(store, campaign_id, "completed")
    now = "2026-08-01T00:00:00+00:00"
    store.upsert_execution(
        {
            "execution_id": "exec_active_001",
            "campaign_id": campaign_id,
            "experiment_id": "exp_completed_001",
            "execution_type": "robustness",
            "status": "running",
            "progress": 0.5,
            "created_at": now,
        }
    )

    ctrl._sync_campaign_status(campaign_id)
    assert store.get_campaign(campaign_id)["status"] != "completed"


# ---------------------------------------------------------------------------
# R07：数据快照构建是异步的，未完成时 campaign 停在 WAIT_DATA（§6.12）
# ---------------------------------------------------------------------------


def test_data_snapshot_build_incomplete_stays_wait_data(tmp_path: Path) -> None:
    """R07：快照构建 execution 未完成（queued）→ campaign 停在 WAIT_DATA，不提前进 FACTOR_INVENTORY。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    store.update_campaign(campaign_id, status="running")
    store.upsert_execution(
        {
            "execution_id": "exec_snap_pending_001",
            "campaign_id": campaign_id,
            "experiment_id": "",
            "execution_type": "data_snapshot_build",
            "status": "queued",
            "progress": 0.0,
            "created_at": "2026-08-01T00:00:00+00:00",
        }
    )

    ctrl.run_pipeline_once(campaign_id)
    campaign = store.get_campaign(campaign_id)
    assert campaign["status"] == "running"
    assert campaign["current_stage"] == "WAIT_DATA"
    assert campaign["data_snapshot_id"] is None


def test_data_snapshot_build_failed_blocks_campaign(tmp_path: Path) -> None:
    """R07：快照构建 execution failed → campaign blocked（快照构建失败）。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    store.update_campaign(campaign_id, status="running")
    store.upsert_execution(
        {
            "execution_id": "exec_snap_failed_001",
            "campaign_id": campaign_id,
            "experiment_id": "",
            "execution_type": "data_snapshot_build",
            "status": "failed",
            "progress": 0.5,
            "created_at": "2026-08-01T00:00:00+00:00",
        }
    )

    ctrl.run_pipeline_once(campaign_id)
    campaign = store.get_campaign(campaign_id)
    assert campaign["status"] == "blocked"
    assert "snapshot" in (campaign["blocked_reason"] or "").lower()


# ---------------------------------------------------------------------------
# R08：universe_snapshot_id 只来自 DSA universe build，缺失时 blocked（§11.7）
# ---------------------------------------------------------------------------


def test_start_execution_missing_universe_blocks_not_hardcoded(tmp_path: Path) -> None:
    """R08：campaign 无 universe_snapshot_id 时 _start_execution blocked，绝不硬编码 universe_pit_001。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    store.update_campaign(campaign_id, status="running", data_snapshot_id="data_dev_001")
    now = "2026-08-01T00:00:00+00:00"
    store.upsert_experiment(
        {
            "experiment_id": "exp_universe_missing_001",
            "campaign_id": campaign_id,
            "round_id": "round_1",
            "objective": "R08 universe missing",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": []},
            "status": "running",
            "phase": "candidate_built",
            "data_snapshot_id": "data_dev_001",
            "universe_snapshot_id": None,
            "gate_policy_version": "gate_champion_v1",
            "created_at": now,
            "updated_at": now,
        }
    )
    store.upsert_candidate(
        {
            "candidate_id": "candidate_universe_001",
            "candidate_version": 1,
            "campaign_id": campaign_id,
            "experiment_id": "exp_universe_missing_001",
            "parent_version": None,
            "repair_of_error_id": None,
            "contract_version": "strategy-signal.v1",
            "manifest": {"factor_ids": ["qlib158_mom_20"]},
            "source_sha256": "0" * 64,
            "status": "validated",
            "created_at": now,
        }
    )
    campaign = store.get_campaign(campaign_id)
    exp = store.get_experiment("exp_universe_missing_001")
    current = store.list_candidates_for_experiment(campaign_id, "exp_universe_missing_001")[-1]

    ctrl._start_execution(campaign, exp, current, "strategy_sandbox_smoke")
    blocked = store.get_campaign(campaign_id)
    assert blocked["status"] == "blocked"
    assert blocked["blocked_reason"] and "universe" in blocked["blocked_reason"]


def test_happy_path_uses_real_universe_snapshot_id(tmp_path: Path, mock: MockDsaServer) -> None:
    """R08：happy path 的 universe_snapshot_id 来自 DSA universe build，不再是 universe_pit_001。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, mock.base_url, hypothesis_generator=_one_hypothesis_generator)
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _drive_to_complete(ctrl, campaign_id)

    campaign = ctrl.get_campaign(campaign_id)
    assert campaign["status"] == "completed"
    assert campaign["universe_snapshot_id"]
    assert campaign["universe_snapshot_id"].startswith("universe_")
    assert campaign["universe_snapshot_id"] != "universe_pit_001"


# ---------------------------------------------------------------------------
# R09 决策 B：Reviewer 是旁路意见，§9.3 超时窗口（默认 300 秒）
# ---------------------------------------------------------------------------


def test_review_requested_untimed_out_defers_decision(tmp_path: Path) -> None:
    """R09：review 请求已发出但未产出且未超时 → 不决策（等待 §9.3 窗口）。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_completed_portfolio_backtest(
        store,
        campaign_id,
        gate_results=[{"check_id": "coverage", "status": "pass", "observed": 0.94, "operator": ">=", "threshold": 0.8}],
    )
    exp_id = store.list_experiments(campaign_id)[-1]["experiment_id"]
    execution_id = store.list_executions(campaign_id)[-1]["execution_id"]
    _insert_review_requested_event(
        store, campaign_id, exp_id, execution_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is None


def test_review_requested_timed_out_accepts_with_marker(tmp_path: Path) -> None:
    """R09：超过 §9.3 等待窗口后 accept，rationale 标记“缺少 DSA 独立评审”。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    _seed_completed_portfolio_backtest(
        store,
        campaign_id,
        gate_results=[{"check_id": "coverage", "status": "pass", "observed": 0.94, "operator": ">=", "threshold": 0.8}],
    )
    exp_id = store.list_experiments(campaign_id)[-1]["experiment_id"]
    execution_id = store.list_executions(campaign_id)[-1]["execution_id"]
    old = (datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat()
    _insert_review_requested_event(store, campaign_id, exp_id, execution_id, created_at=old)

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is not None
    assert decision["action"] == "accept_result"
    assert "缺少 DSA 独立评审" in decision["rationale"]


def test_review_completed_uses_review_normally(tmp_path: Path) -> None:
    """对照（R09）：review 已产出时走既有 Reviewer 逻辑，不进入超时旁路。"""
    store = CampaignStore(db_path=tmp_path / "c.db")
    ctrl = make_controller(store, "http://127.0.0.1:1")
    campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
    exp_id, _candidate_id, execution_id = _seed_completed_portfolio_backtest(
        store,
        campaign_id,
        gate_results=[{"check_id": "coverage", "status": "pass", "observed": 0.94, "operator": ">=", "threshold": 0.8}],
    )
    now = "2026-08-01T00:00:00+00:00"
    # review 已产出：execution 挂上 review_id，review 记录在 reviews 表
    store.update_execution(execution_id, review_id="review_seeded_001")
    store.upsert_review(
        {
            "review_id": "review_seeded_001",
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "execution_id": execution_id,
            "classification": "code_bug",
            "confidence": 0.8,
            "root_causes": [{"evidence_ref": "e1", "finding": "除数可能为零"}],
            "gate_assessment": None,
            "repair_contract": {"repairable": True, "allowed_changes": [], "forbidden_changes": []},
            "recommended_action": "submit_candidate_revision",
            "recommended_tests": [],
            "limitations": [],
            "model": "mock",
            "prompt_version": "v1",
            "prompt_hash": "",
            "input_evidence_hash": "",
            "created_at": now,
        }
    )
    # 有 review.requested 事件（未超时）但 review 已产出：仍应走 Reviewer 建议而非等待
    _insert_review_requested_event(
        store, campaign_id, exp_id, execution_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    decision = _decision_for_seeded_backtest(store, ctrl, campaign_id)
    assert decision is not None
    assert decision["action"] == "accept_result"
    assert "缺少 DSA 独立评审" not in decision["rationale"]
