"""Tests for the automatic repair lineage (§8.3) and the error/repair loop.

Unit tests cover the pure repair guards; end-to-end tests drive the
``execution_failure`` and ``candidate_static_failure`` Mock DSA scenarios to
demonstrate "DSA 报错 -> DSA Reviewer 评审 -> Vibe 决定 -> 新版本提交 / 提前停止".
"""

from __future__ import annotations

from pathlib import Path

from tests.research_loop_test_helpers import MockDsaServer, make_controller, sample_campaign_request
from src.research_controller.repair.lineage import (
    can_repair,
    consecutive_fingerprint_count,
    next_candidate_version,
    repair_budget_remaining,
    source_hash_used,
)
from src.research_controller.state_machine.controller import ResearchCampaignController
from src.research_controller.store.campaign_store import CampaignStore

_FP = "a" * 64
_FP2 = "b" * 64


def _repair(version: int, fingerprint: str = _FP) -> dict:
    return {
        "candidate_id": "c1",
        "candidate_version": version,
        "parent_version": version - 1,
        "repair_of_error_id": f"err_{version}",
        "error_fingerprint": fingerprint,
        "change_summary": "x",
        "preserved_constraints": [],
    }


def _candidate(version: int, sha: str = "s1") -> dict:
    return {"candidate_id": "c1", "candidate_version": version, "source_sha256": sha}


# ---------------------------------------------------------------------------
# 修复谱系纯函数（§8.3）
# ---------------------------------------------------------------------------


def test_repair_budget_max_three_versions() -> None:
    assert repair_budget_remaining([]) == 3
    assert repair_budget_remaining([_repair(2), _repair(3)]) == 1
    assert repair_budget_remaining([_repair(2), _repair(3), _repair(4)]) == 0


def test_can_repair_respects_max_versions() -> None:
    repairs = [_repair(2), _repair(3), _repair(4)]
    allowed, reason = can_repair(_FP, repairs, [_candidate(1), _candidate(2), _candidate(3), _candidate(4)], "newsha")
    assert allowed is False
    assert reason == "repair_exhausted"


def test_can_repair_stops_on_consecutive_same_fingerprint() -> None:
    # 相同错误指纹连续出现 2 次 → 提前停止
    repairs = [_repair(2, _FP)]
    allowed, reason = can_repair(_FP, repairs, [_candidate(1), _candidate(2)], "newsha")
    assert allowed is False
    assert reason == "consecutive_fingerprint_stop"
    # 不同指纹仍可修复
    allowed, _ = can_repair(_FP2, repairs, [_candidate(1), _candidate(2)], "newsha")
    assert allowed is True


def test_consecutive_fingerprint_count() -> None:
    assert consecutive_fingerprint_count(_FP, []) == 1  # 当前失败即第 1 次
    assert consecutive_fingerprint_count(_FP, [_repair(2, _FP)]) == 2
    assert consecutive_fingerprint_count(_FP2, [_repair(2, _FP)]) == 1


def test_source_hash_duplicate_rejected() -> None:
    candidates = [_candidate(1, "sha_v1"), _candidate(2, "sha_v2")]
    assert source_hash_used(candidates, "sha_v1") is True
    assert source_hash_used(candidates, "sha_v3") is False
    # 当前错误指纹与已有修复不同，避免 consecutive 提前触发；只验证 source_hash 去重
    allowed, reason = can_repair(_FP, [_repair(2, _FP2)], candidates, "sha_v1")
    assert allowed is False
    assert reason == "source_hash_duplicate"


def test_next_candidate_version() -> None:
    assert next_candidate_version([]) == 1
    assert next_candidate_version([_candidate(1), _candidate(2)]) == 3


# ---------------------------------------------------------------------------
# 端到端修复闭环（execution_failure 场景）
# ---------------------------------------------------------------------------


def _drive(ctrl: ResearchCampaignController, campaign_id: str, max_iters: int = 80) -> dict:
    for _ in range(max_iters):
        summary = ctrl.run_pipeline_once(campaign_id)
        if summary["complete"]:
            return summary
    return ctrl.run_pipeline_once(campaign_id)


def test_execution_failure_repair_lineage(tmp_path: Path) -> None:
    server = MockDsaServer("execution_failure")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "repair_exec.db")
        ctrl = make_controller(store, server.base_url)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        _drive(ctrl, campaign_id)

        candidates = ctrl.get_candidates(campaign_id)
        versions = [c["candidate_version"] for c in candidates]
        assert versions == [1, 2], "v1 失败后应提交 v2 修复版本"
        v2 = [c for c in candidates if c["candidate_version"] == 2][0]
        assert v2["parent_version"] == 1
        assert v2["repair_of_error_id"]

        repairs = ctrl.get_repairs(campaign_id)
        assert len(repairs) == 1
        assert repairs[0]["parent_version"] == 1
        assert "零值" in repairs[0]["change_summary"] or repairs[0]["change_summary"]
        assert "hypothesis" in repairs[0]["preserved_constraints"]

        # 相同错误指纹连续 2 次 → 提前停止修复
        decisions = ctrl.get_decisions(campaign_id)
        actions = [d["action"] for d in decisions]
        assert "submit_candidate_revision" in actions
        assert actions[-1] == "abandon_candidate"

        experiments = ctrl.get_experiments(campaign_id)
        assert experiments[0]["phase"] == "rejected"
        assert ctrl.get_campaign(campaign_id)["status"] == "completed"

        # DSA 错误已规范化落库
        errors = store.list_errors(campaign_id)
        assert any(e["category"] == "CODE_RUNTIME" and e["code"] == "non_finite_signal" for e in errors)
        # Reviewer 评审已落库（code_bug / submit_candidate_revision）
        reviews = store.list_reviews(campaign_id)
        assert any(r["classification"] == "code_bug" and r["recommended_action"] == "submit_candidate_revision" for r in reviews)
    finally:
        server.stop()


def test_static_failure_repair_lineage(tmp_path: Path) -> None:
    server = MockDsaServer("candidate_static_failure")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "repair_static.db")
        ctrl = make_controller(store, server.base_url)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        _drive(ctrl, campaign_id)

        candidates = ctrl.get_candidates(campaign_id)
        assert [c["candidate_version"] for c in candidates] == [1, 2]
        errors = store.list_errors(campaign_id)
        assert any(e["category"] == "CONTRACT_ERROR" and e["code"] == "python_syntax_error" for e in errors)
        assert ctrl.get_campaign(campaign_id)["status"] == "completed"
    finally:
        server.stop()


def test_reviewer_failure_marks_missing_review(tmp_path: Path) -> None:
    server = MockDsaServer("reviewer_failure")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "repair_review.db")
        ctrl = make_controller(store, server.base_url)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        _drive(ctrl, campaign_id)
        # 无 review.completed；报告应标记缺少 DSA 独立评审
        assert store.list_reviews(campaign_id) == []
        report = ctrl.generate_report(campaign_id)["report_md"]
        assert "缺少 DSA 独立评审" in report
    finally:
        server.stop()
