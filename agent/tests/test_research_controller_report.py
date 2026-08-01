"""Tests for the Vibe final Chinese research report (§15.3).

The report is a pure renderer; tests feed it Mock-DSA-shaped evidence and
assert the fixed 9-section structure and hash/lineage content.
"""

from __future__ import annotations

import json
from pathlib import Path


from tests.research_loop_test_helpers import MockDsaServer, make_controller, sample_campaign_request
from src.research_controller.reporting.report import render_campaign_report, report_section_titles
from src.research_controller.store.campaign_store import CampaignStore

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "research_controller"
    / "contracts"
    / "golden"
)


def _load_golden(name: str) -> dict:
    return json.loads((_GOLDEN_DIR / name).read_text(encoding="utf-8"))


def _sample_data() -> dict:
    evidence_bundle = _load_golden("evidence_bundle.json")
    review = _load_golden("review_report.json")
    error = _load_golden("execution_error.json")
    return {
        "campaign": {
            "campaign_id": "campaign_001",
            "research_goal_id": "goal_001",
            "status": "completed",
            "current_stage": "COMPLETED",
            "config_sha256": "c" * 64,
            "protocol_bundle_sha256": "p" * 64,
            "data_snapshot_id": "data_dev_001",
            "universe_snapshot_id": "universe_pit_001",
            "gate_policy_version": "gate_champion_v1",
            "factor_inventory": {"target": 357, "completed": 120, "skipped": 2, "failed": 1},
            "queue_counts": {"pending": 0, "running": 0, "blocked": 0},
            "last_event_sequences": {"exp_momentum_volume_001": 13},
            "config": {"objective": "研究 A 股量价因子组合的横截面稳定性"},
        },
        "experiments": [
            {
                "experiment_id": "exp_momentum_volume_001",
                "round_id": "round_20260801_01",
                "objective": "验证低相关动量和成交量因子组合",
                "hypothesis": {
                    "mechanism": "动量捕捉趋势延续",
                    "prediction": "等权组合有正向超额",
                    "failure_conditions": ["rank_ic_median <= 0"],
                    "factor_ids": ["qlib158_mom_20", "gtja191_alpha099"],
                },
                "phase": "completed",
                "last_decision_action": "complete_experiment",
            }
        ],
        "candidates": [
            {
                "candidate_id": "candidate_factor_combo_001",
                "candidate_version": 2,
                "experiment_id": "exp_momentum_volume_001",
                "parent_version": 1,
                "repair_of_error_id": "err_non_finite_001",
                "status": "validated",
                "source_sha256": "a" * 64,
                "manifest": {"factor_ids": ["qlib158_mom_20", "gtja191_alpha099"], "seed": 191},
            }
        ],
        "executions": [],
        "errors": [error],
        "reviews": [review],
        "evidence": [{"evidence_id": evidence_bundle["evidence_id"], "execution_id": "exec_001",
                      "evidence": evidence_bundle}],
        "decisions": [
            {
                "decision_id": "decision_001",
                "experiment_id": "exp_momentum_volume_001",
                "round_id": "round_20260801_01",
                "action": "submit_candidate_revision",
                "rationale": "接受 DSA Reviewer 对零成交量除零错误的判断",
                "source_evidence_ids": ["evidence_001"],
                "source_review_id": "review_001",
                "next_candidate_id": "candidate_factor_combo_001",
                "next_candidate_version": 2,
            }
        ],
        "repairs": [
            {
                "candidate_id": "candidate_factor_combo_001",
                "candidate_version": 2,
                "parent_version": 1,
                "repair_of_error_id": "err_non_finite_001",
                "change_summary": "增加零值保护；将非有限值转换为 NaN",
                "preserved_constraints": ["hypothesis", "factor_ids", "data_snapshot_id", "gate_policy_version"],
            }
        ],
        "events": [],
    }


def test_report_has_all_nine_sections() -> None:
    markdown = render_campaign_report(_sample_data())
    for title in report_section_titles():
        assert title in markdown, f"missing section: {title}"


def test_report_contains_goal_and_hypothesis() -> None:
    markdown = render_campaign_report(_sample_data())
    assert "研究 A 股量价因子组合的横截面稳定性" in markdown
    assert "动量捕捉趋势延续" in markdown
    assert "rank_ic_median <= 0" in markdown


def test_report_contains_factor_and_lineage() -> None:
    markdown = render_campaign_report(_sample_data())
    assert "qlib158_mom_20" in markdown
    assert "gtja191_alpha099" in markdown
    assert "candidate_factor_combo_001" in markdown
    assert "增加零值保护；将非有限值转换为 NaN" in markdown


def test_report_contains_dsa_metrics_and_failures() -> None:
    markdown = render_campaign_report(_sample_data())
    assert "evidence_001" in markdown
    assert "coverage=pass" in markdown
    assert "CODE_RUNTIME/non_finite_signal" in markdown


def test_report_contains_reviewer_opinion_and_decision() -> None:
    markdown = render_campaign_report(_sample_data())
    assert "code_bug" in markdown
    assert "零成交量导致除数为零" in markdown
    assert "submit_candidate_revision" in markdown
    assert "接受 DSA Reviewer 对零成交量除零错误的判断" in markdown


def test_report_contains_outcomes_and_next_queue() -> None:
    markdown = render_campaign_report(_sample_data())
    assert "completed" in markdown
    assert "pending=0" in markdown
    assert "exp_momentum_volume_001" in markdown


def test_report_contains_all_hashes() -> None:
    markdown = render_campaign_report(_sample_data())
    assert f"`{'c' * 64}`" in markdown  # config_sha256
    assert f"`{'p' * 64}`" in markdown  # protocol_bundle_sha256
    assert "data_dev_001" in markdown
    assert "gate_champion_v1" in markdown
    assert f"`{'a' * 64}`" in markdown  # candidate source hash
    # artifact 哈希
    assert "artifact_metrics_001" in markdown


def test_report_empty_data_renders_gracefully() -> None:
    markdown = render_campaign_report({"campaign": {"campaign_id": "campaign_x", "status": "running"}})
    assert "campaign_x" in markdown
    for title in report_section_titles():
        assert title in markdown


def test_report_marks_missing_reviewer() -> None:
    data = _sample_data()
    data["reviews"] = []
    markdown = render_campaign_report(data)
    assert "缺少 DSA 独立评审" in markdown


def test_report_marks_review_bypassed_decision() -> None:
    """R09：超时旁路（review_bypassed）时报告头部与决定段都体现缺少独立评审。"""
    data = _sample_data()
    data["review_bypassed"] = True
    data["decisions"][0]["rationale"] = (
        "执行完成且 gate 通过；等待 DSA 独立评审超时（§9.3），接受结果（缺少 DSA 独立评审）"
    )
    markdown = render_campaign_report(data)
    assert "缺少 DSA 独立评审" in markdown
    assert "超时旁路" in markdown


def test_generate_report_via_controller(tmp_path: Path) -> None:
    server = MockDsaServer("happy_path")
    server.start()
    try:
        store = CampaignStore(db_path=tmp_path / "c.db")
        ctrl = make_controller(store, server.base_url)
        campaign_id = ctrl.create_campaign(sample_campaign_request())["campaign_id"]
        for _ in range(60):
            if ctrl.run_pipeline_once(campaign_id)["complete"]:
                break
        result = ctrl.generate_report(campaign_id)
        assert result["report_id"].startswith("report_")
        for title in report_section_titles():
            assert title in result["report_md"]
        latest = ctrl.latest_report(campaign_id)
        assert latest["report_id"] == result["report_id"]
    finally:
        server.stop()
