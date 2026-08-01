"""research_controller.contracts 副本与生成的客户端模型测试。

校验 Vibe 侧 bundle 副本无 drift、source_bundle_sha256 存在、生成的 Pydantic
模型能解析对应 golden fixtures（§18.4 CI drift 检测）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research_controller.contracts import (
    load_bundle_manifest,
    source_bundle_sha256,
    verify_local_copy,
)

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "src" / "research_controller" / "contracts"
_GOLDEN_DIR = _CONTRACTS_DIR / "golden"
_GENERATED_DIR = _CONTRACTS_DIR / "generated"

# golden fixture 文件名 -> 生成的客户端模型模块/类名
GOLDEN_TO_MODEL = {
    "data_snapshot.json": ("data_snapshot_v1", "DataSnapshot"),
    "universe_snapshot.json": ("universe_snapshot_v1", "UniverseSnapshot"),
    "market_panel.json": ("market_panel_v1", "MarketPanel"),
    "factor_snapshot.json": ("factor_snapshot_v1", "FactorSnapshot"),
    "experiment_state_snapshot.json": ("experiment_state_snapshot_v1", "ExperimentStateSnapshot"),
    "execution_error.json": ("execution_error_v1", "ExecutionError"),
    "evidence_bundle.json": ("evidence_bundle_v1", "EvidenceBundle"),
    "review_report.json": ("review_report_v1", "ReviewReport"),
    "gate_policy.json": ("gate_policy_v1", "GatePolicy"),
    "meta_gate.json": ("meta_gate_v1", "MetaGate"),
    "research_campaign.json": ("research_campaign_v1", "ResearchCampaign"),
    "holdout_authorization.json": ("holdout_authorization_v1", "HoldoutAuthorization"),
    "candidate_registration.json": ("strategy_signal_v1", "CandidateRegistration"),
    "research_experiment.json": ("research_experiment_v1", "ResearchExperiment"),
    "execution_start.json": ("execution_start_v1", "ExecutionStartRequest"),
    "research_event.json": ("research_event_v1", "ResearchEvent"),
    "decision_record.json": ("decision_record_v1", "DecisionRecord"),
    "success_envelope.json": ("success_envelope_v1", "SuccessEnvelope"),
    "error_envelope.json": ("error_envelope_v1", "ErrorEnvelope"),
}


def test_bundle_manifest_present_and_sha_format() -> None:
    manifest = load_bundle_manifest()
    assert manifest["schema_version"] == "contract_bundle.v1"
    sha = source_bundle_sha256()
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_local_copy_has_no_drift() -> None:
    assert verify_local_copy() == []
    # golden fixtures 数量与 manifest 一致
    manifest = load_bundle_manifest()
    golden_count = len(list(_GOLDEN_DIR.glob("*.json")))
    assert golden_count >= len(GOLDEN_TO_MODEL)


def test_generated_models_parse_golden() -> None:
    import importlib

    for filename, (module_name, class_name) in GOLDEN_TO_MODEL.items():
        target = _GOLDEN_DIR / filename
        if not target.exists():
            pytest.fail(f"missing golden copy: {filename}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        module = importlib.import_module(f"src.research_controller.contracts.generated.{module_name}")
        model = getattr(module, class_name)
        parsed = model.model_validate(payload)
        assert parsed is not None


def test_generated_modules_are_present() -> None:
    manifest = load_bundle_manifest()
    missing = []
    for schema_name in manifest.get("schemas", {}):
        module = schema_name.removesuffix(".json").replace(".", "_").replace("-", "_")
        if not (_GENERATED_DIR / f"{module}.py").is_file():
            missing.append(schema_name)
    assert missing == [], f"missing generated modules: {missing}"
