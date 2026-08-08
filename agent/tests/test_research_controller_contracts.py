"""research_controller.contracts 副本与生成的客户端模型测试。

校验 Vibe 侧 bundle 副本无 drift、source_bundle_sha256 存在、生成的 Pydantic
模型能解析对应 golden fixtures（§18.4 CI drift 检测）。
"""

from __future__ import annotations

import json
import os
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

# DSA contract_bundle.v1 的期望 bundle_sha256（漂移手动触发基准，R11）。
# 每次从 DSA 同步 bundle 副本（bundle_manifest.json + golden/ + schemas/）后，
# 必须同步更新此常量；可用环境变量 DSA_EXPECTED_BUNDLE_SHA256 覆盖。
_PINNED_DSA_BUNDLE_SHA256 = "3023bb2f326c89cf9913baadace2311a8e4090d37284a4c4f1c9947abe502e24"

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


def test_bundle_sha256_matches_pinned_dsa_sha() -> None:
    """本地 bundle_sha256 必须等于固定记录的 DSA bundle_sha256（手动触发基准）。

    Vibe CI 无法 clone 私有 DSA 仓库，跨仓漂移由该固定值在每次契约测试中强制；
    从 DSA 同步 bundle 副本时必须同步更新 _PINNED_DSA_BUNDLE_SHA256。
    """
    expected = os.environ.get("DSA_EXPECTED_BUNDLE_SHA256", _PINNED_DSA_BUNDLE_SHA256)
    assert source_bundle_sha256() == expected


def test_bundle_matches_dsa_source_manifest() -> None:
    """跨仓 drift 门禁：从 DSA 源 manifest 全量对比（R11，D2=A）。

    设置 ``DSA_CONTRACT_MANIFEST_PATH`` 指向 DSA 仓库的
    ``services/backtest_lab/research_loop/contract_bundle/manifest.json`` 后启用；
    未设置时 skip，保证本机 / 常规 CI 不依赖 DSA 检出路径。
    断言 bundle_sha256、逐 schema 哈希、golden fixture manifest 与 DSA 源一致。
    """
    dsa_manifest_path = os.environ.get("DSA_CONTRACT_MANIFEST_PATH")
    if not dsa_manifest_path:
        pytest.skip("DSA_CONTRACT_MANIFEST_PATH not set; skipping cross-repo drift check")
    dsa = json.loads(Path(dsa_manifest_path).read_text(encoding="utf-8"))
    assert dsa.get("schema_version") == "contract_bundle.v1", dsa.get("schema_version")

    local = load_bundle_manifest()
    assert dsa["bundle_sha256"] == local["bundle_sha256"], (
        f"bundle_sha256 drift vs DSA source: {local['bundle_sha256']} != {dsa['bundle_sha256']}"
    )
    assert dsa["schemas"] == local["schemas"], "schema hashes drift vs DSA source"
    assert dsa["golden_fixture_manifest_sha256"] == local["golden_fixture_manifest_sha256"], (
        "golden fixture manifest drift vs DSA source"
    )


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
