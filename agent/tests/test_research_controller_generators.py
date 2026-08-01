"""Production Agent generation tests for §13 / P1-04.

All provider objects here are explicit deterministic test seams.  Production
assembly itself is verified to require ``verify_online`` before returning a
controller and has no fallback generator.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.research_controller.campaign_api.assemble import _build_production_controller
from src.research_controller.generation.production import (
    GenerationProtocolError,
    GenerationUnavailableError,
    ProductionResearchGenerators,
    validate_strategy_source,
)

_DSA_BACKTEST_LAB = Path("/Users/pandeng/Projects/daily_stock_analysis/services/backtest_lab")
_FACTORS = ["gtja191_099", "qlib158_rank20"]


def _tool_response(name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        tool_calls=[SimpleNamespace(name=name, arguments=arguments)],
    )


def _manifest() -> dict[str, Any]:
    return {
        "factor_ids": list(_FACTORS),
        "parameters": {
            "score_scale": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 2.0,
                "default": 1.0,
            }
        },
        "default_parameters": {"score_scale": 1.0},
        "direction": "higher_is_better",
        "rebalance_frequency": "weekly",
        "top_n": 20,
        "max_holdings": 20,
        "exit_policy": "leave_top_n",
        "seed": 191,
    }


def _source(version: int) -> str:
    operator = "rank" if version == 1 else "zscore"
    return (
        "def generate_scores(factors, params):\n"
        f"    # generated candidate v{version}\n"
        f'    first = {operator}(factors["gtja191_099"])\n'
        f'    second = {operator}(factors["qlib158_rank20"])\n'
        "    return (first + second) * params.get(\"score_scale\", 1.0)\n"
    )


class _ScriptedLLM:
    def __init__(self) -> None:
        self.batch_index = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, timeout=None):  # noqa: ANN001,ANN201 - provider seam
        name = tools[0]["function"]["name"]
        self.calls.append({"name": name, "messages": messages, "timeout": timeout})
        if name == "research_generator_readiness":
            nonce = re.search(r"[0-9a-f]{32}", messages[-1]["content"]).group(0)
            return _tool_response(name, {"nonce": nonce})
        if name == "submit_research_hypotheses":
            start = self.batch_index * 6
            self.batch_index += 1
            return _tool_response(
                name,
                {
                    "hypotheses": [
                        {
                            "mechanism": f"unique_mechanism_token_{start + index}_captures_distinct_market_behaviour",
                            "prediction": f"unique_prediction_token_{start + index}_fails_under_declared_gate",
                            "failure_conditions": ["rank_ic_median <= 0"],
                            "factor_ids": list(_FACTORS),
                            "tags": [f"batch-{self.batch_index}", f"item-{index}"],
                        }
                        for index in range(6)
                    ]
                },
            )
        if name == "submit_strategy_candidate":
            context = json.loads(messages[-1]["content"])
            version = int(context["candidate_version"])
            return _tool_response(
                name,
                {"source_code": _source(version), "manifest": _manifest()},
            )
        raise AssertionError(name)


def _config() -> dict[str, Any]:
    return {
        "objective": "研究 A 股量价因子组合的横截面稳定性",
        "factor_scope": "a_share_native_357",
        "development_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
    }


def _experiment() -> dict[str, Any]:
    return {
        "experiment_id": "exp_generator_001",
        "hypothesis": {
            "mechanism": "A testable mechanism using two registered factors",
            "prediction": "The cross-sectional score must pass declared gates",
            "failure_conditions": ["rank_ic_median <= 0"],
            "factor_ids": list(_FACTORS),
        },
    }


def _assert_dsa_static_ok(source: str) -> None:
    if not _DSA_BACKTEST_LAB.is_dir():
        pytest.skip("DSA checkout unavailable for cross-repo static validation")
    if str(_DSA_BACKTEST_LAB) not in sys.path:
        sys.path.insert(0, str(_DSA_BACKTEST_LAB))
    from research_loop.sandbox.ast_checker import check_strategy_source

    result = check_strategy_source(source, manifest_factor_ids=set(_FACTORS))
    assert result.ok is True, result.to_dict()


def test_registry_scope_is_authoritative_357_real_ids() -> None:
    generator = ProductionResearchGenerators(_ScriptedLLM())
    assert len(generator._factor_ids) == 357
    assert "gtja191_099" in generator._factor_ids
    assert "qlib158_rank20" in generator._factor_ids
    assert "gtja191_alpha099" not in generator._factor_ids
    assert "qlib158_RANK_20" not in generator._factor_ids


def test_hypothesis_batches_are_three_to_ten_and_refill_twenty_unique_items() -> None:
    llm = _ScriptedLLM()
    usage: list[str] = []
    generator = ProductionResearchGenerators(llm, usage_recorder=usage.append)
    existing: list[dict[str, Any]] = []
    for _ in range(20):
        item = generator.hypothesis_generator(_config(), existing)
        assert item is not None
        existing.append(item)

    signatures = {
        (
            item["hypothesis"]["mechanism"],
            item["hypothesis"]["prediction"],
            tuple(item["hypothesis"]["factor_ids"]),
        )
        for item in existing
    }
    assert len(signatures) == 20
    batch_calls = [call for call in llm.calls if call["name"] == "submit_research_hypotheses"]
    assert len(batch_calls) == 4
    assert usage == ["llm_calls"] * 4
    assert all(item["hypothesis"]["factor_ids"] == _FACTORS for item in existing)


def test_hypothesis_batch_buffer_never_spills_between_campaign_lists() -> None:
    llm = _ScriptedLLM()
    generator = ProductionResearchGenerators(llm)
    campaign_a: list[dict[str, Any]] = []
    campaign_b: list[dict[str, Any]] = []

    first_a = generator.hypothesis_generator(_config(), campaign_a)
    first_b = generator.hypothesis_generator(_config(), campaign_b)
    second_a = generator.hypothesis_generator(_config(), campaign_a)

    assert first_a is not None and "token_0_" in first_a["hypothesis"]["mechanism"]
    assert first_b is not None and "token_6_" in first_b["hypothesis"]["mechanism"]
    assert second_a is not None and "token_1_" in second_a["hypothesis"]["mechanism"]


class _DuplicateThenUniqueLLM(_ScriptedLLM):
    def chat(self, messages, tools=None, timeout=None):  # noqa: ANN001,ANN201
        name = tools[0]["function"]["name"]
        if name != "submit_research_hypotheses":
            return super().chat(messages, tools=tools, timeout=timeout)
        self.calls.append({"name": name, "messages": messages, "timeout": timeout})
        suffix = len(self.calls)
        duplicate = {
            "mechanism": "existing_mechanism_token_for_near_duplicate_check",
            "prediction": "existing_prediction_token_for_near_duplicate_check",
            "failure_conditions": ["rank_ic_median <= 0"],
            "factor_ids": list(_FACTORS),
            "tags": ["duplicate"],
        }
        unique = [
            {
                "mechanism": f"fresh_mechanism_{suffix}_{index}_with_independent_economic_path",
                "prediction": f"fresh_prediction_{suffix}_{index}_with_distinct_gate",
                "failure_conditions": ["rank_ic_median <= 0"],
                "factor_ids": list(_FACTORS),
                "tags": ["fresh"],
            }
            for index in range(2)
        ]
        return _tool_response(name, {"hypotheses": [duplicate, *unique]})


def test_near_duplicates_are_removed_and_provider_is_reasked_for_minimum_batch() -> None:
    llm = _DuplicateThenUniqueLLM()
    generator = ProductionResearchGenerators(llm)
    existing = [
        {
            "experiment_id": "exp_existing",
            "hypothesis": {
                "mechanism": "existing_mechanism_token_for_near_duplicate_check",
                "prediction": "existing_prediction_token_for_near_duplicate_check",
                "failure_conditions": ["rank_ic_median <= 0"],
                "factor_ids": list(_FACTORS),
            },
        }
    ]
    first = generator.hypothesis_generator(_config(), existing)
    assert first is not None
    assert "fresh_" in first["hypothesis"]["mechanism"]
    batch_calls = [call for call in llm.calls if call["name"] == "submit_research_hypotheses"]
    assert len(batch_calls) == 2


def test_initial_and_repair_sources_pass_real_dsa_static_check_and_repair_uses_context() -> None:
    llm = _ScriptedLLM()
    generator = ProductionResearchGenerators(llm)
    source_v1, manifest_v1 = generator.code_generator(
        _experiment(),
        "candidate_generator_001",
        1,
    )
    error = {
        "error_id": "err_divzero_001",
        "code": "division_by_zero",
        "error_fingerprint": "a" * 64,
        "message": "non-finite strategy score",
    }
    review = {
        "review_id": "review_001",
        "classification": "code_bug",
        "repair_contract": {
            "allowed_changes": ["use safe_div and zero guard"],
            "must_preserve": ["hypothesis", "factor_ids"],
        },
    }
    source_v2, manifest_v2 = generator.code_generator(
        _experiment(),
        "candidate_generator_001",
        2,
        parent_version=1,
        error=error,
        review=review,
    )

    assert source_v1 != source_v2
    assert manifest_v1["factor_ids"] == manifest_v2["factor_ids"] == _FACTORS
    validate_strategy_source(source_v1, set(_FACTORS))
    validate_strategy_source(source_v2, set(_FACTORS))
    _assert_dsa_static_ok(source_v1)
    _assert_dsa_static_ok(source_v2)
    repair_call = [call for call in llm.calls if call["name"] == "submit_strategy_candidate"][-1]
    repair_prompt = repair_call["messages"][-1]["content"]
    assert "division_by_zero" in repair_prompt
    assert "use safe_div and zero guard" in repair_prompt
    assert "must_preserve" in repair_prompt


def test_repair_without_error_and_review_fails_closed_before_provider_call() -> None:
    llm = _ScriptedLLM()
    generator = ProductionResearchGenerators(llm)
    with pytest.raises(GenerationProtocolError, match="requires both"):
        generator.code_generator(
            _experiment(),
            "candidate_generator_001",
            2,
            parent_version=1,
            error=None,
            review=None,
        )
    assert [call["name"] for call in llm.calls] == ["research_generator_readiness"]


class _OfflineLLM:
    def chat(self, messages, tools=None, timeout=None):  # noqa: ANN001,ANN201
        raise TimeoutError("provider is offline")


def test_offline_provider_has_no_deterministic_fallback() -> None:
    generator = ProductionResearchGenerators(_OfflineLLM())
    with pytest.raises(GenerationUnavailableError, match="TimeoutError"):
        generator.verify_online()


def test_environment_without_model_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import accessor

    config = SimpleNamespace(
        llm=SimpleNamespace(
            langchain_provider="openai",
            langchain_model_name="",
            timeout_seconds=120,
        )
    )
    monkeypatch.setattr(accessor, "get_env_config", lambda: config)
    with pytest.raises(GenerationUnavailableError, match="not configured"):
        ProductionResearchGenerators.from_environment()


def test_production_assembly_verifies_online_and_injects_real_generators() -> None:
    class _Generators:
        def __init__(self) -> None:
            self.verified = False

        def verify_online(self) -> None:
            self.verified = True

        def hypothesis_generator(self, config, existing):  # noqa: ANN001,ANN201
            raise AssertionError("not executed during assembly")

        def code_generator(self, experiment, candidate_id, candidate_version, **kwargs):  # noqa: ANN001,ANN201
            raise AssertionError("not executed during assembly")

    generators = _Generators()
    store = SimpleNamespace()
    client = SimpleNamespace(get_capabilities=lambda: {})
    controller = _build_production_controller(
        store_factory=lambda: store,
        client_factory=lambda: client,
        generator_factory=lambda: generators,
    )
    assert generators.verified is True
    assert controller.store is store
    assert controller.dsa is client
    assert controller.hypothesis_generator.__self__ is generators
    assert controller.code_generator.__self__ is generators
    assert controller.require_longbridge is True
