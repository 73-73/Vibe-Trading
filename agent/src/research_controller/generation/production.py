"""Fail-closed production hypothesis, strategy and repair generation.

The campaign state machine accepts injectable callables so deterministic unit
tests do not need an online model.  The assembled service, however, must use a
real provider and must never mistake those test seams for an Agent.  This
module owns that production boundary:

* one online tool-call probe is required before a controller is assembled;
* hypotheses are requested in bounded batches of 3..10 and near-duplicates are
  rejected before they reach DSA;
* factor ids come from VIBE's bundled, AST-validated Registry (the 357-item CN
  phase-one scope), never from a second handwritten alias list;
* candidate source and manifests are validated locally against the frozen
  ``strategy-signal.v1`` / ``research_sdk.v1`` surface before submission;
* repairs carry both the normalized DSA error and independent review context.

Provider failures and malformed model output raise explicit exceptions.  There
is deliberately no deterministic production fallback.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from src.factors.registry import Registry, get_default_registry
from src.research_controller.contracts.generated.strategy_signal_v1 import StrategyManifest

_SDK_OPERATORS = frozenset(
    {
        "rank",
        "zscore",
        "winsorize",
        "clip",
        "where",
        "abs_value",
        "sign",
        "log1p_abs",
        "safe_div",
        "delay",
        "delta",
        "ts_sum",
        "ts_mean",
        "ts_std",
        "ts_min",
        "ts_max",
        "ts_rank",
        "ts_corr",
        "ts_cov",
        "decay_linear",
        "ema",
    }
)
_WINDOW_ARGS = {
    "delay": (1, 1, 512),
    "delta": (1, 1, 512),
    "ts_sum": (1, 2, 512),
    "ts_mean": (1, 2, 512),
    "ts_std": (1, 2, 512),
    "ts_min": (1, 2, 512),
    "ts_max": (1, 2, 512),
    "ts_rank": (1, 2, 512),
    "ts_corr": (2, 2, 512),
    "ts_cov": (2, 2, 512),
    "decay_linear": (1, 2, 512),
    "ema": (1, 2, 512),
}
_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.ClassDef,
    ast.Lambda,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.Try,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)
_FORBIDDEN_BINOPS = (
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)
_PHASE_ONE_PREFIXES = ("gtja191_", "qlib158_", "academic_")
_EXPECTED_PHASE_ONE_FACTORS = 357
_MAX_PROMPT_TEXT = 12_000
_MAX_SOURCE_BYTES = 65_536
_MIN_BATCH = 3
_MAX_BATCH = 10
_NEAR_DUPLICATE_THRESHOLD = 0.82


class GenerationUnavailableError(RuntimeError):
    """The configured production LLM cannot currently be used."""


class GenerationProtocolError(ValueError):
    """The provider returned output outside the bounded generation contract."""


class _ChatProvider(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: int | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class StrategyValidation:
    factor_ids: tuple[str, ...]
    sdk_call_count: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded_json(value: Any, *, limit: int = _MAX_PROMPT_TEXT) -> str:
    text = _canonical_json(value)
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _canonical_json({"truncated": text[:limit], "sha256": digest})


def _factor_catalog(registry: Registry) -> tuple[str, ...]:
    ids = tuple(
        factor_id
        for factor_id in registry.list(universe="equity_cn")
        if factor_id.startswith(_PHASE_ONE_PREFIXES)
    )
    if len(ids) != _EXPECTED_PHASE_ONE_FACTORS or len(set(ids)) != len(ids):
        raise GenerationUnavailableError(
            "VIBE phase-one factor Registry is not the expected unique 357-item scope"
        )
    return ids


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", value.lower()))


def _hypothesis_signature(value: Mapping[str, Any]) -> str:
    hypothesis = value.get("hypothesis") if isinstance(value.get("hypothesis"), Mapping) else value
    return " ".join(
        [
            str(hypothesis.get("mechanism") or ""),
            str(hypothesis.get("prediction") or ""),
            " ".join(sorted(str(item) for item in (hypothesis.get("factor_ids") or []))),
        ]
    )


def _near_duplicate(candidate: Mapping[str, Any], existing: list[Mapping[str, Any]]) -> bool:
    left = _tokens(_hypothesis_signature(candidate))
    if not left:
        return True
    for item in existing:
        right = _tokens(_hypothesis_signature(item))
        if not right:
            continue
        score = len(left & right) / len(left | right)
        if score >= _NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_READINESS_TOOL = _tool(
    "research_generator_readiness",
    "Echo the supplied nonce to prove the configured model endpoint is online and supports tools.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["nonce"],
        "properties": {"nonce": {"type": "string", "minLength": 16, "maxLength": 64}},
    },
)

_HYPOTHESIS_TOOL = _tool(
    "submit_research_hypotheses",
    "Submit a bounded batch of falsifiable factor hypotheses.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "minItems": _MIN_BATCH,
                "maxItems": _MAX_BATCH,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mechanism", "prediction", "failure_conditions", "factor_ids", "tags"],
                    "properties": {
                        "mechanism": {"type": "string", "minLength": 12, "maxLength": 1000},
                        "prediction": {"type": "string", "minLength": 8, "maxLength": 1000},
                        "failure_conditions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "minLength": 3, "maxLength": 300},
                        },
                        "factor_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {"type": "string"},
                        },
                        "tags": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        },
                    },
                },
            }
        },
    },
)

_CANDIDATE_TOOL = _tool(
    "submit_strategy_candidate",
    "Submit one strategy-signal.v1 candidate using only the frozen research_sdk.v1.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_code", "manifest"],
        "properties": {
            "source_code": {"type": "string", "minLength": 40, "maxLength": _MAX_SOURCE_BYTES},
            "manifest": {"type": "object"},
        },
    },
)


def _tool_call_arguments(response: Any, expected_name: str) -> dict[str, Any]:
    calls = getattr(response, "tool_calls", None)
    if not isinstance(calls, list) or len(calls) != 1:
        raise GenerationProtocolError(f"LLM must call {expected_name} exactly once")
    call = calls[0]
    name = call.get("name") if isinstance(call, Mapping) else getattr(call, "name", None)
    args = call.get("arguments") if isinstance(call, Mapping) else getattr(call, "arguments", None)
    if name != expected_name or not isinstance(args, Mapping):
        raise GenerationProtocolError(f"LLM returned an invalid {expected_name} tool call")
    return dict(args)


def validate_strategy_source(source: str, manifest_factor_ids: set[str] | frozenset[str]) -> StrategyValidation:
    """Validate the subset of strategy-signal.v1 needed before DSA submission."""
    if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
        raise GenerationProtocolError("candidate source exceeds 65536 bytes")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        raise GenerationProtocolError("candidate source is not valid bounded Python") from exc
    body = [
        stmt
        for stmt in tree.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "generate_scores":
        raise GenerationProtocolError("source must contain only def generate_scores(factors, params)")
    function = body[0]
    args = [item.arg for item in function.args.posonlyargs] + [item.arg for item in function.args.args]
    if args != ["factors", "params"] or function.args.vararg or function.args.kwarg:
        raise GenerationProtocolError("generate_scores signature must be (factors, params)")
    if len(list(ast.walk(tree))) > 256:
        raise GenerationProtocolError("candidate AST exceeds 256 nodes")

    referenced: set[str] = set()
    sdk_calls = 0
    for node in ast.walk(function):
        if node is not function and isinstance(node, ast.FunctionDef):
            raise GenerationProtocolError("nested function definitions are forbidden")
        if isinstance(node, _FORBIDDEN_NODES):
            raise GenerationProtocolError(f"forbidden AST node: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, _FORBIDDEN_BINOPS):
            raise GenerationProtocolError(f"forbidden binary operator: {type(node.op).__name__}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "params" and node.attr == "get"):
                raise GenerationProtocolError("only params.get attribute access is allowed")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id not in {"factors", "params"}:
                raise GenerationProtocolError("only factors/params literal subscripts are allowed")
            key = node.slice
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise GenerationProtocolError("factor/parameter subscript must use a literal string")
            if node.value.id == "factors":
                referenced.add(key.value)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SDK_OPERATORS:
                    raise GenerationProtocolError(f"unknown function call: {node.func.id}")
                sdk_calls += 1
                window = _WINDOW_ARGS.get(node.func.id)
                if window is not None and len(node.args) > window[0]:
                    raw = node.args[window[0]]
                    if (
                        not isinstance(raw, ast.Constant)
                        or not isinstance(raw.value, int)
                        or isinstance(raw.value, bool)
                        or not window[1] <= raw.value <= window[2]
                    ):
                        raise GenerationProtocolError(f"{node.func.id} requires a bounded literal window")
            elif not (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "params"
                and node.func.attr == "get"
            ):
                raise GenerationProtocolError("dynamic or attribute function calls are forbidden")
    if sdk_calls < 1 or sdk_calls > 128:
        raise GenerationProtocolError("candidate must use 1..128 research_sdk operators")
    if referenced != set(manifest_factor_ids):
        raise GenerationProtocolError("source factor references must exactly match manifest.factor_ids")
    return StrategyValidation(tuple(sorted(referenced)), sdk_calls)


class ProductionResearchGenerators:
    """Online production generators injected into ``ResearchCampaignController``."""

    def __init__(
        self,
        llm: _ChatProvider,
        *,
        registry: Registry | None = None,
        timeout_seconds: int = 120,
        usage_recorder: Callable[[str], Any] | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry or get_default_registry()
        self._factor_ids = _factor_catalog(self._registry)
        self._factor_id_set = frozenset(self._factor_ids)
        self._timeout_seconds = int(timeout_seconds)
        self._usage_recorder = usage_recorder
        # The controller passes the same mutable ``existing`` list throughout
        # one refill loop. Keying by that exact object prevents a batch created
        # for campaign A from spilling into campaign B when both share a config.
        # Holding the list reference also prevents CPython id reuse. The small
        # bound contains abandoned sessions after budget interruption.
        self._pending_batches: dict[
            int,
            tuple[list[dict[str, Any]], list[dict[str, Any]]],
        ] = {}
        self._lock = threading.Lock()
        self._online_verified = False

    @classmethod
    def from_environment(
        cls,
        *,
        usage_recorder: Callable[[str], Any] | None = None,
    ) -> "ProductionResearchGenerators":
        """Build the configured provider; missing configuration fails closed."""
        try:
            from src.config.accessor import get_env_config, reset_env_config
            from src.providers.chat import ChatLLM
            from src.providers.llm import _ensure_dotenv

            _ensure_dotenv()
            reset_env_config()
            config = get_env_config().llm
            if not config.langchain_provider.strip() or not config.langchain_model_name.strip():
                raise GenerationUnavailableError("production LLM provider/model is not configured")
            llm = ChatLLM(model_name=config.langchain_model_name.strip())
            return cls(
                llm,
                timeout_seconds=config.timeout_seconds,
                usage_recorder=usage_recorder,
            )
        except GenerationUnavailableError:
            raise
        except Exception as exc:  # do not leak provider credentials in diagnostics
            raise GenerationUnavailableError(
                f"production LLM construction failed: {type(exc).__name__}"
            ) from exc

    def _call_tool(
        self,
        *,
        name: str,
        tool: dict[str, Any],
        system: str,
        user: str,
        count_usage: bool = True,
    ) -> dict[str, Any]:
        if count_usage and self._usage_recorder is not None:
            # Reserve before dispatch. A timeout can still consume provider
            # tokens, so pessimistic accounting is the safe budget boundary.
            self._usage_recorder("llm_calls")
        try:
            response = self._llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[tool],
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise GenerationUnavailableError(
                f"production LLM call failed: {type(exc).__name__}"
            ) from exc
        return _tool_call_arguments(response, name)

    def verify_online(self) -> None:
        """Require a real provider round trip before production assembly succeeds."""
        if self._online_verified:
            return
        nonce = uuid.uuid4().hex
        payload = self._call_tool(
            name="research_generator_readiness",
            tool=_READINESS_TOOL,
            system="You are the production VIBE research generator readiness probe. Use the required tool only.",
            user=f"Echo this nonce exactly: {nonce}",
            count_usage=False,
        )
        if payload.get("nonce") != nonce:
            raise GenerationProtocolError("production LLM readiness nonce mismatch")
        self._online_verified = True

    def hypothesis_generator(
        self,
        config: dict[str, Any],
        existing: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return one item from a model-generated 3..10 hypothesis batch."""
        self.verify_online()
        with self._lock:
            key = id(existing)
            session = self._pending_batches.get(key)
            if session is None or session[0] is not existing or not session[1]:
                if len(self._pending_batches) >= 32:
                    self._pending_batches.pop(next(iter(self._pending_batches)))
                pending = self._generate_hypothesis_batch(config, existing)
                session = (existing, pending)
                self._pending_batches[key] = session
            item = session[1].pop(0)
            if not session[1]:
                self._pending_batches.pop(key, None)
            return item

    def _generate_hypothesis_batch(
        self,
        config: dict[str, Any],
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        existing_summaries = [
            {
                "experiment_id": item.get("experiment_id"),
                "hypothesis": item.get("hypothesis") or {},
            }
            for item in existing
        ]
        accepted: list[dict[str, Any]] = []
        comparison: list[Mapping[str, Any]] = [*existing]
        for attempt in range(3):
            payload = self._call_tool(
                name="submit_research_hypotheses",
                tool=_HYPOTHESIS_TOOL,
                system=(
                    "You generate falsifiable CN equity factor research hypotheses. "
                    "Return 3..10 distinct hypotheses through the required tool. Do not claim alpha, "
                    "profitability or access the locked 2022-2025 holdout. Every factor id must come "
                    "from the supplied Registry list and each failure condition must be machine-testable."
                ),
                user=_bounded_json(
                    {
                        "objective": config.get("objective"),
                        "development_window": config.get("development_window"),
                        "factor_scope": config.get("factor_scope"),
                        "registry_factor_ids": self._factor_ids,
                        "existing_hypotheses": existing_summaries,
                        "rejected_near_duplicates": len(comparison) - len(existing),
                        "attempt": attempt + 1,
                    }
                ),
            )
            raw_batch = payload.get("hypotheses")
            if not isinstance(raw_batch, list) or not _MIN_BATCH <= len(raw_batch) <= _MAX_BATCH:
                raise GenerationProtocolError("hypothesis batch must contain 3..10 items")
            for raw in raw_batch:
                if not isinstance(raw, Mapping):
                    raise GenerationProtocolError("hypothesis item must be an object")
                factors = raw.get("factor_ids")
                failures = raw.get("failure_conditions")
                tags = raw.get("tags")
                mechanism = str(raw.get("mechanism") or "").strip()
                prediction = str(raw.get("prediction") or "").strip()
                if len(mechanism) < 12 or len(prediction) < 8:
                    raise GenerationProtocolError("hypothesis mechanism/prediction is too short")
                if (
                    not isinstance(factors, list)
                    or not 1 <= len(factors) <= 4
                    or len(set(factors)) != len(factors)
                    or any(item not in self._factor_id_set for item in factors)
                ):
                    raise GenerationProtocolError("hypothesis contains unknown or duplicate Registry factor ids")
                if not isinstance(failures, list) or not failures or any(not str(item).strip() for item in failures):
                    raise GenerationProtocolError("hypothesis requires explicit failure conditions")
                if not isinstance(tags, list) or not tags:
                    raise GenerationProtocolError("hypothesis requires tags")
                normalized = {
                    "mechanism": mechanism,
                    "prediction": prediction,
                    "failure_conditions": [str(item).strip() for item in failures],
                    "factor_ids": [str(item) for item in factors],
                }
                candidate = {"hypothesis": normalized}
                if _near_duplicate(candidate, comparison):
                    continue
                fingerprint = hashlib.sha256(
                    (_canonical_json(normalized) + ":" + uuid.uuid4().hex).encode("utf-8")
                ).hexdigest()
                experiment = {
                    "experiment_id": f"exp_{fingerprint[:24]}",
                    "round_id": f"round_{fingerprint[24:40]}",
                    "objective": config.get("objective") or "CN equity factor research",
                    "hypothesis": normalized,
                    "data_snapshot_id": None,
                    "universe_snapshot_id": None,
                    "tags": [str(item).strip() for item in tags if str(item).strip()],
                    "window": config.get("development_window") or {},
                }
                accepted.append(experiment)
                comparison.append(experiment)
                if len(accepted) == _MAX_BATCH:
                    break
            if len(accepted) >= _MIN_BATCH:
                break
        if len(accepted) < _MIN_BATCH:
            raise GenerationProtocolError("model could not produce 3 unique hypotheses after duplicate filtering")
        return accepted

    def code_generator(
        self,
        experiment: dict[str, Any],
        candidate_id: str,
        candidate_version: int,
        *,
        parent_version: int | None = None,
        error: dict[str, Any] | None = None,
        review: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate and validate an initial candidate or repair revision."""
        self.verify_online()
        hypothesis = experiment.get("hypothesis") or {}
        factor_ids = hypothesis.get("factor_ids") or []
        if (
            not isinstance(factor_ids, list)
            or not factor_ids
            or len(set(factor_ids)) != len(factor_ids)
            or any(item not in self._factor_id_set for item in factor_ids)
        ):
            raise GenerationProtocolError("experiment factor ids are not in the VIBE phase-one Registry")
        is_repair = parent_version is not None
        if is_repair and (not isinstance(error, Mapping) or not isinstance(review, Mapping)):
            raise GenerationProtocolError("repair generation requires both DSA error and review context")
        context = {
            "candidate_id": candidate_id,
            "candidate_version": candidate_version,
            "parent_version": parent_version,
            "hypothesis": hypothesis,
            "allowed_factor_ids": factor_ids,
            "research_sdk_operators": sorted(_SDK_OPERATORS),
            "strategy_contract": {
                "entrypoint": "def generate_scores(factors, params)",
                "no_imports_io_network_reflection_loops_comprehensions": True,
                "literal_factor_subscripts_only": True,
                "division_must_use_safe_div": True,
            },
            "repair_context": {
                "error": error,
                "review": review,
            }
            if is_repair
            else None,
        }
        payload = self._call_tool(
            name="submit_strategy_candidate",
            tool=_CANDIDATE_TOOL,
            system=(
                "You generate one strategy-signal.v1 candidate through the required tool. "
                "Use every allowed factor id exactly by factors[\"literal_id\"] and only frozen "
                "research_sdk.v1 functions. No imports, pandas, numpy, dynamic lookup, loops or IO. "
                + (
                    "This is a repair: use the supplied error and independent review, preserve the "
                    "review must_preserve constraints, and produce source materially different from v1."
                    if is_repair
                    else "This is the initial version and must pass static validation before submission."
                )
            ),
            user=_bounded_json(context),
        )
        source = payload.get("source_code")
        manifest_raw = payload.get("manifest")
        if not isinstance(source, str) or not isinstance(manifest_raw, Mapping):
            raise GenerationProtocolError("candidate tool output requires source_code and manifest")
        try:
            manifest_model = StrategyManifest.model_validate(dict(manifest_raw))
        except Exception as exc:
            raise GenerationProtocolError("candidate manifest violates strategy-signal.v1") from exc
        manifest = manifest_model.model_dump(mode="json", exclude_none=True)
        if manifest.get("factor_ids") != factor_ids:
            raise GenerationProtocolError("candidate manifest factor_ids must preserve hypothesis order and scope")
        validate_strategy_source(source, set(factor_ids))
        return source, manifest
