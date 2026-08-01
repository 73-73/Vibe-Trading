"""Vibe Research Campaign Controller（§7.2 / §8 / §13）。

面向用户的唯一高层入口。Controller 自己持久化跨 DSA 的状态、事件游标和队列，
不复用仅保存 prompt 的 scheduled job JSON（§7.2.1）。

关键不变量：

- 事件至少一次投递，按 ``message_id`` 去重；状态、队列写入和事件游标更新在
  同一本地事务提交（§4.3 / §7.2.1）。
- 所有创建/启动请求携带幂等键；网络超时不等同提交失败，先按幂等键或资源 ID
  查询再决定是否重提（§4.3）。
- Vibe 重启后先查询 DSA execution 再决定状态，不把所有 RUNNING 重置为 PENDING
  （§16）。
- 每次接受结果 / 修复 / 新假设 / 淘汰前先写不可变 decision（§5.8 / §6.10）。
- 自动修复谱系：每个逻辑候选最多 3 个修复版本；相同错误指纹连续 2 次停止；
  相同源码哈希不得作为新版本重提（§8.3）。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from src.research_controller.client.dsa_client import (
    DsaLoopClient,
    DsaProtocolError,
    DsaUnavailableError,
    sha256_hex,
)
from src.research_controller.contracts import source_bundle_sha256
from src.research_controller.contracts.generated.campaign_create_request_v1 import CampaignCreateRequest
from src.research_controller.repair.lineage import (
    can_repair,
    next_candidate_version,
    repair_budget_remaining,
    source_hash_used,
)
from src.research_controller.reporting.report import render_campaign_report
from src.research_controller.state_machine.machine import (
    CAMPAIGN_TERMINAL_STATUS,
    DEFAULT_BUDGET_CANDIDATE_VERSIONS,
    DEFAULT_BUDGET_LLM_CALLS,
    MAX_INFRA_RETRIES,
    MAX_REPAIR_VERSIONS_PER_CANDIDATE,
    POLL_BACKOFF_MAX_SECONDS,
    TERMINAL_EXPERIMENT_PHASES,
    compute_poll_interval,
)
from src.research_controller.store.campaign_store import CampaignStore
from src.research_controller.store.canonical import canonical_sha256

logger = logging.getLogger(__name__)

LOCKED_HOLDOUT_START = "2022-01-01"
GATE_CHAMPION_V1 = "gate_champion_v1"
STRATEGY_CONTRACT_V1 = "strategy-signal.v1"
FACTOR_TARGET = 357
DEFAULT_MOCK_DATA_SNAPSHOT_ID = "data_dev_001"

# R09 决策 B（§9.3）：Reviewer 意见是旁路参考，等待超时后允许决策但必须标记缺少独立评审。
REVIEW_BYPASS_MARKER = "缺少 DSA 独立评审"
DEFAULT_REVIEW_TIMEOUT_SECONDS = 300.0

_DECISION_ACTIONS = {
    "accept_result",
    "submit_candidate_revision",
    "retry_same_execution",
    "create_data_task",
    "run_additional_test",
    "create_new_hypothesis",
    "abandon_candidate",
    "quarantine_candidate",
    "create_gate_challenger",
    "pause_experiment",
    "complete_experiment",
    "request_human_review",
}


class CampaignValidationError(ValueError):
    """Raised when a campaign request violates §7.2 constraints."""


class CampaignNotFoundError(KeyError):
    """Raised when a campaign id does not exist."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# 默认生成器（可注入替换；真实假设/代码生成由上层 LLM 组件提供）
# ---------------------------------------------------------------------------


def default_hypothesis_generator(config: dict[str, Any], existing: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Produce one deterministic hypothesis from the campaign objective."""
    if existing:
        return None
    objective = config.get("objective") or "研究 A 股量价因子组合的横截面稳定性"
    window = config.get("development_window") or {}
    factor_scope = config.get("factor_scope") or "a_share_native_357"
    factor_ids = ["qlib158_mom_20", "gtja191_alpha099"] if factor_scope == "a_share_native_357" else []
    exp_id = f"exp_{uuid.uuid4().hex[:20]}"
    round_id = f"round_{uuid.uuid4().hex[:12]}"
    return {
        "experiment_id": exp_id,
        "round_id": round_id,
        "objective": objective,
        "hypothesis": {
            "mechanism": "动量捕捉趋势延续，成交量确认参与度；两者相关性低时组合更稳定",
            "prediction": "动量与成交量等权组合在周频调仓下有正向超额",
            "failure_conditions": ["rank_ic_median <= 0", "double_cost_excess_return_direction < 1"],
            "factor_ids": factor_ids,
        },
        "data_snapshot_id": None,
        # R08：universe_snapshot_id 只来自 DSA universe build，不硬编码。
        "universe_snapshot_id": None,
        "tags": ["momentum", "volume"],
        "window": window,
    }


def default_code_generator(
    experiment: dict[str, Any],
    candidate_id: str,
    candidate_version: int,
    *,
    parent_version: int | None = None,
    error: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a deterministic, contract-shaped strategy source + manifest.

    Repair versions include a version marker so each logical version has a
    distinct source hash (§8.3 相同源码哈希不得作为新版本重提). The real code
    generator is an upper-layer LLM component; this default keeps the controller
    testable against the Mock DSA.
    """
    del error, review  # reserved for repair-aware generators
    version_note = f"# v{candidate_version} repair of v{parent_version}" if parent_version is not None else "# v1 initial"
    source_code = (
        "def generate_scores(factors, params):\n"
        f"    {version_note}\n"
        "    primary = list(factors.keys())[0]\n"
        "    return factors[primary] * params.get('score_scale', 1.0)\n"
    )
    manifest = {
        "factor_ids": experiment.get("hypothesis", {}).get("factor_ids", ["qlib158_mom_20"]),
        "parameters": {"score_scale": {"type": "number", "minimum": 0, "maximum": 10}},
        "default_parameters": {"score_scale": 1.0},
        "direction": "higher_is_better",
        "rebalance_frequency": "weekly",
        "top_n": 20,
        "max_holdings": 20,
        "exit_policy": "leave_top_n",
        "seed": 191,
    }
    return source_code, manifest


class ResearchCampaignController:
    """Stateful controller wiring the campaign store to the DSA research-loop API.

    Args:
        store: Persistence backend.
        dsa_client: Loopback DSA research-loop client.
        hypothesis_generator: Callable ``(config, existing_experiments) -> dict | None``.
        code_generator: Callable producing ``(source_code, manifest)``.
        queue_target: §13.2 自动补充到 20 的目标队列长度。
        refill_threshold: 待执行实验少于该值时自动补充。
        snapshot_id_resolver: Callable ``(campaign, build_data) -> snapshot_id | None``
            用于 Mock（真实 DSA 通过事件返回 snapshot id）。
        max_repair_versions: 每个逻辑候选最多修复版本数（§8.3 默认 3）。
        max_consecutive_fingerprint: 相同错误指纹连续次数上限（默认 2）。
    """

    def __init__(
        self,
        store: CampaignStore,
        dsa_client: DsaLoopClient,
        *,
        hypothesis_generator: Callable[..., dict[str, Any] | None] | None = None,
        code_generator: Callable[..., tuple[str, dict[str, Any]]] | None = None,
        queue_target: int = 20,
        refill_threshold: int = 5,
        snapshot_id_resolver: Callable[..., str | None] | None = None,
        max_repair_versions: int = MAX_REPAIR_VERSIONS_PER_CANDIDATE,
        max_consecutive_fingerprint: int = 2,
        review_timeout_seconds: float = DEFAULT_REVIEW_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.dsa = dsa_client
        self.hypothesis_generator = hypothesis_generator or default_hypothesis_generator
        self.code_generator = code_generator or default_code_generator
        self.queue_target = int(queue_target)
        self.refill_threshold = int(refill_threshold)
        self.snapshot_id_resolver = snapshot_id_resolver
        self.max_repair_versions = int(max_repair_versions)
        self.max_consecutive_fingerprint = int(max_consecutive_fingerprint)
        self.review_timeout_seconds = float(review_timeout_seconds)
        self._dsa_unavailable = False
        self._consecutive_failures = 0

    # ==================================================================
    # Campaign 生命周期 API（§7.2）
    # ==================================================================

    def create_campaign(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist a new research campaign (§7.2.1)."""
        req = CampaignCreateRequest.model_validate(request)
        self._validate_campaign_request(req)
        now = _now()
        campaign_id = _short_id("campaign")
        record = {
            "campaign_id": campaign_id,
            "research_goal_id": _short_id("goal"),
            "status": "initializing",
            "current_stage": "WAIT_DATA",
            "config_sha256": canonical_sha256(req.model_dump(mode="json")),
            "protocol_bundle_sha256": source_bundle_sha256(),
            "data_snapshot_id": None,
            "universe_snapshot_id": None,
            "gate_policy_version": GATE_CHAMPION_V1,
            "factor_inventory": {"target": FACTOR_TARGET, "completed": 0, "skipped": 0, "failed": 0},
            "queue_counts": {"pending": 0, "running": 0, "blocked": 0},
            "budget_usage": {"llm_calls_rolling_24h": 0, "candidate_versions_rolling_24h": 0},
            "last_event_sequences": {},
            "config": req.model_dump(mode="json"),
            "blocked_reason": None,
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_campaign(record)
        logger.info("created research campaign %s", campaign_id)
        return self._create_response(record)

    @staticmethod
    def _validate_campaign_request(req: CampaignCreateRequest) -> None:
        if str(req.development_window.end_date) >= LOCKED_HOLDOUT_START:
            raise CampaignValidationError("development_window must end before 2022-01-01")
        if req.automation is not None and req.automation.open_locked_holdout:
            raise CampaignValidationError("open_locked_holdout cannot be enabled at campaign creation")

    def _create_response(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "campaign_id": record["campaign_id"],
            "status": record["status"],
            "research_goal_id": record["research_goal_id"],
            "data_snapshot_id": record.get("data_snapshot_id"),
            "universe_snapshot_id": record.get("universe_snapshot_id"),
            "gate_policy_version": record["gate_policy_version"],
            "factor_inventory": record.get("factor_inventory", {}),
            "created_at": record["created_at"],
        }

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        return self._campaign_detail(campaign)

    def list_campaigns(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.list_campaigns(status=status, limit=limit)
        return [self._campaign_detail(self.store.get_campaign(r["campaign_id"])) for r in rows]

    def list_active_campaign_ids(self) -> list[str]:
        """Return campaign ids still being driven by the pipeline (§13).

        Only ``initializing`` / ``running`` / ``waiting_budget`` campaigns are
        polled; ``paused`` / ``cancelled`` / ``blocked`` / ``completed`` are not.
        Read-only convenience for the pipeline driver; backward compatible.
        """
        rows = self.store.list_campaigns(limit=500)
        return [
            r["campaign_id"]
            for r in rows
            if r["status"] in ("initializing", "running", "waiting_budget")
        ]

    def pause_campaign(self, campaign_id: str, *, cancel_running: bool = False) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        if campaign["status"] in CAMPAIGN_TERMINAL_STATUS:
            raise CampaignValidationError(f"cannot pause campaign in terminal state {campaign['status']}")
        self.store.update_campaign(campaign_id, status="paused", updated_at=_now())
        if cancel_running:
            self._cancel_running_executions(campaign_id)
        return self._campaign_detail(self.store.get_campaign(campaign_id))

    def resume_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        if campaign["status"] != "paused":
            raise CampaignValidationError("only a paused campaign can be resumed")
        self.store.update_campaign(campaign_id, status="running", updated_at=_now())
        return self._campaign_detail(self.store.get_campaign(campaign_id))

    def cancel_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        if campaign["status"] not in ("completed", "cancelled"):
            self.store.update_campaign(
                campaign_id, status="cancelled", current_stage="CANCELLED", updated_at=_now()
            )
            self._cancel_running_executions(campaign_id)
        return self._campaign_detail(self.store.get_campaign(campaign_id))

    def _cancel_running_executions(self, campaign_id: str) -> None:
        for exec_record in self.store.list_executions(campaign_id):
            if exec_record["status"] in ("queued", "running", "waiting_review"):
                try:
                    self.dsa.cancel_execution(exec_record["execution_id"])
                except DsaUnavailableError:
                    logger.warning("DSA unavailable while cancelling %s", exec_record["execution_id"])

    # ==================================================================
    # 查询 API（§7.2.2）
    # ==================================================================

    def get_experiments(self, campaign_id: str) -> list[dict[str, Any]]:
        self._require_campaign(campaign_id)
        return self.store.list_experiments(campaign_id)

    def get_candidates(self, campaign_id: str) -> list[dict[str, Any]]:
        self._require_campaign(campaign_id)
        return self.store.list_candidates(campaign_id)

    def get_repairs(self, campaign_id: str) -> list[dict[str, Any]]:
        self._require_campaign(campaign_id)
        return self.store.list_repairs(campaign_id)

    def get_decisions(self, campaign_id: str) -> list[dict[str, Any]]:
        self._require_campaign(campaign_id)
        return self.store.list_decisions(campaign_id)

    def latest_report(self, campaign_id: str) -> dict[str, Any] | None:
        self._require_campaign(campaign_id)
        return self.store.get_latest_report(campaign_id)

    def generate_report(self, campaign_id: str) -> dict[str, Any]:
        """Build and persist the final Chinese report (§15.3 / R13).

        幂等：若该 campaign 已生成过报告，直接返回既有最新报告，不重复渲染。
        报告在无足够证据 / 缺独立评审时做“含缺口”降级渲染，不抛异常
        （§15.3 / §9.3），因此 completed 自动触发与 HTTP 手动触发都能安全调用。
        """
        self._require_campaign(campaign_id)
        existing = self.store.get_latest_report(campaign_id)
        if existing is not None:
            return existing
        data = self._report_data(campaign_id)
        markdown = render_campaign_report(data)
        report_id = _short_id("report")
        self.store.save_report(campaign_id, report_id, markdown)
        latest = self.store.get_latest_report(campaign_id)
        if latest is None:  # pragma: no cover - save_report 刚刚写入一行
            return {"campaign_id": campaign_id, "report_id": report_id, "report_md": markdown}
        return latest

    def _generate_report_on_completion(self, campaign_id: str) -> None:
        """R13：campaign 进入 completed 时自动产出最终中文报告（幂等）。

        报告生成失败不应回滚已完成的 campaign：此处捕获并记录，仍可通过
        ``POST /reports/generate`` 按需触发。
        """
        try:
            self.generate_report(campaign_id)
        except Exception:
            logger.exception(
                "auto-generate report failed for completed campaign %s "
                "(can be generated on demand via POST /reports/generate)",
                campaign_id,
            )

    def _require_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        return campaign

    def _campaign_detail(self, campaign: dict[str, Any]) -> dict[str, Any]:
        campaign_id = campaign["campaign_id"]
        experiments = self.store.list_experiments(campaign_id)
        executions = self.store.list_executions(campaign_id)
        errors = self.store.list_errors(campaign_id)
        repairs = self.store.list_repairs(campaign_id)
        reviews = self.store.list_reviews(campaign_id)
        decisions = self.store.list_decisions(campaign_id)
        report = self.store.get_latest_report(campaign_id)
        active_executions = [e for e in executions if e["status"] in ("queued", "running", "waiting_review")]
        return {
            "campaign_id": campaign_id,
            "research_goal_id": campaign["research_goal_id"],
            "status": campaign["status"],
            "current_stage": campaign["current_stage"],
            "data_snapshot_id": campaign["data_snapshot_id"],
            "universe_snapshot_id": campaign["universe_snapshot_id"],
            "gate_policy_version": campaign["gate_policy_version"],
            "factor_inventory": campaign.get("factor_inventory", {}),
            "queue_counts": campaign.get("queue_counts", {}),
            "budget_usage": campaign.get("budget_usage", {}),
            "last_event_sequences": campaign.get("last_event_sequences", {}),
            "experiment_count": len(experiments),
            "candidate_count": len(self.store.list_candidates(campaign_id)),
            "execution_count": len(executions),
            "active_executions": [
                {
                    "execution_id": e["execution_id"],
                    "execution_type": e["execution_type"],
                    "status": e["status"],
                    "current_stage": e["current_stage"],
                }
                for e in active_executions
            ],
            "recent_errors": errors[-5:],
            "recent_repairs": repairs[-5:],
            "review_count": len(reviews),
            "decision_count": len(decisions),
            "latest_report": report,
            "blocked_reason": campaign.get("blocked_reason"),
            "created_at": campaign["created_at"],
            "updated_at": campaign["updated_at"],
        }

    # ==================================================================
    # 流水线驱动（§13）
    # ==================================================================

    def run_pipeline_once(self, campaign_id: str) -> dict[str, Any]:
        """One pipeline step: reconcile DSA executions -> poll events -> advance.

        Safe to call repeatedly and from a fresh process after restart; all
        actions are guarded by local state + idempotency records.
        """
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        if campaign["status"] in ("paused", "cancelled", "blocked", "completed"):
            return self._summary(campaign_id)
        self._reconcile_executions(campaign)
        self._poll_events(campaign)
        self._advance(campaign)
        self._sync_campaign_status(campaign_id)
        return self._summary(campaign_id)

    def poll_events_once(self, campaign_id: str) -> dict[str, Any]:
        self._require_campaign(campaign_id)
        self._poll_events(self.store.get_campaign(campaign_id))
        self._sync_campaign_status(campaign_id)
        return self._summary(campaign_id)

    def reconcile_executions_once(self, campaign_id: str) -> dict[str, Any]:
        self._require_campaign(campaign_id)
        self._reconcile_executions(self.store.get_campaign(campaign_id))
        return self._summary(campaign_id)

    def poll_interval_seconds(self, campaign_id: str) -> float:
        """Return the §6.6 poll interval for the campaign's current state."""
        campaign = self.store.get_campaign(campaign_id)
        executions = self.store.list_executions(campaign_id) if campaign else []
        active = any(e["status"] in ("queued", "running") for e in executions)
        # R09：execution 不再被 review.requested 覆盖为 waiting_review 状态；
        # 用 current_stage 标记等待评审，轮询仍按评审节奏（§6.6）。
        waiting_review = any(
            e["status"] == "waiting_review" or e.get("current_stage") == "waiting_review" for e in executions
        )
        backoff = 0.0
        if self._dsa_unavailable:
            backoff = min(float(2**self._consecutive_failures), POLL_BACKOFF_MAX_SECONDS)
        return compute_poll_interval(
            any_active_execution=active,
            any_waiting_review=waiting_review,
            backoff_seconds=backoff,
        )

    def _summary(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        experiments = self.store.list_experiments(campaign_id) if campaign else []
        terminal = sum(1 for e in experiments if e["phase"] in TERMINAL_EXPERIMENT_PHASES)
        return {
            "campaign_id": campaign_id,
            "status": campaign["status"] if campaign else "unknown",
            "current_stage": campaign["current_stage"] if campaign else "",
            "experiments_total": len(experiments),
            "experiments_terminal": terminal,
            "complete": bool(campaign and campaign["status"] in ("completed", "cancelled")),
        }

    # ==================================================================
    # 执行对账（§16 恢复）
    # ==================================================================

    def _reconcile_executions(self, campaign: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        for exec_record in self.store.list_executions(campaign_id):
            if exec_record["status"] not in ("queued", "running", "waiting_review"):
                continue
            try:
                result = self.dsa.get_execution(exec_record["execution_id"])
            except DsaUnavailableError:
                self._mark_unavailable()
                continue
            except DsaProtocolError:
                continue
            if result["status"] == "ok":
                summary = (result.get("data") or {}).get("execution") or {}
                fields: dict[str, Any] = {}
                if summary.get("status"):
                    fields["status"] = summary["status"]
                if summary.get("progress") is not None:
                    fields["progress"] = summary["progress"]
                if summary.get("current_stage"):
                    fields["current_stage"] = summary["current_stage"]
                if summary.get("error_id"):
                    fields["error_id"] = summary["error_id"]
                if summary.get("evidence_id"):
                    fields["evidence_id"] = summary["evidence_id"]
                if summary.get("started_at"):
                    fields["started_at"] = summary["started_at"]
                if summary.get("finished_at"):
                    fields["finished_at"] = summary["finished_at"]
                if fields:
                    self.store.update_execution(exec_record["execution_id"], **fields)
            elif (result.get("error") or {}).get("code") == "execution_not_found":
                # 从未被 DSA 接受：不盲目重提，标记 interrupted 由 _advance 决定
                logger.warning("DSA execution %s not found; marking interrupted", exec_record["execution_id"])
                self.store.update_execution(exec_record["execution_id"], status="interrupted")

    # ==================================================================
    # 事件轮询 / 去重 / 游标（§4.3 / §6.6）
    # ==================================================================

    def _poll_events(self, campaign: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        if campaign["status"] not in ("running", "waiting_budget", "initializing"):
            return
        experiments = self.store.list_experiments(campaign_id)
        if not experiments:
            return
        seqs = campaign.get("last_event_sequences", {})
        bundle: dict[str, Any] = {"events": [], "errors": [], "reviews": [], "evidence": []}
        for exp in experiments:
            exp_id = exp["experiment_id"]
            cursor = int(seqs.get(exp_id, 0))
            for _page in range(20):
                try:
                    result = self.dsa.poll_events(exp_id, after_sequence=cursor, page_size=100)
                except DsaUnavailableError:
                    self._mark_unavailable()
                    return
                except DsaProtocolError:
                    return
                if result["status"] == "error":
                    error = result.get("error") or {}
                    if error.get("code") == "event_cursor_expired":
                        # 波次 C：不再仅 _block_campaign，先做尽力本地恢复。
                        recovered = self._recover_cursor_expired(
                            campaign, exp_id, error.get("message", "")
                        )
                        if not recovered:
                            return  # 已 blocked
                        # 尽力恢复成功：重置游标到 0 触发全量重新拉取（已落库事件按
                        # message_id 去重），避免下次以过期游标再次 410。
                        self.store.set_campaign_cursor(campaign_id, exp_id, 0)
                        cursor = 0
                        continue
                    elif result.get("retryable"):
                        self._mark_unavailable()
                    return
                data = result.get("data") or {}
                items = data.get("items") or []
                if items:
                    bundle["events"] = bundle["events"] or []
                    for item in items:
                        bundle["events"].append((campaign_id, exp_id, item))
                        self._stage_event_mutations(campaign, exp_id, item, bundle)
                    bundle["cursor"] = (campaign_id, exp_id, int(data.get("next_sequence", cursor)))
                    cursor = int(data.get("next_sequence", cursor))
                if not data.get("has_more"):
                    break
        if bundle.get("events") or bundle.get("cursor"):
            self.store.apply_bundle(bundle)

    def _stage_event_mutations(
        self, campaign: dict[str, Any], exp_id: str, item: dict[str, Any], bundle: dict[str, Any]
    ) -> None:
        """Derive local state mutations for one event (applied in one txn)."""
        event_type = item.get("event_type", "")
        payload = item.get("payload") or {}
        now = _now()
        if event_type == "experiment.created":
            bundle.setdefault("experiments", []).append((exp_id, {"status": "active", "phase": "created", "updated_at": now}))
        elif event_type == "candidate.registered":
            bundle.setdefault("experiments", []).append((exp_id, {"phase": "candidate_built", "updated_at": now}))
        elif event_type == "candidate.validated":
            cid = payload.get("candidate_id")
            ver = payload.get("candidate_version")
            if cid:
                bundle.setdefault("candidates", []).append((cid, ver, {"status": "validated"}))
                bundle.setdefault("experiments", []).append((exp_id, {"phase": "static_validated", "updated_at": now}))
        elif event_type == "candidate.rejected":
            cid = payload.get("candidate_id")
            ver = payload.get("candidate_version")
            error = self._fetch_error(campaign, exp_id, payload.get("error_id"))
            if error:
                bundle.setdefault("errors", []).append(error)
            if cid:
                bundle.setdefault("candidates", []).append((cid, ver, {"status": "static_failed"}))
                bundle.setdefault("experiments", []).append((exp_id, {"phase": "static_failed", "updated_at": now}))
        elif event_type == "candidate.quarantined":
            cid = payload.get("candidate_id")
            ver = payload.get("candidate_version")
            if cid:
                bundle.setdefault("candidates", []).append((cid, ver, {"status": "quarantined"}))
                bundle.setdefault("experiments", []).append((exp_id, {"phase": "quarantined", "updated_at": now}))
        elif event_type == "execution.queued":
            exec_id = payload.get("execution_id")
            if exec_id:
                bundle.setdefault("executions", []).append((exec_id, {"status": "queued", "current_stage": "queued"}))
        elif event_type == "execution.started":
            exec_id = payload.get("execution_id")
            if exec_id:
                bundle.setdefault("executions", []).append(
                    (exec_id, {"status": "running", "current_stage": "running", "started_at": now})
                )
        elif event_type == "execution.progress":
            # 可丢弃观测事件：仅更新进度，不触发状态迁移（§6.6.1）
            exec_id = payload.get("execution_id")
            if exec_id:
                fields: dict[str, Any] = {}
                if payload.get("progress") is not None:
                    fields["progress"] = payload["progress"]
                if payload.get("stage"):
                    fields["current_stage"] = payload["stage"]
                if fields:
                    bundle.setdefault("executions", []).append((exec_id, fields))
        elif event_type == "execution.completed":
            exec_id = payload.get("execution_id")
            evidence_id = payload.get("evidence_id")
            if exec_id:
                bundle.setdefault("executions", []).append(
                    (exec_id, {"status": "completed", "evidence_id": evidence_id, "finished_at": now, "progress": 1.0})
                )
                if evidence_id:
                    evidence = self._fetch_evidence(campaign, exp_id, exec_id, evidence_id)
                    if evidence:
                        bundle.setdefault("evidence", []).append(evidence)
                self._phase_on_execution_terminal(campaign, exp_id, exec_id, success=True, bundle=bundle)
        elif event_type == "execution.failed":
            exec_id = payload.get("execution_id")
            error_id = payload.get("error_id")
            error = self._fetch_error(campaign, exp_id, error_id)
            if error:
                bundle.setdefault("errors", []).append(error)
            if exec_id:
                bundle.setdefault("executions", []).append(
                    (exec_id, {"status": "failed", "error_id": error_id, "finished_at": now})
                )
                self._phase_on_execution_terminal(campaign, exp_id, exec_id, success=False, bundle=bundle)
        elif event_type == "execution.cancelled":
            exec_id = payload.get("execution_id")
            if exec_id:
                bundle.setdefault("executions", []).append((exec_id, {"status": "cancelled", "finished_at": now}))
        elif event_type == "execution.interrupted":
            exec_id = payload.get("execution_id")
            if exec_id:
                bundle.setdefault("executions", []).append((exec_id, {"status": "interrupted"}))
        elif event_type == "review.requested":
            exec_id = payload.get("execution_id")
            if exec_id:
                # R09 决策 B：Reviewer 是旁路意见，不覆盖 execution 真实终态（保留
                # completed/failed）；仅用 current_stage 标记等待评审，供 §9.3 超时窗口判断。
                bundle.setdefault("executions", []).append((exec_id, {"current_stage": "waiting_review"}))
        elif event_type == "review.completed":
            exec_id = payload.get("execution_id")
            review_id = payload.get("review_id")
            review = self._fetch_review(campaign, exp_id, exec_id, review_id)
            if review:
                bundle.setdefault("reviews", []).append(review)
            if exec_id and review_id:
                fields: dict[str, Any] = {"review_id": review_id}
                exec_record = self.store.get_execution(exec_id)
                if exec_record and exec_record.get("status") in ("completed", "failed"):
                    # 评审已产出：把 current_stage 恢复为真实终态，避免 poll 一直按评审节奏
                    fields["current_stage"] = exec_record["status"]
                bundle.setdefault("executions", []).append((exec_id, fields))
        elif event_type == "review.failed":
            exec_id = payload.get("execution_id")
            if exec_id:
                bundle.setdefault("executions", []).append((exec_id, {"review_id": None}))

    def _phase_on_execution_terminal(
        self, campaign: dict[str, Any], exp_id: str, exec_id: str, *, success: bool, bundle: dict[str, Any]
    ) -> None:
        exec_record = self.store.get_execution(exec_id)
        if exec_record is None:
            return
        exec_type = exec_record.get("execution_type", "")
        if exec_type == "strategy_sandbox_smoke":
            phase = "sandbox_passed" if success else "sandbox_failed"
        elif exec_type == "portfolio_backtest":
            phase = "backtest_passed" if success else "backtest_failed"
        elif exec_type == "robustness":
            phase = "robustness_passed" if success else "robustness_failed"
        else:
            return
        bundle.setdefault("experiments", []).append((exp_id, {"phase": phase, "updated_at": _now()}))

    def _fetch_error(self, campaign: dict[str, Any], exp_id: str, error_id: Any) -> dict[str, Any] | None:
        if not error_id:
            return None
        try:
            result = self.dsa.get_error(error_id)
        except (DsaUnavailableError, DsaProtocolError):
            return None
        if result["status"] != "ok":
            return None
        error = (result.get("data") or {}).get("error") or {}
        if not error.get("error_id"):
            return None
        return {
            "error_id": error["error_id"],
            "campaign_id": campaign["campaign_id"],
            "experiment_id": exp_id,
            "execution_id": error.get("execution_id") or "",
            "attempt_id": error.get("attempt_id") or "",
            "candidate_id": error.get("candidate_id"),
            "candidate_version": error.get("candidate_version"),
            "stage": error.get("stage") or "",
            "category": error.get("category") or "",
            "code": error.get("code") or "",
            "retryable": bool(error.get("retryable")),
            "repairable": bool(error.get("repairable")),
            "message": error.get("message") or "",
            "source_location": error.get("source_location"),
            "observed": error.get("observed"),
            "allowed_actions": error.get("allowed_actions") or [],
            "forbidden_changes": error.get("forbidden_changes") or [],
            "error_fingerprint": error.get("error_fingerprint") or "",
            "created_at": error.get("created_at") or _now(),
        }

    def _fetch_review(self, campaign: dict[str, Any], exp_id: str, exec_id: Any, review_id: Any) -> dict[str, Any] | None:
        if not review_id:
            return None
        try:
            result = self.dsa.get_review(review_id)
        except (DsaUnavailableError, DsaProtocolError):
            return None
        if result["status"] != "ok":
            return None
        review = (result.get("data") or {}).get("review") or {}
        if not review.get("review_id"):
            return None
        return {
            "review_id": review["review_id"],
            "campaign_id": campaign["campaign_id"],
            "experiment_id": exp_id,
            "execution_id": exec_id or review.get("execution_id") or "",
            "classification": review.get("classification") or "",
            "confidence": float(review.get("confidence") or 0.0),
            "root_causes": review.get("root_causes") or [],
            "gate_assessment": review.get("gate_assessment"),
            "repair_contract": review.get("repair_contract"),
            "recommended_action": review.get("recommended_action") or "",
            "recommended_tests": review.get("recommended_tests") or [],
            "limitations": review.get("limitations") or [],
            "model": review.get("model") or "",
            "prompt_version": review.get("prompt_version") or "",
            "prompt_hash": review.get("prompt_hash") or "",
            "input_evidence_hash": review.get("input_evidence_hash") or "",
            "created_at": review.get("created_at") or _now(),
        }

    def _fetch_evidence(self, campaign: dict[str, Any], exp_id: str, exec_id: str, evidence_id: Any) -> dict[str, Any] | None:
        if not evidence_id:
            return None
        try:
            result = self.dsa.get_evidence(evidence_id)
        except (DsaUnavailableError, DsaProtocolError):
            return None
        if result["status"] != "ok":
            return None
        evidence = (result.get("data") or {}).get("evidence") or {}
        if not evidence.get("evidence_id"):
            return None
        return {
            "evidence_id": evidence["evidence_id"],
            "campaign_id": campaign["campaign_id"],
            "experiment_id": exp_id,
            "execution_id": exec_id,
            "evidence": evidence,
            "created_at": _now(),
        }

    def _block_campaign(self, campaign: dict[str, Any], reason: str) -> None:
        logger.warning("research campaign %s blocked: %s", campaign["campaign_id"], reason)
        self.store.update_campaign(
            campaign["campaign_id"], status="blocked", blocked_reason=reason, updated_at=_now()
        )

    def _recover_cursor_expired(self, campaign: dict[str, Any], exp_id: str, message: str) -> bool:
        """410 ``event_cursor_expired`` 的尽力本地恢复（波次 C）。

        DSA 侧 ``experiment_state_snapshot.v1``（§11.9）恢复资源未实现，
        ``state_snapshot_id`` 恒为 None，无法从 DSA 全量恢复。这里从本地 store
        已落库 events（executions/evidence/reviews 已随事件同步）重建该实验的
        latest 状态：

        - 若该实验没有未终态 execution（queued/running/waiting_review），说明
          已落库事件足以支撑继续推进（决策 / 完成），返回 True；
        - 否则存在依赖 DSA 事件才能推进的 in-flight 执行，返回 False 并
          blocked，blocked_reason 如实标注“游标过期且 DSA 无 state snapshot
          恢复资源（§11.9 未实现）”。

        ``last_event_sequences`` 已停在最新已处理 sequence；调用方视返回决定
        重置游标做全量重新拉取。
        """
        campaign_id = campaign["campaign_id"]
        executions = self.store.list_executions(campaign_id, exp_id)
        active = [e for e in executions if e["status"] in ("queued", "running", "waiting_review")]
        evidence_ids = [e.get("evidence_id") for e in executions if e.get("evidence_id")]
        review_ids = [e.get("review_id") for e in executions if e.get("review_id")]
        if not active:
            logger.warning(
                "event cursor expired for %s; DSA state snapshot restore resource unavailable "
                "（§11.9 未实现），从本地重建 latest 状态（executions=%d evidence=%d reviews=%d）并继续",
                exp_id,
                len(executions),
                len(evidence_ids),
                len(review_ids),
            )
            return True
        self._block_campaign(
            campaign,
            f"event cursor expired for {exp_id}: {message}; "
            f"DSA state snapshot 恢复资源未实现（§11.9），本地状态不足以继续",
        )
        return False

    # ==================================================================
    # 主动推进（§13）
    # ==================================================================

    def _advance(self, campaign: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        if campaign["status"] == "initializing":
            self._start_after_initializing(campaign)
            return
        if campaign["status"] != "running":
            return
        if not self._budget_ok(campaign):
            self._enter_waiting_budget(campaign)
            return
        if campaign.get("universe_snapshot_id") is None:
            # 先构建 universe（DSA §6.12-6.13/§11.7），数据快照引用真实 universe id；
            # 顺序不可颠倒——build_data_snapshot 会校验 universe 引用存在（R08 fail-closed）。
            self._ensure_universe_snapshot(campaign)
            return
        if campaign["data_snapshot_id"] is None or campaign["current_stage"] == "WAIT_DATA":
            self._ensure_data_snapshot(campaign)
            return
        self._ensure_factor_inventory(campaign)
        self._ensure_experiment_queue(campaign)
        for exp in self.store.list_experiments(campaign_id):
            self._advance_experiment(campaign, exp)

    def _start_after_initializing(self, campaign: dict[str, Any]) -> None:
        try:
            result = self.dsa.get_capabilities()
        except (DsaUnavailableError, DsaProtocolError):
            self._mark_unavailable()
            return
        if result["status"] == "ok":
            self.store.update_campaign(campaign["campaign_id"], status="running", updated_at=_now())
        else:
            self._block_campaign(campaign, "DSA capabilities check failed")

    def _ensure_data_snapshot(self, campaign: dict[str, Any]) -> None:
        """R07：数据快照构建是异步的（202 + execution_id）。

        进入 WAIT_DATA 等待构建 execution 终态，且快照质量 valid 后才进
        FACTOR_INVENTORY；failed → campaign blocked（快照构建失败）。
        """
        campaign_id = campaign["campaign_id"]
        config = campaign.get("config", {})
        window = config.get("development_window", {})
        builds = [e for e in self.store.list_executions(campaign_id) if e["execution_type"] == "data_snapshot_build"]
        if not builds:
            payload = {
                "layer": "development",
                "market": config.get("market", "CN"),
                "frequency": config.get("frequency", "1d"),
                "start_date": window.get("start_date"),
                "end_date": window.get("end_date"),
                "adjustment": "qfq",
                "mapping_version": "cn_daily_mapping.v1",
                # 引用已构建的 universe 快照（R08），不用配置的 universe_policy 字符串。
                "universe_snapshot_id": campaign.get("universe_snapshot_id"),
            }
            result = self.dsa.build_data_snapshot(payload, idempotency_key=f"snapshot_{campaign_id}")
            if result["status"] == "error":
                if result.get("retryable"):
                    self._mark_unavailable()
                else:
                    self._block_campaign(campaign, f"data snapshot build failed: {(result.get('error') or {}).get('code')}")
                return
            data = result.get("data") or {}
            exec_id = data.get("execution_id") or f"snap_{campaign_id}"
            self.store.upsert_execution(
                {
                    "execution_id": exec_id,
                    "campaign_id": campaign_id,
                    "experiment_id": "",
                    "execution_type": "data_snapshot_build",
                    "status": "queued",
                    "progress": 0.0,
                    "created_at": _now(),
                }
            )
            fields: dict[str, Any] = {"current_stage": "WAIT_DATA", "updated_at": _now()}
            snap_id = data.get("data_snapshot_id")
            if snap_id:
                # 真实 DSA 同步返回 data_snapshot_id：预登记，但仍等 execution 完成并校验质量
                fields["data_snapshot_id"] = snap_id
            self.store.update_campaign(campaign_id, **fields)
            return
        build = builds[-1]
        if build["status"] in ("queued", "running"):
            self.store.update_campaign(campaign_id, current_stage="WAIT_DATA", updated_at=_now())
            return
        if build["status"] == "failed":
            self._block_campaign(campaign, "data snapshot build failed（快照构建 execution failed）")
            return
        # completed / interrupted / cancelled：解析快照 id 并校验质量 valid 才晋级
        snap_id = campaign.get("data_snapshot_id") or self._resolve_data_snapshot_id(campaign)
        if snap_id and self._data_snapshot_valid(campaign, snap_id):
            self.store.update_campaign(
                campaign_id, data_snapshot_id=snap_id, current_stage="FACTOR_INVENTORY", updated_at=_now()
            )
            return
        # 未解析到有效快照：留在 WAIT_DATA，不强行晋级（下轮再校验）
        self.store.update_campaign(campaign_id, current_stage="WAIT_DATA", updated_at=_now())

    def _data_snapshot_valid(self, campaign: dict[str, Any], snap_id: str) -> bool:
        """R07：快照质量 valid（data_snapshot.v1 ``quality.status == valid``）才算就绪。"""
        try:
            result = self.dsa.get_data_snapshot(snap_id)
        except (DsaUnavailableError, DsaProtocolError):
            return False
        if result["status"] != "ok":
            return False
        data = result.get("data") or {}
        snapshot = data.get("data_snapshot") or data
        quality = snapshot.get("quality") or {}
        status = snapshot.get("status") or quality.get("status") or ""
        return status == "valid"

    def _ensure_universe_snapshot(self, campaign: dict[str, Any]) -> None:
        """R08：数据快照就绪后调用 DSA universe build 获取真实 universe_snapshot_id。

        universe_snapshot_id 只来自 DSA（§6.12-6.13 / §11.7），不再硬编码
        ``universe_pit_001``；构建失败（非可重试）→ campaign blocked。
        """
        campaign_id = campaign["campaign_id"]
        config = campaign.get("config", {})
        window = config.get("development_window", {})
        universe_snapshot_id = f"universe_{campaign_id}"
        payload = {
            "snapshot_id": universe_snapshot_id,
            "market": config.get("market", "CN"),
            "start_date": window.get("start_date"),
            "end_date": window.get("end_date"),
            "selection_policy_version": config.get("universe_policy") or "pit_liquid_active_v1",
        }
        try:
            result = self.dsa.build_universe_snapshot(payload, idempotency_key=universe_snapshot_id)
        except DsaUnavailableError:
            self._mark_unavailable()
            return
        if result["status"] == "error":
            if result.get("retryable"):
                self._mark_unavailable()
            else:
                self._block_campaign(
                    campaign,
                    f"universe snapshot build failed: {(result.get('error') or {}).get('code')}",
                )
            return
        data = result.get("data") or {}
        built_id = data.get("universe_snapshot_id") or data.get("snapshot_id") or universe_snapshot_id
        if not self._universe_snapshot_valid(campaign, built_id):
            # 快照未确认就绪：留在 FACTOR_INVENTORY，下轮再校验
            return
        self.store.update_campaign(campaign_id, universe_snapshot_id=built_id, updated_at=_now())

    def _universe_snapshot_valid(self, campaign: dict[str, Any], uni_id: str) -> bool:
        """R08：universe 快照可被 DSA 查询（存在即 valid，§11.7 只落盘 valid 行）。"""
        try:
            result = self.dsa.get_universe_snapshot(uni_id)
        except (DsaUnavailableError, DsaProtocolError):
            return False
        return result["status"] == "ok"

    def _resolve_data_snapshot_id(self, campaign: dict[str, Any]) -> str | None:
        if self.snapshot_id_resolver is not None:
            try:
                return self.snapshot_id_resolver(campaign, {})
            except (DsaUnavailableError, DsaProtocolError):
                return None
        # Mock 兜底：真实 DSA 在 build 返回或事件中携带 data_snapshot_id；
        # Mock 的 get_data_snapshot 返回 golden snapshot（§6.12 已知缺口）。
        try:
            result = self.dsa.get_data_snapshot(DEFAULT_MOCK_DATA_SNAPSHOT_ID)
        except (DsaUnavailableError, DsaProtocolError):
            return None
        if result["status"] != "ok":
            return None
        data = result.get("data") or {}
        snapshot = data.get("data_snapshot") or data
        return snapshot.get("snapshot_id")

    def _ensure_factor_inventory(self, campaign: dict[str, Any]) -> None:
        inventory = campaign.get("factor_inventory") or {}
        if inventory.get("target") != FACTOR_TARGET:
            self.store.update_campaign(
                campaign["campaign_id"],
                factor_inventory_json=json.dumps(
                    {"target": FACTOR_TARGET, "completed": 0, "skipped": 0, "failed": 0},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                updated_at=_now(),
            )
        # WP5 未实现真实因子计算：此处只记账，因子漏斗由 WP5 填充（§11.10）。

    def _ensure_experiment_queue(self, campaign: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        config = campaign.get("config", {})
        experiments = self.store.list_experiments(campaign_id)
        pending = [e for e in experiments if e["phase"] not in TERMINAL_EXPERIMENT_PHASES]
        if len(pending) >= self.refill_threshold:
            return
        # §13.2：待执行实验少于 5 个自动补充到 20
        for _ in range(max(0, self.queue_target - len(pending))):
            if not self._budget_ok(campaign):
                self._enter_waiting_budget(campaign)
                return
            hypothesis = self.hypothesis_generator(config, experiments)
            if hypothesis is None:
                return
            exp_id = hypothesis.get("experiment_id")
            if not exp_id or self.store.get_experiment(exp_id) is not None:
                continue
            self._register_experiment(campaign, hypothesis)
            experiments.append(hypothesis)

    def _register_experiment(self, campaign: dict[str, Any], hypothesis: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        config = campaign.get("config", {})
        window = config.get("development_window", {})
        exp_id = hypothesis["experiment_id"]
        payload = {
            "experiment_id": exp_id,
            "round_id": hypothesis.get("round_id") or f"round_{campaign_id}",
            "objective": hypothesis.get("objective") or config.get("objective", ""),
            "hypothesis": hypothesis.get("hypothesis", {}),
            "research_window": window,
            "data_snapshot_id": hypothesis.get("data_snapshot_id") or campaign["data_snapshot_id"],
            # R08：universe_snapshot_id 以 campaign 的真实 DSA 快照为准，hypothesis 值仅作兜底
            "universe_snapshot_id": campaign.get("universe_snapshot_id") or hypothesis.get("universe_snapshot_id"),
            "gate_policy_version": campaign["gate_policy_version"],
            "tags": hypothesis.get("tags") or [],
        }
        result = self.dsa.register_experiment(payload, idempotency_key=exp_id)
        if result["status"] == "error":
            if result.get("retryable"):
                self._mark_unavailable()
            else:
                logger.warning("experiment registration failed: %s", (result.get("error") or {}).get("code"))
            return
        now = _now()
        self.store.upsert_experiment(
            {
                "experiment_id": exp_id,
                "campaign_id": campaign_id,
                "round_id": payload["round_id"],
                "objective": payload["objective"],
                "hypothesis": payload["hypothesis"],
                "status": "active",
                "phase": "created",
                "data_snapshot_id": payload["data_snapshot_id"],
                "universe_snapshot_id": payload["universe_snapshot_id"],
                "gate_policy_version": payload["gate_policy_version"],
                "created_at": now,
                "updated_at": now,
            }
        )
        self.store.record_budget_usage("llm_calls")
        self._update_budget_usage(campaign_id)

    # ------------------------------------------------------------------
    # 单实验推进
    # ------------------------------------------------------------------

    def _advance_experiment(self, campaign: dict[str, Any], exp: dict[str, Any]) -> None:
        campaign_id = campaign["campaign_id"]
        exp_id = exp["experiment_id"]
        if exp["phase"] in TERMINAL_EXPERIMENT_PHASES:
            return
        candidates = self.store.list_candidates_for_experiment(campaign_id, exp_id)
        if not candidates:
            if exp["phase"] == "repairing":
                logger.warning("experiment %s marked repairing but no parent candidate; skipping", exp_id)
                return
            self._build_candidate(campaign, exp, parent_version=None, repair_of_error_id=None)
            return
        current = candidates[-1]
        if exp["phase"] == "repairing":
            error, review = self._repair_context(campaign, exp, current)
            self._build_candidate(
                campaign,
                exp,
                parent_version=current["candidate_version"],
                repair_of_error_id=error["error_id"] if error else current.get("repair_of_error_id"),
                error=error,
                review=review,
            )
            return
        if exp["phase"] == "promoted":
            robustness = self._latest_execution(campaign_id, exp_id, current, "robustness")
            if robustness is None or robustness["status"] in ("interrupted", "cancelled"):
                self._start_execution(campaign, exp, current, "robustness")
            elif robustness["status"] == "failed":
                self._maybe_decide(campaign, exp, current)
            elif robustness["status"] == "completed":
                self._maybe_decide(campaign, exp, current)
            return
        if current["status"] == "static_failed":
            self._maybe_decide(campaign, exp, current)
            return
        if current["status"] in ("validated", "sandbox_pending", "registered"):
            sandbox = self._latest_execution(campaign_id, exp_id, current, "strategy_sandbox_smoke")
            if sandbox is None or sandbox["status"] in ("interrupted", "cancelled"):
                self._start_execution(campaign, exp, current, "strategy_sandbox_smoke")
                return
            if sandbox["status"] == "failed":
                self._maybe_decide(campaign, exp, current)
                return
            if sandbox["status"] == "completed":
                backtest = self._latest_execution(campaign_id, exp_id, current, "portfolio_backtest")
                if backtest is None or backtest["status"] in ("interrupted", "cancelled"):
                    self._start_execution(campaign, exp, current, "portfolio_backtest")
                    return
                if backtest["status"] == "failed":
                    self._maybe_decide(campaign, exp, current)
                    return
                if backtest["status"] == "completed":
                    self._maybe_decide(campaign, exp, current)
                    return

    def _latest_execution(
        self, campaign_id: str, exp_id: str, candidate: dict[str, Any], exec_type: str
    ) -> dict[str, Any] | None:
        matches = [
            e
            for e in self.store.list_executions(campaign_id, exp_id)
            if e.get("candidate_id") == candidate["candidate_id"]
            and e.get("candidate_version") == candidate["candidate_version"]
            and e.get("execution_type") == exec_type
        ]
        return matches[-1] if matches else None

    def _repair_context(
        self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        exp_id = exp["experiment_id"]
        execs = [
            e
            for e in self.store.list_executions(campaign["campaign_id"], exp_id)
            if e.get("candidate_id") == current["candidate_id"]
            and e.get("candidate_version") == current["candidate_version"]
            and e.get("error_id")
        ]
        error = self.store.get_error(execs[-1]["error_id"]) if execs and execs[-1].get("error_id") else None
        if error is None:
            errors = [
                e
                for e in self.store.list_errors(campaign["campaign_id"])
                if e.get("candidate_id") == current["candidate_id"]
            ]
            error = errors[-1] if errors else None
        review = None
        if execs:
            review_id = self.store.get_execution(execs[-1]["execution_id"])
            review_id = review_id.get("review_id") if review_id else None
            review = self.store.get_review(review_id) if review_id else None
        return error, review

    def _build_candidate(
        self,
        campaign: dict[str, Any],
        exp: dict[str, Any],
        *,
        parent_version: int | None,
        repair_of_error_id: str | None,
        error: dict[str, Any] | None = None,
        review: dict[str, Any] | None = None,
    ) -> None:
        campaign_id = campaign["campaign_id"]
        exp_id = exp["experiment_id"]
        candidates = self.store.list_candidates_for_experiment(campaign_id, exp_id)
        candidate_id = candidates[0]["candidate_id"] if candidates else f"candidate_{exp_id}_c1"
        version = next_candidate_version(candidates)
        if parent_version is not None:
            error, review = self._repair_context(campaign, exp, candidates[-1]) if error is None else (error, review)
            allowed, reason = self._repair_allowed(campaign, exp, candidate_id, error)
            if not allowed:
                logger.warning("repair blocked for %s: %s", candidate_id, reason)
                self._apply_decision(
                    campaign,
                    exp,
                    {"action": "abandon_candidate", "phase": "rejected", "rationale": f"自动修复被阻止: {reason}"},
                )
                return
        source_code, manifest = self.code_generator(
            exp,
            candidate_id,
            version,
            parent_version=parent_version,
            error=error,
            review=review,
        )
        source_sha = sha256_hex(source_code.encode("utf-8"))
        if source_hash_used(candidates, source_sha):
            logger.warning("source hash %s already used for %s; not resubmitting", source_sha, candidate_id)
            self._apply_decision(
                campaign,
                exp,
                {"action": "abandon_candidate", "phase": "rejected", "rationale": "相同源码哈希不得作为新版本重提（§8.3）"},
            )
            return
        payload = {
            "candidate_id": candidate_id,
            "candidate_version": version,
            "parent_version": parent_version,
            "repair_of_error_id": repair_of_error_id,
            "contract_version": STRATEGY_CONTRACT_V1,
            "source_code": source_code,
            "source_sha256": source_sha,
            "manifest": manifest,
        }
        result = self.dsa.register_candidate(exp_id, payload)
        if result["status"] == "error":
            code = (result.get("error") or {}).get("code")
            if code == "candidate_version_exists":
                logger.warning("candidate %s v%d already exists (recovery path)", candidate_id, version)
                return
            logger.warning("candidate registration failed: %s", code)
            self._apply_decision(
                campaign, exp, {"action": "request_human_review", "phase": "blocked", "rationale": f"候选注册失败: {code}"}
            )
            return
        now = _now()
        status = (result.get("data") or {}).get("status", "registered")
        self.store.upsert_candidate(
            {
                "candidate_id": candidate_id,
                "candidate_version": version,
                "campaign_id": campaign_id,
                "experiment_id": exp_id,
                "parent_version": parent_version,
                "repair_of_error_id": repair_of_error_id,
                "contract_version": STRATEGY_CONTRACT_V1,
                "manifest": manifest,
                "source_sha256": source_sha,
                "status": status,
                "created_at": now,
            }
        )
        if parent_version is not None:
            change_summary = "；".join((review or {}).get("repair_contract", {}).get("allowed_changes", []) or [])
            preserved = (review or {}).get("repair_contract", {}).get("must_preserve", []) or []
            fingerprint = (error or {}).get("error_fingerprint", "")
            self.store.insert_repair(
                {
                    "candidate_id": candidate_id,
                    "candidate_version": version,
                    "campaign_id": campaign_id,
                    "experiment_id": exp_id,
                    "parent_version": parent_version,
                    "repair_of_error_id": repair_of_error_id or "",
                    "error_fingerprint": fingerprint,
                    "change_summary": change_summary or "自动修复版本",
                    "preserved_constraints": preserved,
                    "created_at": now,
                }
            )
        self.store.record_budget_usage("candidate_versions")
        self._update_budget_usage(campaign_id)
        if status == "validated":
            self.store.update_experiment(exp_id, phase="static_validated", updated_at=now)
        elif status == "static_failed":
            self.store.update_experiment(exp_id, phase="static_failed", updated_at=now)
        else:
            self.store.update_experiment(exp_id, phase="candidate_built", updated_at=now)

    def _repair_allowed(
        self, campaign: dict[str, Any], exp: dict[str, Any], candidate_id: str, error: dict[str, Any] | None
    ) -> tuple[bool, str]:
        repairs = [r for r in self.store.list_repairs(campaign["campaign_id"]) if r["candidate_id"] == candidate_id]
        candidates = self.store.list_candidates_for_experiment(campaign["campaign_id"], exp["experiment_id"])
        if error is None:
            return repair_budget_remaining(repairs, max_repair_versions=self.max_repair_versions) > 0, "budget_ok"
        return can_repair(
            error.get("error_fingerprint", ""),
            repairs,
            candidates,
            "",
            max_repair_versions=self.max_repair_versions,
            max_consecutive_same_fingerprint=self.max_consecutive_fingerprint,
        )

    def _start_execution(
        self, campaign: dict[str, Any], exp: dict[str, Any], candidate: dict[str, Any], exec_type: str
    ) -> None:
        universe_snapshot_id = campaign.get("universe_snapshot_id")
        if not universe_snapshot_id:
            # R08：universe_snapshot_id 必须来自 DSA universe build；缺失时 blocked，
            # 绝不回退硬编码 "universe_pit_001"。
            self._block_campaign(
                campaign,
                "universe_snapshot_id missing: cannot start execution without a real DSA universe snapshot",
            )
            return
        exp_id = exp["experiment_id"]
        config = campaign.get("config", {})
        window = config.get("development_window", {})
        payload = {
            "execution_type": exec_type,
            "candidate_id": candidate["candidate_id"],
            "candidate_version": candidate["candidate_version"],
            "data_snapshot_id": campaign["data_snapshot_id"],
            "universe_snapshot_id": universe_snapshot_id,
            "window": window,
            "gate_policy_version": campaign["gate_policy_version"],
            "runtime_profile": "sandbox_smoke" if exec_type == "strategy_sandbox_smoke" else exec_type,
            "seed": (candidate.get("manifest") or {}).get("seed", 191),
        }
        key = f"start_{exec_type}_{candidate['candidate_id']}_v{candidate['candidate_version']}"
        request_sha = canonical_sha256(payload)
        prior = self.store.idempotency_lookup("start_execution", key)
        if prior is not None:
            if prior["request_sha256"] == request_sha and prior.get("resource_id"):
                exec_id = prior["resource_id"]
                if self.store.get_execution(exec_id) is None:
                    self._rehydrate_execution(campaign, exp, candidate, exec_type, exec_id)
                return
            key = f"{key}_r{uuid.uuid4().hex[:8]}"
        try:
            result = self.dsa.start_execution(exp_id, payload, idempotency_key=key)
        except DsaUnavailableError:
            self._mark_unavailable()
            return
        if result["status"] == "error":
            if result.get("retryable"):
                self._mark_unavailable()
            else:
                logger.warning("start execution failed: %s", (result.get("error") or {}).get("code"))
            return
        data = result.get("data") or {}
        exec_id = data.get("execution_id")
        if not exec_id:
            return
        self.store.save_idempotency(
            "start_execution", key, request_sha, result.get("http_status", 200), json.dumps(data), exec_id
        )
        self._upsert_execution_row(campaign, exp, candidate, exec_type, exec_id)

    def _rehydrate_execution(
        self, campaign: dict[str, Any], exp: dict[str, Any], candidate: dict[str, Any], exec_type: str, exec_id: str
    ) -> None:
        self._upsert_execution_row(campaign, exp, candidate, exec_type, exec_id)

    def _upsert_execution_row(
        self, campaign: dict[str, Any], exp: dict[str, Any], candidate: dict[str, Any], exec_type: str, exec_id: str
    ) -> None:
        self.store.upsert_execution(
            {
                "execution_id": exec_id,
                "campaign_id": campaign["campaign_id"],
                "experiment_id": exp["experiment_id"],
                "candidate_id": candidate["candidate_id"],
                "candidate_version": candidate["candidate_version"],
                "execution_type": exec_type,
                "status": "queued",
                "progress": 0.0,
                "current_stage": "queued",
                "created_at": _now(),
            }
        )

    # ------------------------------------------------------------------
    # 决策（§6.10 / §8.3 / §13.1 VIBE_DECISION）
    # ------------------------------------------------------------------

    def _maybe_decide(self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]) -> None:
        decision = self._compute_decision(campaign, exp, current)
        if decision is None:
            return
        self._apply_decision(campaign, exp, decision)

    def _compute_decision(
        self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any] | None:
        campaign_id = campaign["campaign_id"]
        exp_id = exp["experiment_id"]
        execs = [
            e
            for e in self.store.list_executions(campaign_id, exp_id)
            if e.get("candidate_id") == current["candidate_id"]
            and e.get("candidate_version") == current["candidate_version"]
        ]
        latest = execs[-1] if execs else None
        error = self.store.get_error(latest["error_id"]) if latest and latest.get("error_id") else None
        review = None
        if latest:
            review = self.store.get_review(latest["review_id"]) if latest.get("review_id") else None

        # 静态检查失败（无执行）
        if current["status"] == "static_failed":
            error = error or self._latest_error_for_candidate(campaign, exp, current)
            if error is None:
                return None
            return self._decision_for_error(campaign, exp, current, error, review, stage="static_validate")

        if latest is None:
            return None

        # robustness 通过 → 完成
        if latest["status"] == "completed" and latest["execution_type"] == "robustness":
            return {"action": "complete_experiment", "phase": "completed", "rationale": "robustness 验证通过，实验完成"}
        if latest["status"] == "failed" and latest["execution_type"] == "robustness":
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "robustness 验证失败，候选淘汰"}

        # 执行失败 → 按错误分类决定
        if latest["status"] == "failed":
            if error is None:
                return None
            return self._decision_for_error(campaign, exp, current, error, review, stage="execution")

        # 组合回测完成 → 按 gate + Reviewer 决定
        if latest["status"] == "completed" and latest["execution_type"] == "portfolio_backtest":
            evidence = self.store.get_evidence(latest["evidence_id"]) if latest.get("evidence_id") else None
            review_pending, review_requested_at = self._execution_review_pending(campaign, latest)
            return self._decision_for_completed_backtest(
                campaign,
                exp,
                current,
                evidence,
                review,
                review_pending=review_pending,
                review_requested_at=review_requested_at,
            )

        return None

    def _decision_for_error(
        self,
        campaign: dict[str, Any],
        exp: dict[str, Any],
        current: dict[str, Any],
        error: dict[str, Any],
        review: dict[str, Any] | None,
        *,
        stage: str,
    ) -> dict[str, Any]:
        category = error.get("category", "")
        if category == "SECURITY_ERROR":
            return {"action": "quarantine_candidate", "phase": "quarantined", "rationale": "安全违规，候选隔离"}
        if category == "DATA_ERROR":
            return {"action": "create_data_task", "phase": "waiting_data", "rationale": "数据错误，创建数据修复任务而非改代码掩盖"}
        if category == "RESEARCH_REJECTION":
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "研究拒绝，归档候选并生成新假设"}
        if category == "INFRA_ERROR":
            retries = self._infra_retry_count(campaign, exp, current)
            if retries < MAX_INFRA_RETRIES:
                return {
                    "action": "retry_same_execution",
                    "phase": exp["phase"],
                    "rationale": f"基础设施错误，退避重试（第 {retries + 1}/{MAX_INFRA_RETRIES} 次）",
                    "retry_execution_type": self._last_execution_type(campaign, exp, current),
                }
            return {"action": "request_human_review", "phase": "blocked", "rationale": "基础设施重试次数耗尽，需要人工介入"}
        # CODE_RUNTIME / CONTRACT_ERROR：等待 Review 后决定修复或淘汰
        review_action = (review or {}).get("recommended_action")
        if review is None:
            if error.get("repairable") and "submit_candidate_revision" in (error.get("allowed_actions") or []):
                return self._revision_decision(campaign, exp, current, error, review)
            return None  # 等待 Reviewer
        if review_action in ("quarantine_candidate",):
            return {"action": "quarantine_candidate", "phase": "quarantined", "rationale": "Reviewer 建议隔离"}
        if review_action in ("abandon_candidate",):
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "Reviewer 建议淘汰"}
        if review_action == "submit_candidate_revision":
            allowed, _reason = self._repair_allowed(campaign, exp, current["candidate_id"], error)
            if allowed:
                return self._revision_decision(campaign, exp, current, error, review)
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "修复预算耗尽或相同错误指纹连续出现，提前停止修复"}
        if review_action == "retry_same_execution":
            return {"action": "retry_same_execution", "phase": exp["phase"], "rationale": "Reviewer 建议重试同一执行",
                    "retry_execution_type": self._last_execution_type(campaign, exp, current)}
        if error.get("repairable"):
            return self._revision_decision(campaign, exp, current, error, review)
        return {"action": "abandon_candidate", "phase": "rejected", "rationale": "错误不可修复，候选淘汰"}

    def _revision_decision(
        self,
        campaign: dict[str, Any],
        exp: dict[str, Any],
        current: dict[str, Any],
        error: dict[str, Any],
        review: dict[str, Any] | None,
    ) -> dict[str, Any]:
        allowed, reason = self._repair_allowed(campaign, exp, current["candidate_id"], error)
        if not allowed:
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": f"自动修复被阻止: {reason}"}
        rationale = "接受 DSA Reviewer 判断，提交候选修复版本"
        if review:
            root_causes = [r.get("finding", "") for r in review.get("root_causes", [])]
            if root_causes:
                rationale = f"接受 DSA Reviewer 判断（{'; '.join(root_causes)}），提交候选修复版本"
        return {"action": "submit_candidate_revision", "phase": "repairing", "rationale": rationale}

    def _decision_for_completed_backtest(
        self,
        campaign: dict[str, Any],
        exp: dict[str, Any],
        current: dict[str, Any],
        evidence: dict[str, Any] | None,
        review: dict[str, Any] | None,
        *,
        review_pending: bool = False,
        review_requested_at: str | None = None,
    ) -> dict[str, Any] | None:
        gate_results = (evidence or {}).get("evidence", {}).get("gate_results", []) if evidence else []
        # fail-closed（R04）：空 gate_results 表示没有真实回测指标（not_evaluable），
        # 不允许当作“无失败”晋级，必须等真实 gate 结果再人工评审。
        if not gate_results:
            return {
                "action": "request_human_review",
                "phase": "blocked",
                "rationale": "evidence 无 gate 结果（not_evaluable），不允许晋级；等待真实回测指标",
            }
        gate_failed = any(g.get("status") == "fail" for g in gate_results)
        if gate_failed:
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "gate 检查失败（研究拒绝），归档候选并生成新假设"}
        if review and review.get("classification") == "research_rejected":
            return {"action": "abandon_candidate", "phase": "rejected", "rationale": "Reviewer 判定为研究拒绝"}
        if review and review.get("recommended_action") == "submit_candidate_revision":
            # §9.3 Reviewer 意见不是裁判结果：执行与 gate 均通过时 Vibe 可接受并记录理由
            return {
                "action": "accept_result",
                "phase": "promoted",
                "rationale": "执行完成且 gate 通过；Reviewer 提出代码隐患建议但无对应执行错误，Vibe 接受结果并保留 Reviewer 意见",
            }
        # R09 决策 B：Reviewer 是旁路意见。已发出 review 请求但未产出且未超时 → 等待；
        # 超时后 accept，但 rationale 与最终报告标记“缺少 DSA 独立评审”（§9.3）。
        if review is None and review_pending and not self._review_timeout_elapsed(review_requested_at):
            return None
        if review is None and review_pending:
            return {
                "action": "accept_result",
                "phase": "promoted",
                "rationale": f"执行完成且 gate 通过；等待 DSA 独立评审超时（§9.3），接受结果（{REVIEW_BYPASS_MARKER}）",
            }
        return {"action": "accept_result", "phase": "promoted", "rationale": "执行完成且 gate 通过，接受结果并进入 robustness"}

    def _execution_review_pending(self, campaign: dict[str, Any], exec_record: dict[str, Any] | None) -> tuple[bool, str | None]:
        """R09：该 execution 是否在等待 DSA 独立评审（review.requested 已发出且未 review.completed）。

        返回 ``(pending, requested_at)``。review.failed 也计入缺失评审：从失败事件时间起
        进入 §9.3 超时窗口；review.completed 到货后视为已产出，清除 pending。
        """
        exec_id = exec_record.get("execution_id") if exec_record else None
        if not exec_id:
            return False, None
        pending_at: str | None = None
        for event in self.store.list_events(campaign["campaign_id"]):
            if (event.get("payload") or {}).get("execution_id") != exec_id:
                continue
            event_type = event.get("event_type", "")
            if event_type == "review.requested":
                pending_at = event.get("created_at")
            elif event_type == "review.completed":
                pending_at = None
            elif event_type == "review.failed":
                pending_at = pending_at or event.get("created_at")
        return (pending_at is not None), pending_at

    def _review_timeout_elapsed(self, requested_at: str | None) -> bool:
        """R09：是否已超过 §9.3 默认 300 秒的独立评审等待窗口。"""
        if not requested_at:
            return True
        try:
            requested = datetime.fromisoformat(str(requested_at))
        except ValueError:
            return True
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - requested).total_seconds() >= self.review_timeout_seconds

    def _apply_decision(self, campaign: dict[str, Any], exp: dict[str, Any], decision: dict[str, Any]) -> None:
        action = decision["action"]
        if action not in _DECISION_ACTIONS:
            raise CampaignValidationError(f"unknown decision action: {action}")
        exp_id = exp["experiment_id"]
        campaign_id = campaign["campaign_id"]
        decision_id = _short_id("decision")
        current = self.store.list_candidates_for_experiment(campaign_id, exp_id)
        current = current[-1] if current else None
        next_version = None
        if action == "submit_candidate_revision" and current is not None:
            next_version = current["candidate_version"] + 1
        review_id = None
        evidence_ids: list[str] = []
        for exec_record in self.store.list_executions(campaign_id, exp_id):
            if exec_record.get("review_id"):
                review_id = exec_record["review_id"]
            if exec_record.get("evidence_id"):
                evidence_ids.append(exec_record["evidence_id"])
        record = {
            "decision_id": decision_id,
            "campaign_id": campaign_id,
            "experiment_id": exp_id,
            "round_id": exp.get("round_id") or f"round_{campaign_id}",
            "action": action,
            "rationale": decision.get("rationale", ""),
            "source_evidence_ids": evidence_ids[:32],
            "source_review_id": review_id,
            "next_candidate_id": current["candidate_id"] if current else None,
            "next_candidate_version": next_version,
            "created_at": _now(),
        }
        inserted = self.store.insert_decision(record)
        if inserted:
            # 先写不可变 decision，再创建候选 / 启动任务（§6.10）
            self._record_decision_to_dsa(exp_id, record)
        phase = decision.get("phase")
        if action == "retry_same_execution":
            retry_type = decision.get("retry_execution_type")
            self.store.update_experiment(exp_id, last_decision_action=action, updated_at=_now())
            if retry_type and current is not None:
                self._start_execution(campaign, exp, current, retry_type)
            return
        if phase:
            self.store.update_experiment(exp_id, phase=phase, last_decision_action=action, updated_at=_now())
        else:
            self.store.update_experiment(exp_id, last_decision_action=action, updated_at=_now())

    def _record_decision_to_dsa(self, exp_id: str, record: dict[str, Any]) -> None:
        payload = {
            "decision_id": record["decision_id"],
            "round_id": record["round_id"],
            "action": record["action"],
            "rationale": record["rationale"],
            "source_evidence_ids": record["source_evidence_ids"],
            "source_review_id": record["source_review_id"],
            "next_candidate_id": record["next_candidate_id"],
            "next_candidate_version": record["next_candidate_version"],
        }
        try:
            self.dsa.record_decision(exp_id, payload, idempotency_key=record["decision_id"])
        except (DsaUnavailableError, DsaProtocolError):
            logger.warning("could not record decision %s to DSA", record["decision_id"])

    def _latest_error_for_candidate(
        self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any] | None:
        errors = [
            e
            for e in self.store.list_errors(campaign["campaign_id"])
            if e.get("candidate_id") == current["candidate_id"]
        ]
        if errors:
            return self.store.get_error(errors[-1]["error_id"]) or errors[-1]
        return None

    def _infra_retry_count(self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]) -> int:
        count = 0
        for exec_record in self.store.list_executions(campaign["campaign_id"], exp["experiment_id"]):
            if (
                exec_record.get("candidate_id") == current["candidate_id"]
                and exec_record.get("candidate_version") == current["candidate_version"]
                and exec_record.get("status") == "failed"
                and exec_record.get("error_id")
            ):
                error = self.store.get_error(exec_record["error_id"])
                if error and error.get("category") == "INFRA_ERROR":
                    count += 1
        return count

    def _last_execution_type(self, campaign: dict[str, Any], exp: dict[str, Any], current: dict[str, Any]) -> str:
        matches = [
            e
            for e in self.store.list_executions(campaign["campaign_id"], exp["experiment_id"])
            if e.get("candidate_id") == current["candidate_id"]
            and e.get("candidate_version") == current["candidate_version"]
        ]
        return matches[-1]["execution_type"] if matches else "strategy_sandbox_smoke"

    # ------------------------------------------------------------------
    # 预算 / 状态同步（§13.3）
    # ------------------------------------------------------------------

    def _budget_ok(self, campaign: dict[str, Any]) -> bool:
        config = campaign.get("config", {})
        budgets = config.get("budgets", {}) or {}
        llm_limit = int(budgets.get("llm_calls_rolling_24h") or DEFAULT_BUDGET_LLM_CALLS)
        version_limit = int(budgets.get("candidate_versions_rolling_24h") or DEFAULT_BUDGET_CANDIDATE_VERSIONS)
        if self.store.rolling_budget_usage("llm_calls") >= llm_limit:
            return False
        if self.store.rolling_budget_usage("candidate_versions") >= version_limit:
            return False
        return True

    def _enter_waiting_budget(self, campaign: dict[str, Any]) -> None:
        if campaign["status"] != "waiting_budget":
            logger.info("research campaign %s entered waiting_budget (§13.3)", campaign["campaign_id"])
            self.store.update_campaign(campaign["campaign_id"], status="waiting_budget", updated_at=_now())

    def _update_budget_usage(self, campaign_id: str) -> None:
        usage = {
            "llm_calls_rolling_24h": self.store.rolling_budget_usage("llm_calls"),
            "candidate_versions_rolling_24h": self.store.rolling_budget_usage("candidate_versions"),
        }
        self.store.update_campaign(
            campaign_id,
            budget_usage_json=json.dumps(usage, ensure_ascii=False, sort_keys=True),
            updated_at=_now(),
        )

    def _sync_campaign_status(self, campaign_id: str) -> None:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            return
        if campaign["status"] in ("cancelled", "blocked", "completed"):
            return
        experiments = self.store.list_experiments(campaign_id)
        executions = self.store.list_executions(campaign_id)
        stage = self._dominant_stage(campaign, experiments)
        active_executions = [e for e in executions if e["status"] in ("queued", "running", "waiting_review")]
        # R06：completed 判定必须先排除 blocked / waiting_data，且要求无活跃 execution；
        # 否则 blocked 实验会被 “全部实验终态” 误判为 completed（原 1509 的 blocked 检查走不到）。
        if (
            experiments
            and not any(e["phase"] in ("waiting_data", "blocked") for e in experiments)
            and all(e["phase"] in TERMINAL_EXPERIMENT_PHASES for e in experiments)
            and not active_executions
        ):
            self.store.update_campaign(
                campaign_id,
                status="completed",
                current_stage="COMPLETED",
                queue_counts_json=json.dumps(self._queue_counts(experiments), ensure_ascii=False, sort_keys=True),
                updated_at=_now(),
            )
            # R13：进入 completed 时自动生成最终中文报告（幂等，可安全重入）。
            self._generate_report_on_completion(campaign_id)
            return
        if any(e["phase"] in ("waiting_data", "blocked") for e in experiments):
            self.store.update_campaign(
                campaign_id, status="blocked", current_stage=stage, queue_counts_json=json.dumps(self._queue_counts(experiments), ensure_ascii=False, sort_keys=True), updated_at=_now()
            )
            return
        if not self._budget_ok(campaign):
            self.store.update_campaign(
                campaign_id,
                status="waiting_budget",
                current_stage=stage,
                queue_counts_json=json.dumps(self._queue_counts(experiments), ensure_ascii=False, sort_keys=True),
                updated_at=_now(),
            )
            return
        self.store.update_campaign(
            campaign_id,
            current_stage=stage,
            queue_counts_json=json.dumps(self._queue_counts(experiments), ensure_ascii=False, sort_keys=True),
            updated_at=_now(),
        )

    @staticmethod
    def _dominant_stage(campaign: dict[str, Any], experiments: list[dict[str, Any]]) -> str:
        if not experiments:
            return campaign.get("current_stage") or "WAIT_DATA"
        phases = {e["phase"] for e in experiments}
        if any(e["phase"] in ("waiting_data", "blocked") for e in experiments):
            return "VIBE_DECISION"
        if "repairing" in phases or "static_failed" in phases or "sandbox_failed" in phases or "backtest_failed" in phases:
            return "VIBE_DECISION"
        if "robustness_failed" in phases:
            return "VIBE_DECISION"
        if "promoted" in phases:
            return "ROBUSTNESS"
        if "backtest_passed" in phases:
            return "DSA_REVIEW"
        if "sandbox_passed" in phases:
            return "PORTFOLIO_BACKTEST"
        if "static_validated" in phases:
            return "SANDBOX_VALIDATE"
        if "candidate_built" in phases:
            return "CANDIDATE_BUILD"
        if "created" in phases:
            return "HYPOTHESIS_GENERATION"
        return campaign.get("current_stage") or "WAIT_DATA"

    @staticmethod
    def _queue_counts(experiments: list[dict[str, Any]]) -> dict[str, int]:
        running = 0
        blocked = 0
        for e in experiments:
            if e["phase"] in ("waiting_data", "blocked"):
                blocked += 1
            elif e["phase"] in (
                "sandbox_passed",
                "backtest_passed",
                "promoted",
                "repairing",
                "static_failed",
                "sandbox_failed",
                "backtest_failed",
                "robustness_failed",
            ):
                running += 1
        total = len(experiments)
        pending = max(0, total - running - blocked)
        return {"pending": pending, "running": running, "blocked": blocked}

    def _mark_unavailable(self) -> None:
        self._dsa_unavailable = True
        self._consecutive_failures += 1

    def reset_availability(self) -> None:
        self._dsa_unavailable = False
        self._consecutive_failures = 0

    # ==================================================================
    # 报告数据
    # ==================================================================

    def _report_data(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.get_campaign(campaign_id)
        if campaign is None:
            raise CampaignNotFoundError(campaign_id)
        decisions = self.store.list_decisions(campaign_id)
        return {
            "campaign": campaign,
            "experiments": self.store.list_experiments(campaign_id),
            "candidates": self.store.list_candidates(campaign_id),
            "executions": self.store.list_executions(campaign_id),
            "errors": self.store.list_errors(campaign_id),
            "reviews": self.store.list_reviews(campaign_id),
            "evidence": self.store.list_evidence(campaign_id),
            "decisions": decisions,
            "repairs": self.store.list_repairs(campaign_id),
            "events": self.store.list_events(campaign_id),
            # R09：是否存在在缺少 DSA 独立评审（§9.3 超时旁路）情况下做出的决定
            "review_bypassed": any(REVIEW_BYPASS_MARKER in (d.get("rationale") or "") for d in decisions),
        }
