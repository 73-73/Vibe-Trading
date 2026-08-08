"""Research Controller pipeline stages and transition helpers (§13.1).

The campaign-level ``current_stage`` follows the §13.1 pipeline; experiment
level flow is tracked via per-experiment ``phase`` values below. Stage and phase
names are stable strings used for API responses and the final report.
"""

from __future__ import annotations

# §13.1 主状态机阶段（Campaign 级 current_stage）
PIPELINE_STAGES = [
    "WAIT_DATA",
    "FACTOR_INVENTORY",
    "FACTOR_COMPUTE",
    "HYPOTHESIS_GENERATION",
    "CANDIDATE_BUILD",
    "PREFLIGHT",
    "SANDBOX_VALIDATE",
    "PORTFOLIO_BACKTEST",
    "DSA_REVIEW",
    "VIBE_DECISION",
]

# VIBE_DECISION 之后的决策去向（§13.1）
DECISION_TERMINALS = [
    "REPAIR_PENDING",
    "NEW_HYPOTHESIS",
    "ROBUSTNESS",
    "GATE_CHALLENGER",
    "REJECTED",
    "QUARANTINED",
    "COMPLETED",
]

CAMPAIGN_STATUS = {
    "initializing",
    "running",
    "paused",
    "waiting_budget",
    "blocked",
    "completed",
    "cancelled",
}

CAMPAIGN_TERMINAL_STATUS = {"completed", "cancelled", "blocked"}

# Experiment 级 phase（Controller 内部流转）
EXPERIMENT_PHASES = {
    "created",  # 实验已注册，待构建候选
    "candidate_built",  # 候选已注册，等待静态检查
    "static_validated",  # 静态检查通过
    "static_failed",  # 静态检查失败（已记录 error）
    "sandbox_passed",  # 沙箱冒烟通过
    "sandbox_failed",  # 沙箱冒烟失败
    "backtest_passed",  # 组合回测通过
    "backtest_failed",  # 组合回测失败
    "repairing",  # 已决定提交修复版本
    "promoted",  # 已接受，进入 robustness
    "rejected",  # 研究拒绝 / 淘汰
    "quarantined",  # 安全违规隔离
    "waiting_data",  # 等待数据任务
    "robustness_passed",  # robustness 通过
    "robustness_failed",  # robustness 失败
    "completed",  # 实验完成
    "blocked",  # 需要人工介入
}

TERMINAL_EXPERIMENT_PHASES = {"rejected", "quarantined", "completed", "blocked", "waiting_data"}

# 候选状态（对齐 DSA §5.2）
CANDIDATE_STATUS = {
    "registered",
    "static_failed",
    "sandbox_pending",
    "sandbox_failed",
    "validated",
    "rejected",
    "quarantined",
    "promoted",
}

# 执行状态（对齐 DSA §5.3）
EXECUTION_STATUS = {
    "queued",
    "running",
    "waiting_review",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
}

# §6.6 轮询节奏
POLL_INTERVAL_ACTIVE_SECONDS = 2.0
POLL_INTERVAL_REVIEW_SECONDS = 5.0
POLL_INTERVAL_IDLE_SECONDS = 15.0
POLL_BACKOFF_MAX_SECONDS = 60.0

# §13.3 滚动资源护栏默认值
DEFAULT_BUDGET_LLM_CALLS = 100
DEFAULT_BUDGET_CANDIDATE_VERSIONS = 50
DEFAULT_SANDBOX_CONCURRENCY = 1
DEFAULT_FACTOR_BATCH_CONCURRENCY = 2
MAX_REPAIR_VERSIONS_PER_CANDIDATE = 3
MAX_INFRA_RETRIES = 5


def next_pipeline_stage(stage: str) -> str:
    """Return the next §13.1 pipeline stage after *stage*."""
    if stage in PIPELINE_STAGES:
        idx = PIPELINE_STAGES.index(stage)
        if idx + 1 < len(PIPELINE_STAGES):
            return PIPELINE_STAGES[idx + 1]
    return stage


def is_terminal_experiment_phase(phase: str) -> bool:
    return phase in TERMINAL_EXPERIMENT_PHASES


def compute_poll_interval(
    *,
    any_active_execution: bool,
    any_waiting_review: bool,
    backoff_seconds: float = 0.0,
) -> float:
    """Return the poll interval per §6.6.

    - DSA backoff (after an unavailable response) uses exponential backoff
      capped at 60 s and takes precedence.
    - Active executions poll every 2 s.
    - Waiting for the Reviewer polls every 5 s.
    - Idle polls every 15 s.
    """
    if backoff_seconds > 0:
        return min(backoff_seconds, POLL_BACKOFF_MAX_SECONDS)
    if any_active_execution:
        return POLL_INTERVAL_ACTIVE_SECONDS
    if any_waiting_review:
        return POLL_INTERVAL_REVIEW_SECONDS
    return POLL_INTERVAL_IDLE_SECONDS
