"""Vibe 最终中文研究报告生成（§15.3）。

报告固定包含 9 项内容：

1. 本轮目标与预注册假设；
2. 使用的 Vibe 因子及其家族；
3. 候选代码版本和自动修复谱系；
4. DSA 原始指标和失败清单；
5. DSA Reviewer 的独立意见；
6. Vibe 接受或反对该意见的理由；
7. 晋级、淘汰、隔离、等待数据或创建 challenger 的决定；
8. 下一轮自动队列；
9. 所有 artifact、数据、公式、代码和规则哈希。

渲染是纯函数：输入从 Controller Store 聚合的 :func:`ReportData`，输出固定结构
的中文 Markdown。报告只做事实陈列，不把工程通过、回测高收益描述为已证明 Alpha
（§20）。
"""

from __future__ import annotations

from typing import Any

_REPORT_VERSION = "report.v1"

# 报告段标题（§15.3 固定结构）
_SECTIONS = [
    ("1. 研究目标与预注册假设", "research_goal"),
    ("2. Vibe 因子及家族", "factors"),
    ("3. 候选代码版本与修复谱系", "candidates"),
    ("4. DSA 原始指标与失败清单", "evidence"),
    ("5. DSA Reviewer 独立意见", "reviews"),
    ("6. Vibe 接受 / 反对意见的理由", "decisions"),
    ("7. 晋级 / 淘汰 / 隔离 / 等待 / Challenger 决定", "outcomes"),
    ("8. 下一轮自动队列", "next_queue"),
    ("9. Artifact / 数据 / 公式 / 代码 / 规则哈希", "hashes"),
]


def report_section_titles() -> list[str]:
    """Return the fixed Chinese section titles for the report."""
    return [f"## {title}" for title, _ in _SECTIONS]


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _fmt(value: Any) -> str:
    if value is None:
        return "（无）"
    return str(value)


def _bullet(items: list[Any], default: str = "（无）") -> str:
    if not items:
        return f"- {default}"
    return "\n".join(f"- {_fmt(item)}" for item in items)


def _render_goal(data: dict[str, Any]) -> str:
    campaign = data.get("campaign") or {}
    lines = [
        f"- Campaign: `{campaign.get('campaign_id', '')}`",
        f"- Research Goal: `{campaign.get('research_goal_id', '')}`",
        f"- 目标: {_fmt(campaign.get('config', {}).get('objective'))}",
        f"- 市场 / 频率: {campaign.get('config', {}).get('market')} / {campaign.get('config', {}).get('frequency')}",
        f"- 开发窗口: {campaign.get('config', {}).get('development_window', {})}",
        f"- 因子范围: {_fmt(campaign.get('config', {}).get('factor_scope'))}",
        f"- 策略 / 预算: {campaign.get('config', {}).get('automation', {})}",
    ]
    hypotheses = []
    for exp in data.get("experiments", []):
        hyp = exp.get("hypothesis", {})
        hypotheses.append(
            f"### 假设 {exp.get('experiment_id')}\n"
            f"- 机制: {_fmt(hyp.get('mechanism'))}\n"
            f"- 预测: {_fmt(hyp.get('prediction'))}\n"
            f"- 失败条件: {_bullet(hyp.get('failure_conditions', []))}\n"
            f"- 因子: {_bullet(hyp.get('factor_ids', []))}"
        )
    if hypotheses:
        lines.append("\n**预注册假设：**\n\n" + "\n\n".join(hypotheses))
    return "\n".join(lines)


def _render_factors(data: dict[str, Any]) -> str:
    campaign = data.get("campaign") or {}
    inventory = campaign.get("factor_inventory", {})
    lines = [
        f"- 因子库存: target={inventory.get('target', 0)} completed={inventory.get('completed', 0)} "
        f"skipped={inventory.get('skipped', 0)} failed={inventory.get('failed', 0)}",
        "- 因子族：Qlib158（154）、GTJA191（191）、Academic（12）；Alpha101 为迁移队列，Fundamental 等待 PIT 字段。",
    ]
    used_factor_ids: set[str] = set()
    for exp in data.get("experiments", []):
        used_factor_ids.update(exp.get("hypothesis", {}).get("factor_ids", []))
    for cand in data.get("candidates", []):
        used_factor_ids.update(cand.get("manifest", {}).get("factor_ids", []))
    if used_factor_ids:
        lines.append(f"- 本轮实际使用的因子 ID：{', '.join(sorted(used_factor_ids))}")
    return "\n".join(lines)


def _render_candidates(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    repairs = data.get("repairs", [])
    repair_by_key = {(r.get("candidate_id"), r.get("candidate_version")): r for r in repairs}
    experiments = {e["experiment_id"]: e for e in data.get("experiments", [])}
    lines = []
    for cand in candidates:
        cid = cand.get("candidate_id")
        ver = cand.get("candidate_version")
        exp = experiments.get(cand.get("experiment_id"), {})
        repair = repair_by_key.get((cid, ver))
        parts = [
            f"### {cid} v{ver}",
            f"- 实验: {cand.get('experiment_id')}",
            f"- 状态: {cand.get('status')}",
            f"- 父版本: {_fmt(cand.get('parent_version'))}",
            f"- 修复目标错误: {_fmt(cand.get('repair_of_error_id'))}",
            f"- 源码 SHA-256: `{cand.get('source_sha256')}`",
            f"- 实验阶段: {exp.get('phase', '')}",
        ]
        if repair:
            parts.append(
                f"- 修复变更摘要: {_fmt(repair.get('change_summary'))}\n"
                f"- 保留约束: {_bullet(repair.get('preserved_constraints', []))}"
            )
        lines.append("\n".join(parts))
    if not candidates:
        lines.append("（本轮未注册候选）")
    return "\n".join(lines)


def _render_evidence(data: dict[str, Any]) -> str:
    evidence_list = data.get("evidence", [])
    errors = data.get("errors", [])
    lines = []
    for ev in evidence_list:
        bundle = ev.get("evidence", {})
        metric_parts = []
        for key, value in (bundle.get("metrics") or {}).items():
            status = value.get("status", "") if isinstance(value, dict) else ""
            metric_parts.append(f"{key}={status}")
        lines.append(
            f"### 证据 {ev.get('evidence_id')}\n"
            f"- 执行: {ev.get('execution_id')} / 类型: {bundle.get('execution_type')}\n"
            f"- 窗口: {bundle.get('window')}\n"
            f"- 数据质量: {bundle.get('data_quality')}\n"
            f"- gate 结果: {_bullet([g.get('check_id') + '=' + g.get('status', '') for g in bundle.get('gate_results', [])])}\n"
            f"- 指标模块: {_bullet(metric_parts)}\n"
            f"- bundle SHA-256: `{bundle.get('bundle_sha256')}`"
        )
    if not evidence_list:
        lines.append("（本轮尚无完成执行的证据包）")
    lines.append("\n**失败清单：**")
    if errors:
        for err in errors:
            lines.append(
                f"- `{err.get('error_id')}` [{err.get('category')}/{err.get('code')}] "
                f"阶段={err.get('stage')} 指纹=`{err.get('error_fingerprint')}` "
                f"可修复={err.get('repairable')} 可重试={err.get('retryable')}"
            )
    else:
        lines.append("- （无执行错误）")
    return "\n".join(lines)


def _render_reviews(data: dict[str, Any]) -> str:
    reviews = data.get("reviews", [])
    if not reviews:
        return "（本轮缺少 DSA 独立评审，报告已标记）"
    lines = []
    for review in reviews:
        cause_parts = []
        for cause in review.get("root_causes", []):
            cause_parts.append(f"{cause.get('evidence_ref')}: {cause.get('finding')}")
        repair = review.get("repair_contract", {}) or {}
        lines.append(
            f"### 评审 {review.get('review_id')}\n"
            f"- 执行: {review.get('execution_id')}\n"
            f"- 分类: {review.get('classification')} / 置信度: {review.get('confidence')}\n"
            f"- 根因: {_bullet(cause_parts)}\n"
            f"- 修复契约: 可修复={bool(repair.get('repairable'))} "
            f"允许={repair.get('allowed_changes', [])} 禁止={repair.get('forbidden_changes', [])}\n"
            f"- 推荐动作: {review.get('recommended_action')}\n"
            f"- 推荐测试: {_bullet(review.get('recommended_tests', []))}"
        )
    return "\n\n".join(lines)


def _render_decisions(data: dict[str, Any]) -> str:
    decisions = data.get("decisions", [])
    if not decisions:
        return "（本轮尚无 Vibe 决定）"
    lines = []
    for decision in decisions:
        lines.append(
            f"### 决定 {decision.get('decision_id')}\n"
            f"- 动作: `{decision.get('action')}`\n"
            f"- 依据: {_fmt(decision.get('rationale'))}\n"
            f"- 证据: {_bullet(decision.get('source_evidence_ids', []))} "
            f"评审: {_fmt(decision.get('source_review_id'))}\n"
            f"- 下一候选: {_fmt(decision.get('next_candidate_id'))} v{_fmt(decision.get('next_candidate_version'))}"
        )
        if "缺少 DSA 独立评审" in (decision.get("rationale") or ""):
            lines.append("- ⚠ 该决定在缺少 DSA 独立评审（§9.3 超时旁路）情况下做出")
    return "\n\n".join(lines)


def _render_outcomes(data: dict[str, Any]) -> str:
    experiments = data.get("experiments", [])
    if not experiments:
        return "（暂无实验决定）"
    lines = []
    for exp in experiments:
        lines.append(
            f"- `{exp.get('experiment_id')}` 阶段=`{exp.get('phase')}` 最后决定=`{_fmt(exp.get('last_decision_action'))}`"
        )
    return "\n".join(lines)


def _render_next_queue(data: dict[str, Any]) -> str:
    campaign = data.get("campaign") or {}
    queue = campaign.get("queue_counts", {})
    lines = [
        f"- 状态: {campaign.get('status')} / 当前阶段: {campaign.get('current_stage')}",
        f"- 队列: pending={queue.get('pending', 0)} running={queue.get('running', 0)} blocked={queue.get('blocked', 0)}",
        f"- 事件游标: {campaign.get('last_event_sequences', {})}",
    ]
    pending_experiments = [e for e in data.get("experiments", []) if e.get("phase") not in _terminal_phases()]
    if pending_experiments:
        lines.append("- 待处理实验: " + ", ".join(e["experiment_id"] for e in pending_experiments))
    return "\n".join(lines)


def _render_hashes(data: dict[str, Any]) -> str:
    campaign = data.get("campaign") or {}
    lines = [
        f"- 配置哈希 config_sha256: `{campaign.get('config_sha256')}`",
        f"- 协议 bundle 哈希 protocol_bundle_sha256: `{campaign.get('protocol_bundle_sha256')}`",
        f"- 数据快照 data_snapshot_id: `{_fmt(campaign.get('data_snapshot_id'))}`",
        f"- 股票池 universe_snapshot_id: `{_fmt(campaign.get('universe_snapshot_id'))}`",
        f"- 判分规则 gate_policy_version: `{_fmt(campaign.get('gate_policy_version'))}`",
    ]
    lines.append("- 候选源码哈希:")
    for cand in data.get("candidates", []):
        lines.append(
            f"  - `{cand.get('candidate_id')}` v{cand.get('candidate_version')} = `{cand.get('source_sha256')}`"
        )
    for ev in data.get("evidence", []):
        bundle = ev.get("evidence", {})
        if bundle:
            lines.append(f"- 证据 {ev.get('evidence_id')} bundle_sha256: `{bundle.get('bundle_sha256')}`")
    lines.append("- artifact 引用:")
    artifacts: dict[str, str] = {}
    for ev in data.get("evidence", []):
        for artifact in (ev.get("evidence") or {}).get("artifacts", []):
            artifacts[artifact.get("artifact_id", "")] = artifact.get("sha256", "")
    for artifact_id, sha in artifacts.items():
        lines.append(f"  - `{artifact_id}` = `{sha}`")
    if not artifacts:
        lines.append("  - （无 artifact）")
    return "\n".join(lines)


def _terminal_phases() -> set[str]:
    return {"rejected", "quarantined", "completed", "blocked"}


def render_campaign_report(data: dict[str, Any]) -> str:
    """Render the fixed-structure Vibe Chinese research report (§15.3)."""
    campaign = data.get("campaign") or {}
    renderers = {
        "research_goal": _render_goal,
        "factors": _render_factors,
        "candidates": _render_candidates,
        "evidence": _render_evidence,
        "reviews": _render_reviews,
        "decisions": _render_decisions,
        "outcomes": _render_outcomes,
        "next_queue": _render_next_queue,
        "hashes": _render_hashes,
    }
    header = [
        "# Vibe 研究 Campaign 中文报告",
        f"- Campaign: `{campaign.get('campaign_id', '')}`",
        f"- 状态: {campaign.get('status', '')} / 阶段: {campaign.get('current_stage', '')}",
        f"- 报告版本: {_REPORT_VERSION} / 生成时间: {_now()}",
        "",
        "> 免责声明：本报告仅用于研究、模拟与回测，不构成实盘投资建议或盈利证明（§20）。",
        "",
    ]
    if data.get("review_bypassed"):
        header.append("> ⚠ 本报告包含在缺少 DSA 独立评审（§9.3 超时旁路）情况下做出的决定。")
        header.append("")
    body: list[str] = []
    for title, key in _SECTIONS:
        body.append(f"## {title}")
        body.append(renderers[key](data))
        body.append("")
    return "\n".join(header + body)
