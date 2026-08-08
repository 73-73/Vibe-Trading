"""自动修复谱系守卫（§8.3）。

默认策略：

- 每个逻辑候选最多 3 个自动修复版本（``MAX_REPAIR_VERSIONS_PER_CANDIDATE``）。
- 相同错误指纹连续出现 2 次时提前停止修复。
- 相同源码哈希不得作为新版本重新提交。
- ``SECURITY_ERROR`` 的旧版本永久保留为 quarantined；修复必须生成新版本。

这些规则是纯函数，便于在 Controller 决策前做确定性校验。
"""

from __future__ import annotations

from typing import Any

from ..state_machine.machine import MAX_REPAIR_VERSIONS_PER_CANDIDATE


class RepairBlockedError(RuntimeError):
    """Raised / returned reason when an automatic repair is not allowed."""


def repair_budget_remaining(repairs: list[dict[str, Any]], *, max_repair_versions: int = MAX_REPAIR_VERSIONS_PER_CANDIDATE) -> int:
    """Return how many automatic repair versions remain for a candidate."""
    return max(0, max_repair_versions - len(repairs))


def consecutive_fingerprint_count(error_fingerprint: str, repairs: list[dict[str, Any]]) -> int:
    """Count trailing repairs (for one candidate) whose fingerprint matches.

    The returned value includes the current failure, so two consecutive
    occurrences of the same fingerprint yield 2.
    """
    trailing = 0
    for repair in reversed(repairs):
        if repair.get("error_fingerprint") == error_fingerprint:
            trailing += 1
        else:
            break
    return trailing + 1


def source_hash_used(candidates: list[dict[str, Any]], source_sha256: str) -> bool:
    """Return whether *source_sha256* was already submitted for the candidate."""
    return any(cand.get("source_sha256") == source_sha256 for cand in candidates)


def can_repair(
    error_fingerprint: str,
    repairs: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    source_sha256: str,
    *,
    max_repair_versions: int = MAX_REPAIR_VERSIONS_PER_CANDIDATE,
    max_consecutive_same_fingerprint: int = 2,
) -> tuple[bool, str]:
    """Decide whether a new repair version may be submitted (§8.3).

    Args:
        error_fingerprint: Normalized error fingerprint of the current failure.
        repairs: Repair rows already recorded for this logical candidate.
        candidates: Candidate rows already recorded for this logical candidate.
        source_sha256: Hash of the proposed new source code.

    Returns:
        ``(allowed, reason)`` where a False reason is one of
        ``repair_exhausted`` / ``consecutive_fingerprint_stop`` /
        ``source_hash_duplicate``.
    """
    if repair_budget_remaining(repairs, max_repair_versions=max_repair_versions) <= 0:
        return False, "repair_exhausted"
    if consecutive_fingerprint_count(error_fingerprint, repairs) >= max_consecutive_same_fingerprint:
        return False, "consecutive_fingerprint_stop"
    if source_hash_used(candidates, source_sha256):
        return False, "source_hash_duplicate"
    return True, "repairable"


def next_candidate_version(candidates: list[dict[str, Any]]) -> int:
    """Return the next candidate version (max existing + 1, starting at 1)."""
    versions = [int(cand.get("candidate_version", 0)) for cand in candidates]
    return (max(versions) + 1) if versions else 1
