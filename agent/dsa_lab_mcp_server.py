"""Standalone stdio MCP bridge to Daily Stock Analysis' Backtest Lab service.

Spawned by the Vibe-Trading agent via ``~/.vibe-trading/agent.json``
(``mcpServers.dsa_lab``). Exposes one MCP tool per DSA action; each tool
proxies to the loopback-only DSA Backtest Lab at ``http://127.0.0.1:8011``.

The bridge also exposes the research-loop.v1 tools (``get_research_loop_*``,
``register_research_*``, ``start_research_execution``, ``poll_research_events``
and the data tools) per spec §7.1. Those proxy to the loopback-only DSA
research-loop API at ``http://127.0.0.1:8011`` by default (``DSA_RESEARCH_LOOP_URL``).

Intentionally imports NO ``src.*`` modules — DSA-specific logic lives only
in this file so the core tool registry / env schema stay generic.

MCP-layer constraints (§7.1): loopback-only URLs, ID charset validation,
``source_code`` ≤ 64 KiB, HTTP 429/502/503/504 marked retryable, and 400/403/
409/422 not automatically retried. The bridge never decides a repair strategy —
it only forwards DSA's structured status and errors.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP

_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_RESPONSE_BYTES = 10_000_000
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_DEFAULT_DSA_LAB_URL = "http://127.0.0.1:8011"
_DEFAULT_DSA_LAB_TIMEOUT = 300.0
_DEFAULT_DSA_RESEARCH_LOOP_URL = "http://127.0.0.1:8011"
_SOURCE_CODE_MAX_BYTES = 65_536
_RESEARCH_LOOP_PREFIX = "/internal/v1/research-loop"
_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})

mcp = FastMCP("dsa-lab")


def _validated_base_url(raw: str, *, var_name: str = "DSA_LAB_URL") -> str:
    """Validate that ``raw`` is a loopback-only HTTP URL and normalize it."""
    parsed = urlparse(raw.strip())
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(f"{var_name} must be a loopback HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{var_name} contains unsupported URL components")
    return raw.strip().rstrip("/")


def _resource_id(value: Any, *, name: str) -> str:
    """Normalize a batch/run id to the allowed ``[A-Za-z0-9_-]`` charset."""
    normalized = str(value or "").strip()
    if not _RESOURCE_ID_RE.fullmatch(normalized):
        raise ValueError(f"{name} must contain only letters, numbers, '_' or '-'")
    return normalized


def _coerce_timeout(raw: Any, *, var_name: str = "DSA_LAB_TIMEOUT_SECONDS") -> float:
    """Coerce a timeout env value to a positive float."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{var_name} must be a positive number")
    if value <= 0:
        raise ValueError(f"{var_name} must be a positive number")
    return value


def _default_client(*, base_url: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout, trust_env=False)


def _request_spec(
    *,
    action: str,
    payload: Any,
    batch_id: Any,
    run_id: Any,
    page: int,
    page_size: int,
    idempotency_key: Any,
) -> tuple[str, str, dict[str, Any] | None, dict[str, str], dict[str, int] | None]:
    """Map a DSA action to its HTTP method, path, body, headers and params."""
    headers: dict[str, str] = {}
    params: dict[str, int] | None = None
    body: dict[str, Any] | None = None

    if action == "health":
        return "GET", "/health", None, headers, None
    if action == "refresh_catalog":
        return "POST", "/internal/v1/csv-catalog/refresh", None, headers, None
    if action == "list_strategies":
        return "GET", "/internal/v1/strategies", None, headers, None
    if action == "run_factor_research":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for run_factor_research")
        return "POST", "/internal/v1/research/factors", payload, headers, None
    if action in {"preflight", "create_batch"}:
        if not isinstance(payload, dict):
            raise ValueError(f"payload is required for {action}")
        body = payload
        if action == "preflight":
            return "POST", "/internal/v1/preflight", body, headers, None
        key = str(idempotency_key or uuid.uuid4().hex).strip()
        if not _RESOURCE_ID_RE.fullmatch(key):
            raise ValueError("idempotency_key must contain only letters, numbers, '_' or '-'")
        headers["Idempotency-Key"] = key
        return "POST", "/internal/v1/batches", body, headers, None
    if action == "list_batches":
        params = {"page": page, "page_size": min(page_size, 100)}
        return "GET", "/internal/v1/batches", None, headers, params
    if action in {"get_batch", "list_runs", "cancel_batch"}:
        normalized = _resource_id(batch_id, name="batch_id")
        if action == "get_batch":
            return "GET", f"/internal/v1/batches/{normalized}", None, headers, None
        if action == "cancel_batch":
            return "POST", f"/internal/v1/batches/{normalized}/cancel", None, headers, None
        params = {"page": page, "page_size": page_size}
        return "GET", f"/internal/v1/batches/{normalized}/runs", None, headers, params
    if action == "get_run":
        normalized = _resource_id(run_id, name="run_id")
        return "GET", f"/internal/v1/runs/{normalized}", None, headers, None
    raise ValueError(f"unsupported action: {action}")


def _error(
    action: str,
    error: str,
    *,
    http_status: int | None = None,
    retryable: bool | None = None,
    detail: Any = None,
) -> str:
    """Return a JSON error envelope in the DSA tool's stable shape."""
    payload: dict[str, Any] = {"status": "error", "action": action, "error": error}
    if http_status is not None:
        payload["http_status"] = http_status
    if retryable is not None:
        payload["retryable"] = retryable
    if detail is not None:
        payload["detail"] = detail
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _lab_request(
    action: str,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    payload: Any = None,
    batch_id: Any = None,
    run_id: Any = None,
    page: int = 1,
    page_size: int = 50,
    idempotency_key: Any = None,
    client_factory: Callable[..., Any] | None = None,
) -> str:
    """Issue one DSA Backtest Lab request and return a JSON envelope string.

    ``base_url`` / ``timeout`` default to the ``DSA_LAB_URL`` /
    ``DSA_LAB_TIMEOUT_SECONDS`` env vars. ``client_factory`` is an injectable
    seam for tests (a callable returning a context manager with a
    ``request(method, path, **kwargs)`` method); when omitted a real
    ``httpx.Client`` with ``trust_env=False`` is used.
    """
    try:
        effective_base_url = _validated_base_url(
            base_url if base_url is not None else os.getenv("DSA_LAB_URL", _DEFAULT_DSA_LAB_URL),
            var_name="DSA_LAB_URL",
        )
        effective_timeout = _coerce_timeout(
            timeout if timeout is not None else os.getenv("DSA_LAB_TIMEOUT_SECONDS", str(_DEFAULT_DSA_LAB_TIMEOUT)),
            var_name="DSA_LAB_TIMEOUT_SECONDS",
        )
        if page < 1 or not 1 <= page_size <= 200:
            return _error(action, "invalid_pagination")
        method, path, body, headers, params = _request_spec(
            action=action,
            payload=payload,
            batch_id=batch_id,
            run_id=run_id,
            page=page,
            page_size=page_size,
            idempotency_key=idempotency_key,
        )
        factory = client_factory or _default_client
        with factory(base_url=effective_base_url, timeout=effective_timeout) as client:
            response = client.request(method, path, json=body, headers=headers, params=params)
    except (TypeError, ValueError) as exc:
        return _error(action, str(exc))
    except httpx.RequestError:
        return _error(action, "dsa_backtest_lab_unavailable")

    if len(response.content) > _MAX_RESPONSE_BYTES:
        return _error(action, "dsa_response_too_large", http_status=response.status_code)
    try:
        resp_body: Any = response.json()
    except ValueError:
        resp_body = {"message": response.text[:1000]}
    if response.is_error:
        detail = resp_body.get("detail", resp_body) if isinstance(resp_body, dict) else resp_body
        return _error(action, "dsa_request_failed", http_status=response.status_code, detail=detail)
    return json.dumps({"status": "ok", "action": action, "data": resp_body}, ensure_ascii=False, allow_nan=False)


@mcp.tool()
def health() -> str:
    """Check the DSA Backtest Lab health endpoint."""
    return _lab_request("health")


@mcp.tool()
def refresh_catalog() -> str:
    """Refresh the local DSA factor CSV catalog cache."""
    return _lab_request("refresh_catalog")


@mcp.tool()
def list_strategies() -> str:
    """List all DSA strategies available in the local Backtest Lab."""
    return _lab_request("list_strategies")


@mcp.tool()
def preflight(payload: dict[str, Any]) -> str:
    """Validate a strategy experiment against the DSA Backtest Lab without submitting it."""
    return _lab_request("preflight", payload=payload)


@mcp.tool()
def create_batch(payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Submit a DSA strategy batch for execution."""
    return _lab_request("create_batch", payload=payload, idempotency_key=idempotency_key)


@mcp.tool()
def list_batches(page: int = 1, page_size: int = 50) -> str:
    """List DSA strategy batches, newest first."""
    return _lab_request("list_batches", page=page, page_size=page_size)


@mcp.tool()
def get_batch(batch_id: str) -> str:
    """Fetch one DSA strategy batch by id."""
    return _lab_request("get_batch", batch_id=batch_id)


@mcp.tool()
def list_runs(batch_id: str, page: int = 1, page_size: int = 50) -> str:
    """List execution runs for a DSA strategy batch."""
    return _lab_request("list_runs", batch_id=batch_id, page=page, page_size=page_size)


@mcp.tool()
def get_run(run_id: str) -> str:
    """Fetch one DSA execution run by id."""
    return _lab_request("get_run", run_id=run_id)


@mcp.tool()
def cancel_batch(batch_id: str) -> str:
    """Cancel a DSA strategy batch and its queued runs."""
    return _lab_request("cancel_batch", batch_id=batch_id)


@mcp.tool()
def run_factor_research(payload: dict[str, Any]) -> str:
    """Run bounded DSA factor research over the local factor catalog."""
    return _lab_request("run_factor_research", payload=payload)


# ---------------------------------------------------------------------------
# research-loop.v1 tools（§7.1）
# ---------------------------------------------------------------------------


def _http_status_retryable(http_status: int) -> bool:
    """§7.1: 429/502/503/504 可重试；400/403/409/422 不自动重试。"""
    return http_status in _RETRYABLE_HTTP_STATUS


def _source_code_too_large(source_code: Any) -> bool:
    """Return whether *source_code* exceeds the 64 KiB contract limit."""
    if not isinstance(source_code, str):
        return True
    return len(source_code.encode("utf-8")) > _SOURCE_CODE_MAX_BYTES


def _research_loop_spec(
    *,
    action: str,
    payload: Any,
    experiment_id: Any,
    resource_id: Any,
    after_sequence: int,
    page_size: int,
    idempotency_key: Any,
) -> tuple[str, str, dict[str, Any] | None, dict[str, str], dict[str, Any] | None]:
    """Map a research-loop.v1 action to its HTTP method/path/body/headers/params."""
    headers: dict[str, str] = {}
    params: dict[str, Any] | None = None

    def _key() -> str:
        key = str(idempotency_key or uuid.uuid4().hex).strip()
        if not _RESOURCE_ID_RE.fullmatch(key):
            raise ValueError("idempotency_key must contain only letters, numbers, '_' or '-'")
        return key

    if action == "get_research_loop_capabilities":
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/capabilities", None, headers, None
    if action == "register_research_experiment":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for register_research_experiment")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/experiments", payload, headers, None
    if action == "register_strategy_candidate":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for register_strategy_candidate")
        source = payload.get("source_code")
        if not isinstance(source, str):
            raise ValueError("source_code is required for register_strategy_candidate")
        if _source_code_too_large(source):
            raise ValueError("source_code exceeds 64 KiB limit")
        normalized = _resource_id(experiment_id, name="experiment_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/experiments/{normalized}/candidates", payload, headers, None
    if action == "start_research_execution":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for start_research_execution")
        normalized = _resource_id(experiment_id, name="experiment_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/experiments/{normalized}/executions", payload, headers, None
    if action == "get_research_execution":
        normalized = _resource_id(resource_id, name="execution_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/executions/{normalized}", None, headers, None
    if action == "poll_research_events":
        normalized = _resource_id(experiment_id, name="experiment_id")
        params = {"after_sequence": int(after_sequence), "page_size": max(1, min(int(page_size), 200))}
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/experiments/{normalized}/events", None, headers, params
    if action == "get_research_error":
        normalized = _resource_id(resource_id, name="error_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/errors/{normalized}", None, headers, None
    if action == "get_research_evidence":
        normalized = _resource_id(resource_id, name="evidence_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/evidence/{normalized}", None, headers, None
    if action == "get_research_review":
        normalized = _resource_id(resource_id, name="review_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/reviews/{normalized}", None, headers, None
    if action == "record_research_decision":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for record_research_decision")
        normalized = _resource_id(experiment_id, name="experiment_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/experiments/{normalized}/decisions", payload, headers, None
    if action == "cancel_research_execution":
        normalized = _resource_id(resource_id, name="execution_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/executions/{normalized}/cancel", None, headers, None
    if action == "build_data_snapshot":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for build_data_snapshot")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/data-snapshots/build", payload, headers, None
    if action == "get_data_snapshot":
        normalized = _resource_id(resource_id, name="data_snapshot_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/data-snapshots/{normalized}", None, headers, None
    if action == "export_market_panel":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for export_market_panel")
        normalized = _resource_id(resource_id, name="data_snapshot_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/data-snapshots/{normalized}/panel-exports", payload, headers, None
    if action == "create_artifact_upload":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for create_artifact_upload")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/artifact-uploads", payload, headers, None
    if action == "complete_artifact_upload":
        normalized = _resource_id(resource_id, name="upload_id")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/artifact-uploads/{normalized}/complete", None, headers, None
    if action == "register_factor_snapshot":
        if not isinstance(payload, dict):
            raise ValueError("payload is required for register_factor_snapshot")
        headers["Idempotency-Key"] = _key()
        return "POST", f"{_RESEARCH_LOOP_PREFIX}/factor-snapshots", payload, headers, None
    if action == "get_factor_snapshot":
        normalized = _resource_id(resource_id, name="factor_snapshot_id")
        return "GET", f"{_RESEARCH_LOOP_PREFIX}/factor-snapshots/{normalized}", None, headers, None
    raise ValueError(f"unsupported action: {action}")


def _research_loop_request(
    action: str,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    payload: Any = None,
    experiment_id: Any = None,
    resource_id: Any = None,
    after_sequence: int = 0,
    page_size: int = 100,
    idempotency_key: Any = None,
    client_factory: Callable[..., Any] | None = None,
) -> str:
    """Issue one DSA research-loop.v1 request and return a JSON envelope string."""
    try:
        effective_base_url = _validated_base_url(
            base_url if base_url is not None else os.getenv("DSA_RESEARCH_LOOP_URL", _DEFAULT_DSA_RESEARCH_LOOP_URL),
            var_name="DSA_RESEARCH_LOOP_URL",
        )
        effective_timeout = _coerce_timeout(
            timeout if timeout is not None else os.getenv("DSA_LAB_TIMEOUT_SECONDS", str(_DEFAULT_DSA_LAB_TIMEOUT)),
            var_name="DSA_LAB_TIMEOUT_SECONDS",
        )
        method, path, body, headers, params = _research_loop_spec(
            action=action,
            payload=payload,
            experiment_id=experiment_id,
            resource_id=resource_id,
            after_sequence=after_sequence,
            page_size=page_size,
            idempotency_key=idempotency_key,
        )
        factory = client_factory or _default_client
        with factory(base_url=effective_base_url, timeout=effective_timeout) as client:
            response = client.request(method, path, json=body, headers=headers, params=params)
    except (TypeError, ValueError) as exc:
        return _error(action, str(exc))
    except httpx.RequestError:
        return _error(action, "dsa_research_loop_unavailable", http_status=503, retryable=True)

    if len(response.content) > _MAX_RESPONSE_BYTES:
        return _error(action, "dsa_response_too_large", http_status=response.status_code)
    try:
        resp_body: Any = response.json()
    except ValueError:
        resp_body = {"message": response.text[:1000]}
    if response.is_error:
        detail = resp_body.get("error", resp_body) if isinstance(resp_body, dict) else resp_body
        return _error(
            action,
            "dsa_request_failed",
            http_status=response.status_code,
            retryable=_http_status_retryable(response.status_code),
            detail=detail,
        )
    return json.dumps({"status": "ok", "action": action, "data": resp_body}, ensure_ascii=False, allow_nan=False)


@mcp.tool()
def get_research_loop_capabilities() -> str:
    """List the DSA research-loop protocol capabilities and limits."""
    return _research_loop_request("get_research_loop_capabilities")


@mcp.tool()
def register_research_experiment(payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Register a research experiment (hypothesis) with DSA (§6.2)."""
    return _research_loop_request("register_research_experiment", payload=payload, idempotency_key=idempotency_key)


@mcp.tool()
def register_strategy_candidate(experiment_id: str, payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Register a strategy candidate version with DSA (§6.3). source_code ≤ 64 KiB."""
    return _research_loop_request(
        "register_strategy_candidate",
        payload=payload,
        experiment_id=experiment_id,
        idempotency_key=idempotency_key,
    )


@mcp.tool()
def start_research_execution(experiment_id: str, payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Start a DSA research execution (§6.4)."""
    return _research_loop_request(
        "start_research_execution",
        payload=payload,
        experiment_id=experiment_id,
        idempotency_key=idempotency_key,
    )


@mcp.tool()
def get_research_execution(execution_id: str) -> str:
    """Fetch a DSA research execution status summary (§6.5)."""
    return _research_loop_request("get_research_execution", resource_id=execution_id)


@mcp.tool()
def poll_research_events(experiment_id: str, after_sequence: int = 0, page_size: int = 100) -> str:
    """Incrementally read research events after a sequence cursor (§6.6)."""
    return _research_loop_request(
        "poll_research_events", experiment_id=experiment_id, after_sequence=after_sequence, page_size=page_size
    )


@mcp.tool()
def get_research_error(error_id: str) -> str:
    """Fetch a normalized DSA execution error (§6.7 / §8.1)."""
    return _research_loop_request("get_research_error", resource_id=error_id)


@mcp.tool()
def get_research_evidence(evidence_id: str) -> str:
    """Fetch DSA evidence summary and artifact metadata (§6.8)."""
    return _research_loop_request("get_research_evidence", resource_id=evidence_id)


@mcp.tool()
def get_research_review(review_id: str) -> str:
    """Fetch a DSA Research Reviewer report (§6.9 / §9.2)."""
    return _research_loop_request("get_research_review", resource_id=review_id)


@mcp.tool()
def record_research_decision(experiment_id: str, payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Record Vibe's immutable next-step decision (§6.10)."""
    return _research_loop_request(
        "record_research_decision", payload=payload, experiment_id=experiment_id, idempotency_key=idempotency_key
    )


@mcp.tool()
def cancel_research_execution(execution_id: str) -> str:
    """Cancel a DSA research execution (§6.11)."""
    return _research_loop_request("cancel_research_execution", resource_id=execution_id)


@mcp.tool()
def build_data_snapshot(payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Build a DSA data snapshot asynchronously (§6.12)."""
    return _research_loop_request("build_data_snapshot", payload=payload, idempotency_key=idempotency_key)


@mcp.tool()
def get_data_snapshot(data_snapshot_id: str) -> str:
    """Fetch a DSA data snapshot manifest (§6.12 / §11.6)."""
    return _research_loop_request("get_data_snapshot", resource_id=data_snapshot_id)


@mcp.tool()
def export_market_panel(data_snapshot_id: str, payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Export a standard market panel from a valid data snapshot (§6.13)."""
    return _research_loop_request(
        "export_market_panel", payload=payload, resource_id=data_snapshot_id, idempotency_key=idempotency_key
    )


@mcp.tool()
def create_artifact_upload(payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Create a large artifact upload session (§6.14). Content is uploaded separately."""
    return _research_loop_request("create_artifact_upload", payload=payload, idempotency_key=idempotency_key)


@mcp.tool()
def complete_artifact_upload(upload_id: str, idempotency_key: str = "") -> str:
    """Complete an artifact upload and obtain the immutable artifact_id (§6.14)."""
    return _research_loop_request(
        "complete_artifact_upload", resource_id=upload_id, idempotency_key=idempotency_key
    )


@mcp.tool()
def register_factor_snapshot(payload: dict[str, Any], idempotency_key: str = "") -> str:
    """Register a factor snapshot manifest (§6.15)."""
    return _research_loop_request("register_factor_snapshot", payload=payload, idempotency_key=idempotency_key)


@mcp.tool()
def get_factor_snapshot(factor_snapshot_id: str) -> str:
    """Fetch a factor snapshot manifest and validation summary (§6.15)."""
    return _research_loop_request("get_factor_snapshot", resource_id=factor_snapshot_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
