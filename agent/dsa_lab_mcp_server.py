"""Standalone stdio MCP bridge to Daily Stock Analysis' Backtest Lab service.

Spawned by the Vibe-Trading agent via ``~/.vibe-trading/agent.json``
(``mcpServers.dsa_lab``). Exposes one MCP tool per DSA action; each tool
proxies to the loopback-only DSA Backtest Lab at ``http://127.0.0.1:8011``.

Intentionally imports NO ``src.*`` modules — DSA-specific logic lives only
in this file so the core tool registry / env schema stay generic.
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

mcp = FastMCP("dsa-lab")


def _validated_base_url(raw: str) -> str:
    """Validate that ``raw`` is a loopback-only HTTP URL and normalize it."""
    parsed = urlparse(raw.strip())
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("DSA_LAB_URL must be a loopback HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("DSA_LAB_URL contains unsupported URL components")
    return raw.strip().rstrip("/")


def _resource_id(value: Any, *, name: str) -> str:
    """Normalize a batch/run id to the allowed ``[A-Za-z0-9_-]`` charset."""
    normalized = str(value or "").strip()
    if not _RESOURCE_ID_RE.fullmatch(normalized):
        raise ValueError(f"{name} must contain only letters, numbers, '_' or '-'")
    return normalized


def _coerce_timeout(raw: Any) -> float:
    """Coerce a timeout env value to a positive float."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("DSA_LAB_TIMEOUT_SECONDS must be a positive number")
    if value <= 0:
        raise ValueError("DSA_LAB_TIMEOUT_SECONDS must be a positive number")
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
    detail: Any = None,
) -> str:
    """Return a JSON error envelope in the DSA tool's stable shape."""
    payload: dict[str, Any] = {"status": "error", "action": action, "error": error}
    if http_status is not None:
        payload["http_status"] = http_status
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
            base_url if base_url is not None else os.getenv("DSA_LAB_URL", _DEFAULT_DSA_LAB_URL)
        )
        effective_timeout = _coerce_timeout(
            timeout if timeout is not None else os.getenv("DSA_LAB_TIMEOUT_SECONDS", str(_DEFAULT_DSA_LAB_TIMEOUT))
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
