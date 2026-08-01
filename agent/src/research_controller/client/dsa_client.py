"""Loopback-only HTTP client for the DSA ``research-loop.v1`` API.

This client is the Vibe-side transport for the Research Controller
(§7 / §18.7 WP4). It mirrors the mapping in ``agent/dsa_lab_mcp_server.py`` but
is a programmatic client instead of an MCP tool bridge:

- Only loopback HTTP URLs are accepted (§7.1).
- All IDs pass the ``^[A-Za-z0-9_-]{1,128}$`` charset check (§4.1).
- ``source_code`` is rejected above 64 KiB before any HTTP request (§6.1 /
  §7.1).
- HTTP ``429`` / ``502`` / ``503`` / ``504`` are marked retryable; ``400`` /
  ``403`` / ``409`` / ``422`` are not automatically retried (§7.1).
- Responses larger than the capabilities ``mcp_response_bytes_max`` are
  rejected locally.
- Transport failures raise :class:`DsaUnavailableError`; the controller uses
  exponential backoff instead of treating a timeout as a committed write
  (§4.3).

Every method returns a normalized envelope dict::

    {"status": "ok", "action": ..., "http_status": 200, "retryable": False,
     "data": {...}}
    {"status": "error", "action": ..., "http_status": 422, "retryable": False,
     "error": {...}}

``base_url`` is configurable so the same client can switch from the Mock DSA
server to the real DSA Backtest Lab. Mock success is not integration success.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from src.config.env_schema import EnvConfig

HTTP_PREFIX = "/internal/v1/research-loop"
RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_MAX_RESPONSE_BYTES = 10_000_000
_SOURCE_CODE_MAX_BYTES = 65_536

# §7.1: 429/502/503/504 标记可重试；400/403/409/422 默认不自动重试。
_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})


class DsaProtocolError(RuntimeError):
    """The DSA peer returned something that is not a valid research-loop envelope."""


class DsaUnavailableError(RuntimeError):
    """The DSA peer could not be reached (connection / timeout).

    This is distinct from a structured DSA error: the controller must treat a
    network timeout as *not committed* and query by idempotency key or resource
    id before re-issuing (§4.3).
    """


def validate_loopback_base_url(raw: str) -> str:
    """Validate that ``raw`` is a loopback-only HTTP URL and normalize it.

    Raises:
        ValueError: When the URL is not a loopback HTTP URL or contains
            unsupported components (credentials, query, fragment).
    """
    parsed = urlparse(raw.strip())
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("DSA research-loop URL must be a loopback HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("DSA research-loop URL contains unsupported URL components")
    return raw.strip().rstrip("/")


def validate_resource_id(value: Any, *, name: str) -> str:
    """Normalize an id to the allowed ``[A-Za-z0-9_-]`` charset (§4.1)."""
    normalized = str(value or "").strip()
    if not RESOURCE_ID_RE.fullmatch(normalized):
        raise ValueError(f"{name} must contain only letters, numbers, '_' or '-'")
    return normalized


def http_status_retryable(http_status: int) -> bool:
    """Return whether the HTTP status is retryable per §7.1."""
    return http_status in _RETRYABLE_HTTP_STATUS


def source_code_too_large(source_code: str) -> bool:
    """Return whether *source_code* exceeds the 64 KiB contract limit."""
    return len(source_code.encode("utf-8")) > _SOURCE_CODE_MAX_BYTES


def sha256_hex(value: bytes) -> str:
    """Return the lowercase hex SHA-256 of *value*."""
    return hashlib.sha256(value).hexdigest()


class DsaLoopClient:
    """Programmatic client for the DSA research-loop.v1 API.

    Args:
        base_url: Loopback HTTP base URL (e.g. ``http://127.0.0.1:8011``).
            Defaults to ``EnvConfig().dsa.research_loop_url`` (read from the
            ``DSA_RESEARCH_LOOP_URL`` env var, falling back to
            ``http://127.0.0.1:8011``).
        timeout: Request timeout in seconds.  Defaults to
            ``EnvConfig().dsa.research_loop_timeout_seconds``.
        client_factory: Injectable seam for tests: a callable returning a
            context manager exposing ``request(method, path, **kwargs)``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        dsa_cfg = EnvConfig().dsa
        raw = base_url if base_url is not None else dsa_cfg.research_loop_url
        self.base_url = validate_loopback_base_url(raw)
        self.timeout = float(
            timeout if timeout is not None else dsa_cfg.research_loop_timeout_seconds
        )
        self.client_factory = client_factory

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory(base_url=self.base_url, timeout=self.timeout)
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, trust_env=False)

    def request(
        self,
        action: str,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue one HTTP request and return the normalized envelope dict.

        Raises:
            DsaUnavailableError: On transport errors (DNS / connect / timeout).
            DsaProtocolError: On oversized or non-JSON responses.
        """
        with self._client() as client:
            try:
                response = client.request(method, path, json=body, headers=headers, params=params)
            except httpx.RequestError as exc:
                raise DsaUnavailableError(f"dsa_research_loop_unavailable: {exc}") from exc

        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise DsaProtocolError("dsa_response_too_large")
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise DsaProtocolError("dsa_response_not_json") from exc
        if not isinstance(payload, dict):
            raise DsaProtocolError("dsa_response_not_object")

        http_status = response.status_code
        if response.is_error:
            error = payload.get("error", payload)
            return {
                "status": "error",
                "action": action,
                "http_status": http_status,
                "retryable": http_status_retryable(http_status),
                "error": error,
                "data": None,
            }
        return {
            "status": "ok",
            "action": action,
            "http_status": http_status,
            "retryable": False,
            "data": payload.get("data", payload),
        }

    def _post(
        self,
        action: str,
        path: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if idempotency_key is not None:
            key = validate_resource_id(idempotency_key, name="idempotency_key")
            headers["Idempotency-Key"] = key
        return self.request(action, "POST", path, body=body, headers=headers)

    @staticmethod
    def _new_idempotency_key(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:24]}"

    def _idempotency_key(self, raw: str | None, *, prefix: str) -> str:
        """Return *raw* when it passes the ``[A-Za-z0-9_-]{1,128}`` check, else a uuid key.

        Every write must carry a deterministic, charset-safe ``Idempotency-Key``
        so the real DSA ``_missing_key`` guard is never tripped (§4.3).  When the
        payload lacks the identifying field (or the composed key would fail the
        resource-id check), fall back to a fresh uuid key.
        """
        if raw is not None:
            try:
                return validate_resource_id(raw, name="idempotency_key")
            except ValueError:
                pass
        return self._new_idempotency_key(prefix)

    # ------------------------------------------------------------------
    # Capabilities & experiment lifecycle (§6.1 - §6.4)
    # ------------------------------------------------------------------

    def get_capabilities(self) -> dict[str, Any]:
        return self.request("get_research_loop_capabilities", "GET", f"{HTTP_PREFIX}/capabilities")

    def register_experiment(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "register_research_experiment",
            f"{HTTP_PREFIX}/experiments",
            payload,
            idempotency_key=idempotency_key or self._new_idempotency_key("exp"),
        )

    def register_candidate(self, experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        exp_id = validate_resource_id(experiment_id, name="experiment_id")
        source = payload.get("source_code")
        if source is not None and source_code_too_large(str(source)):
            return {
                "status": "error",
                "action": "register_strategy_candidate",
                "http_status": 422,
                "retryable": False,
                "error": {
                    "code": "source_code_too_large",
                    "category": "CONTRACT_ERROR",
                    "message": "source_code exceeds 64 KiB",
                    "retryable": False,
                    "details": {},
                    "artifact_refs": [],
                },
                "data": None,
            }
        candidate_id = payload.get("candidate_id")
        candidate_version = payload.get("candidate_version")
        raw_key = (
            f"cand_{candidate_id}_v{candidate_version}"
            if candidate_id is not None and candidate_version is not None
            else None
        )
        return self._post(
            "register_strategy_candidate",
            f"{HTTP_PREFIX}/experiments/{exp_id}/candidates",
            payload,
            idempotency_key=self._idempotency_key(raw_key, prefix="cand"),
        )

    def start_execution(
        self,
        experiment_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        exp_id = validate_resource_id(experiment_id, name="experiment_id")
        return self._post(
            "start_research_execution",
            f"{HTTP_PREFIX}/experiments/{exp_id}/executions",
            payload,
            idempotency_key=idempotency_key or self._new_idempotency_key("exec"),
        )

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        exec_id = validate_resource_id(execution_id, name="execution_id")
        return self.request("get_research_execution", "GET", f"{HTTP_PREFIX}/executions/{exec_id}")

    def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        exec_id = validate_resource_id(execution_id, name="execution_id")
        return self._post(
            "cancel_research_execution",
            f"{HTTP_PREFIX}/executions/{exec_id}/cancel",
            {},
            idempotency_key=self._idempotency_key(f"cancel_{exec_id}", prefix="cancel"),
        )

    # ------------------------------------------------------------------
    # Events / errors / evidence / reviews / decisions (§6.6 - §6.10)
    # ------------------------------------------------------------------

    def poll_events(
        self,
        experiment_id: str,
        *,
        after_sequence: int = 0,
        page_size: int = 100,
    ) -> dict[str, Any]:
        exp_id = validate_resource_id(experiment_id, name="experiment_id")
        params = {"after_sequence": int(after_sequence), "page_size": max(1, min(int(page_size), 200))}
        return self.request("poll_research_events", "GET", f"{HTTP_PREFIX}/experiments/{exp_id}/events", params=params)

    def get_error(self, error_id: str) -> dict[str, Any]:
        err_id = validate_resource_id(error_id, name="error_id")
        return self.request("get_research_error", "GET", f"{HTTP_PREFIX}/errors/{err_id}")

    def get_evidence(self, evidence_id: str) -> dict[str, Any]:
        ev_id = validate_resource_id(evidence_id, name="evidence_id")
        return self.request("get_research_evidence", "GET", f"{HTTP_PREFIX}/evidence/{ev_id}")

    def get_review(self, review_id: str) -> dict[str, Any]:
        rev_id = validate_resource_id(review_id, name="review_id")
        return self.request("get_research_review", "GET", f"{HTTP_PREFIX}/reviews/{rev_id}")

    def record_decision(
        self,
        experiment_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        exp_id = validate_resource_id(experiment_id, name="experiment_id")
        # decision_id doubles as the idempotency key: a decision is immutable
        # and re-recording the same decision is a no-op (§4.3 / §6.10).
        key = idempotency_key or payload.get("decision_id") or self._new_idempotency_key("decision")
        return self._post(
            "record_research_decision",
            f"{HTTP_PREFIX}/experiments/{exp_id}/decisions",
            payload,
            idempotency_key=key,
        )

    # ------------------------------------------------------------------
    # Data / market panel / artifact / factor snapshots (§6.12 - §6.15)
    # ------------------------------------------------------------------

    def build_data_snapshot(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "build_data_snapshot",
            f"{HTTP_PREFIX}/data-snapshots/build",
            payload,
            idempotency_key=idempotency_key or self._new_idempotency_key("snap"),
        )

    def get_data_snapshot(self, data_snapshot_id: str) -> dict[str, Any]:
        snap_id = validate_resource_id(data_snapshot_id, name="data_snapshot_id")
        return self.request("get_data_snapshot", "GET", f"{HTTP_PREFIX}/data-snapshots/{snap_id}")

    def build_universe_snapshot(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Build a ``universe_snapshot.v1`` via DSA (§6.12-6.13 / §11.7).

        The payload must include a ``snapshot_id`` with the ``universe_`` prefix,
        a ``start_date`` / ``end_date`` window and an optional
        ``selection_policy_version``; DSA reads the stock-pool source from its own
        configuration (Vibe must not pass host paths, §6.12).
        """
        return self._post(
            "build_universe_snapshot",
            f"{HTTP_PREFIX}/universe-snapshots/build",
            payload,
            idempotency_key=idempotency_key or self._new_idempotency_key("uni"),
        )

    def get_universe_snapshot(self, universe_snapshot_id: str) -> dict[str, Any]:
        uni_id = validate_resource_id(universe_snapshot_id, name="universe_snapshot_id")
        return self.request("get_universe_snapshot", "GET", f"{HTTP_PREFIX}/universe-snapshots/{uni_id}")

    def export_market_panel(self, data_snapshot_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        snap_id = validate_resource_id(data_snapshot_id, name="data_snapshot_id")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return self._post(
            "export_market_panel",
            f"{HTTP_PREFIX}/data-snapshots/{snap_id}/panel-exports",
            payload,
            idempotency_key=self._idempotency_key(f"panel_{snap_id}_{digest}", prefix="panel"),
        )

    def create_artifact_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            "create_artifact_upload",
            f"{HTTP_PREFIX}/artifact-uploads",
            payload,
            idempotency_key=f"upload_{uuid.uuid4().hex[:16]}",
        )

    def complete_artifact_upload(self, upload_id: str) -> dict[str, Any]:
        up_id = validate_resource_id(upload_id, name="upload_id")
        return self._post(
            "complete_artifact_upload",
            f"{HTTP_PREFIX}/artifact-uploads/{up_id}/complete",
            {},
            idempotency_key=self._idempotency_key(f"up_{up_id}", prefix="up"),
        )

    def register_factor_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = payload.get("snapshot_id") or uuid.uuid4().hex[:16]
        return self._post(
            "register_factor_snapshot",
            f"{HTTP_PREFIX}/factor-snapshots",
            payload,
            idempotency_key=self._idempotency_key(f"frag_{snapshot_id}", prefix="frag"),
        )

    def get_factor_snapshot(self, factor_snapshot_id: str) -> dict[str, Any]:
        snap_id = validate_resource_id(factor_snapshot_id, name="factor_snapshot_id")
        return self.request("get_factor_snapshot", "GET", f"{HTTP_PREFIX}/factor-snapshots/{snap_id}")


def serialize_json(value: Any) -> str:
    """Serialize *value* to JSON without NaN / Infinity (finite-only §4.1)."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
