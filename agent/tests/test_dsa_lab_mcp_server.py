"""Tests for the standalone DSA Backtest Lab MCP server.

Two layers:
- Unit tests (no network, no subprocess): pure helpers plus ``_lab_request``
  with an injected fake HTTP client.
- Integration tests (``@pytest.mark.integration``): spawn the real
  ``dsa_lab_mcp_server.py`` over stdio via ``load_agent_config`` +
  ``build_registry``, backed by an in-test fake DSA HTTP service so no real
  DSA Backtest Lab is required.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

import dsa_lab_mcp_server as dsa

# ---------------------------------------------------------------------------
# Fake HTTP client for unit-testing _lab_request without a real service
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        response: httpx.Response | None = None,
        error: Exception | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"method": method, "path": path, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None, "no response configured"
        return self.response


def _ok_response(*, status: int = 200, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status, json=body)


# ---------------------------------------------------------------------------
# Unit tests — URL / resource-id validation
# ---------------------------------------------------------------------------


def test_validated_base_url_accepts_loopback_http() -> None:
    assert dsa._validated_base_url("http://127.0.0.1:8011") == "http://127.0.0.1:8011"
    assert dsa._validated_base_url("http://localhost:8011/") == "http://localhost:8011"
    assert dsa._validated_base_url("http://[::1]:8011") == "http://[::1]:8011"
    assert dsa._validated_base_url("  http://127.0.0.1:8011  ") == "http://127.0.0.1:8011"


def test_validated_base_url_rejects_non_loopback() -> None:
    for bad in (
        "https://127.0.0.1:8011",
        "http://example.com",
        "http://user:pass@127.0.0.1:8011",
        "http://127.0.0.1:8011?x=1",
        "http://127.0.0.1:8011#frag",
    ):
        with pytest.raises(ValueError):
            dsa._validated_base_url(bad)


def test_resource_id_accepts_allowed_chars() -> None:
    assert dsa._resource_id("ma_cross-2026", name="batch_id") == "ma_cross-2026"
    assert dsa._resource_id("ABC_123", name="batch_id") == "ABC_123"


def test_resource_id_rejects_invalid_chars() -> None:
    for bad in ("bad/id", "", "has space", "x" * 129, "a.b"):
        with pytest.raises(ValueError):
            dsa._resource_id(bad, name="batch_id")


# ---------------------------------------------------------------------------
# Unit tests — _request_spec action mapping
# ---------------------------------------------------------------------------


def _spec(**kwargs: Any) -> tuple[str, str, Any, dict[str, str], Any]:
    defaults: dict[str, Any] = {
        "action": "health",
        "payload": None,
        "batch_id": None,
        "run_id": None,
        "page": 1,
        "page_size": 50,
        "idempotency_key": None,
    }
    defaults.update(kwargs)
    return dsa._request_spec(**defaults)


def test_request_spec_maps_health() -> None:
    assert _spec(action="health") == ("GET", "/health", None, {}, None)


def test_request_spec_maps_refresh_catalog() -> None:
    assert _spec(action="refresh_catalog") == ("POST", "/internal/v1/csv-catalog/refresh", None, {}, None)


def test_request_spec_maps_list_strategies() -> None:
    assert _spec(action="list_strategies") == ("GET", "/internal/v1/strategies", None, {}, None)


def test_request_spec_maps_preflight() -> None:
    payload = {"strategy_id": "ma_cross"}
    assert _spec(action="preflight", payload=payload) == (
        "POST",
        "/internal/v1/preflight",
        payload,
        {},
        None,
    )


def test_request_spec_preflight_requires_dict_payload() -> None:
    with pytest.raises(ValueError):
        _spec(action="preflight", payload="nope")


def test_request_spec_maps_create_batch_with_idempotency_key() -> None:
    payload = {"strategy_id": "ma_cross"}
    method, path, body, headers, params = _spec(
        action="create_batch", payload=payload, idempotency_key="research-001"
    )
    assert method == "POST"
    assert path == "/internal/v1/batches"
    assert body == payload
    assert headers == {"Idempotency-Key": "research-001"}
    assert params is None


def test_request_spec_create_batch_generates_key_when_absent() -> None:
    _method, _path, _body, headers, _params = _spec(
        action="create_batch", payload={"strategy_id": "ma_cross"}
    )
    key = headers["Idempotency-Key"]
    assert dsa._RESOURCE_ID_RE.fullmatch(key), key


def test_request_spec_maps_list_batches_caps_page_size() -> None:
    method, path, body, headers, params = _spec(action="list_batches", page=2, page_size=300)
    assert method == "GET"
    assert path == "/internal/v1/batches"
    assert params == {"page": 2, "page_size": 100}


def test_request_spec_maps_get_batch() -> None:
    assert _spec(action="get_batch", batch_id="b1") == ("GET", "/internal/v1/batches/b1", None, {}, None)


def test_request_spec_maps_list_runs() -> None:
    method, path, body, headers, params = _spec(action="list_runs", batch_id="b1", page=1, page_size=50)
    assert method == "GET"
    assert path == "/internal/v1/batches/b1/runs"
    assert params == {"page": 1, "page_size": 50}


def test_request_spec_maps_cancel_batch() -> None:
    assert _spec(action="cancel_batch", batch_id="b1") == (
        "POST",
        "/internal/v1/batches/b1/cancel",
        None,
        {},
        None,
    )


def test_request_spec_maps_get_run() -> None:
    assert _spec(action="get_run", run_id="r1") == ("GET", "/internal/v1/runs/r1", None, {}, None)


def test_request_spec_rejects_unknown_action() -> None:
    with pytest.raises(ValueError):
        _spec(action="not_a_real_action")


# ---------------------------------------------------------------------------
# Unit tests — _coerce_timeout
# ---------------------------------------------------------------------------


def test_coerce_timeout_accepts_positive_float() -> None:
    assert dsa._coerce_timeout("45.5") == 45.5
    assert dsa._coerce_timeout(300) == 300.0


def test_coerce_timeout_rejects_invalid() -> None:
    for bad in ("abc", "0", "-1", None):
        with pytest.raises(ValueError):
            dsa._coerce_timeout(bad)


# ---------------------------------------------------------------------------
# Unit tests — _lab_request with an injected client
# ---------------------------------------------------------------------------


def test_lab_request_issues_request_via_injected_client() -> None:
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=_ok_response(body={"items": [{"id": "ma_cross"}]}),
    )
    result = dsa._lab_request("list_strategies", client_factory=lambda **kw: fake)

    assert fake.calls[0]["method"] == "GET"
    assert fake.calls[0]["path"] == "/internal/v1/strategies"
    assert fake.base_url == "http://127.0.0.1:8011"
    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["action"] == "list_strategies"
    assert payload["data"]["items"][0]["id"] == "ma_cross"


def test_lab_request_forwards_payload_and_idempotency_header() -> None:
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=_ok_response(status=201, body={"batch_id": "batch-001"}),
    )
    result = dsa._lab_request(
        "create_batch",
        payload={"strategy_id": "ma_cross"},
        idempotency_key="research-001",
        client_factory=lambda **kw: fake,
    )

    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/internal/v1/batches"
    assert call["kwargs"]["json"] == {"strategy_id": "ma_cross"}
    assert call["kwargs"]["headers"] == {"Idempotency-Key": "research-001"}
    assert json.loads(result)["data"]["batch_id"] == "batch-001"


def test_lab_request_invalid_pagination() -> None:
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=_ok_response(body={}),
    )
    result = dsa._lab_request("list_batches", page=0, client_factory=lambda **kw: fake)
    assert json.loads(result)["error"] == "invalid_pagination"
    assert fake.calls == []


def test_lab_request_surfaces_error_detail() -> None:
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=httpx.Response(
            422,
            json={"detail": {"code": "preflight_failed", "message": "bad window"}},
        ),
    )
    result = dsa._lab_request("preflight", payload={"strategy_id": "x"}, client_factory=lambda **kw: fake)
    payload = json.loads(result)
    assert payload["status"] == "error"
    assert payload["http_status"] == 422
    assert payload["detail"]["code"] == "preflight_failed"


def test_lab_request_response_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dsa, "_MAX_RESPONSE_BYTES", 10)
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=httpx.Response(200, content=b"x" * 100),
    )
    result = dsa._lab_request("health", client_factory=lambda **kw: fake)
    assert json.loads(result)["error"] == "dsa_response_too_large"


def test_lab_request_rejects_non_loopback_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSA_LAB_URL", "https://example.com")
    fake = _FakeClient(base_url="http://127.0.0.1:8011", timeout=300.0, response=_ok_response(body={}))
    result = dsa._lab_request("health", client_factory=lambda **kw: fake)
    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "must be a loopback HTTP URL" in payload["error"]


def test_lab_request_respects_explicit_base_url_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSA_LAB_URL", "https://example.com")
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        response=_ok_response(body={"status": "ok"}),
    )
    result = dsa._lab_request("health", base_url="http://127.0.0.1:8011", client_factory=lambda **kw: fake)
    assert json.loads(result)["status"] == "ok"


def test_lab_request_request_error_yields_unavailable() -> None:
    fake = _FakeClient(
        base_url="http://127.0.0.1:8011",
        timeout=300.0,
        error=httpx.ConnectError("refused"),
    )
    result = dsa._lab_request("health", client_factory=lambda **kw: fake)
    assert json.loads(result)["error"] == "dsa_backtest_lab_unavailable"


# ---------------------------------------------------------------------------
# Integration tests — spawn the real server over stdio via build_registry
# ---------------------------------------------------------------------------

_AGENT_DIR = Path(__file__).resolve().parent.parent
_DSA_SERVER = _AGENT_DIR / "dsa_lab_mcp_server.py"
_PYTHON = sys.executable


class _FakeDsaLabHandler(BaseHTTPRequestHandler):
    """In-test fake DSA Backtest Lab HTTP service."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        elif self.path == "/internal/v1/strategies":
            self._reply(200, {"items": [{"id": "ma_cross"}]})
        else:
            self._reply(404, {"detail": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        if self.path == "/internal/v1/batches":
            self._reply(201, {"batch_id": "batch-001"})
        elif self.path == "/internal/v1/preflight":
            self._reply(200, {"preflight": "ok"})
        else:
            self._reply(404, {"detail": "not found"})

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:  # silence request logging
        pass


@pytest.fixture(scope="module")
def dsa_lab_fake_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeDsaLabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _make_agent_json(tmp_path: Path, base_url: str) -> Path:
    config: dict[str, Any] = {
        "mcpServers": {
            "dsa_lab": {
                "command": _PYTHON,
                "args": [str(_DSA_SERVER)],
                "env": {
                    "DSA_LAB_URL": base_url,
                    "DSA_LAB_TIMEOUT_SECONDS": "300",
                },
                "enabledTools": ["*"],
                "toolTimeout": 300,
            }
        }
    }
    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(json.dumps(config))
    return cfg_path


def _build_registry(tmp_path: Path, base_url: str):
    from src.config.loader import load_agent_config
    from src.tools import build_registry

    cfg_path = _make_agent_json(tmp_path, base_url)
    agent_config = load_agent_config(config_path=cfg_path)
    return build_registry(agent_config=agent_config)


def _inner_dsa_payload(result: str) -> dict[str, Any]:
    """Extract the DSA envelope nested inside the MCP wrapper payload."""
    outer = json.loads(result)
    text = outer.get("text")
    if isinstance(text, str):
        try:
            inner = json.loads(text)
        except json.JSONDecodeError:
            inner = None
        if isinstance(inner, dict):
            return inner
    data = outer.get("data")
    if isinstance(data, str):
        try:
            inner = json.loads(data)
        except json.JSONDecodeError:
            inner = None
        if isinstance(inner, dict):
            return inner
    if isinstance(outer, dict) and "action" in outer and "status" in outer:
        return outer
    raise AssertionError(f"could not extract DSA payload from: {result[:300]}")


@pytest.mark.integration
def test_dsa_lab_tools_registered(tmp_path: Path, dsa_lab_fake_server: str) -> None:
    registry = _build_registry(tmp_path, dsa_lab_fake_server)
    names = {n for n in registry.tool_names if n.startswith("mcp_dsa_lab_")}
    assert names == {
        # 一期工具（不删除/改名）
        "mcp_dsa_lab_health",
        "mcp_dsa_lab_refresh_catalog",
        "mcp_dsa_lab_list_strategies",
        "mcp_dsa_lab_preflight",
        "mcp_dsa_lab_create_batch",
        "mcp_dsa_lab_list_batches",
        "mcp_dsa_lab_get_batch",
        "mcp_dsa_lab_list_runs",
        "mcp_dsa_lab_get_run",
        "mcp_dsa_lab_cancel_batch",
        "mcp_dsa_lab_run_factor_research",
        # M4 research-loop.v1 工具（§7.1 追加，不改一期工具）
        "mcp_dsa_lab_get_research_loop_capabilities",
        "mcp_dsa_lab_register_research_experiment",
        "mcp_dsa_lab_register_strategy_candidate",
        "mcp_dsa_lab_start_research_execution",
        "mcp_dsa_lab_get_research_execution",
        "mcp_dsa_lab_poll_research_events",
        "mcp_dsa_lab_get_research_error",
        "mcp_dsa_lab_get_research_evidence",
        "mcp_dsa_lab_get_research_review",
        "mcp_dsa_lab_record_research_decision",
        "mcp_dsa_lab_cancel_research_execution",
        "mcp_dsa_lab_build_data_snapshot",
        "mcp_dsa_lab_get_data_snapshot",
        "mcp_dsa_lab_export_market_panel",
        "mcp_dsa_lab_create_artifact_upload",
        "mcp_dsa_lab_complete_artifact_upload",
        "mcp_dsa_lab_register_factor_snapshot",
        "mcp_dsa_lab_get_factor_snapshot",
    }
    for name in names:
        assert registry.get(name).is_readonly is False


@pytest.mark.integration
def test_dsa_lab_health_roundtrip(tmp_path: Path, dsa_lab_fake_server: str) -> None:
    registry = _build_registry(tmp_path, dsa_lab_fake_server)
    result = registry.get("mcp_dsa_lab_health").execute()
    inner = _inner_dsa_payload(result)
    assert inner["status"] == "ok"
    assert inner["action"] == "health"


@pytest.mark.integration
def test_dsa_lab_list_strategies_roundtrip(tmp_path: Path, dsa_lab_fake_server: str) -> None:
    registry = _build_registry(tmp_path, dsa_lab_fake_server)
    result = registry.get("mcp_dsa_lab_list_strategies").execute()
    inner = _inner_dsa_payload(result)
    assert inner["data"]["items"][0]["id"] == "ma_cross"


@pytest.mark.integration
def test_dsa_lab_create_batch_forwards_payload(tmp_path: Path, dsa_lab_fake_server: str) -> None:
    registry = _build_registry(tmp_path, dsa_lab_fake_server)
    result = registry.get("mcp_dsa_lab_create_batch").execute(
        payload={"strategy_id": "ma_cross"}, idempotency_key="test-001"
    )
    inner = _inner_dsa_payload(result)
    assert inner["status"] == "ok"
    assert inner["data"]["batch_id"] == "batch-001"
