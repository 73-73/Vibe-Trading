"""Tests for the M4 research-loop.v1 tools added to the DSA MCP bridge (§7.1).

Unit tests cover the new ``_research_loop_spec`` mapping, idempotency headers,
retryable classification and the 64 KiB source_code guard. Integration tests
spawn the real ``dsa_lab_mcp_server.py`` over stdio and drive the new tools
against the real Mock DSA server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import dsa_lab_mcp_server as dsa
from tests.research_loop_test_helpers import MockDsaServer

_AGENT_DIR = Path(__file__).resolve().parent.parent
_DSA_SERVER = _AGENT_DIR / "dsa_lab_mcp_server.py"
_PYTHON = sys.executable


class _FakeClient:
    def __init__(self, *, base_url: str, timeout: float, response: httpx.Response | None = None,
                 error: Exception | None = None) -> None:
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
        assert self.response is not None
        return self.response


def _ok_response(status: int = 200, data: Any = None) -> httpx.Response:
    return httpx.Response(status, json={"protocol_version": "research-loop.v1", "status": "ok",
                                        "action": "x", "data": data})


def _err_response(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"protocol_version": "research-loop.v1", "status": "error",
                                        "action": "x",
                                        "error": {"code": code, "category": "CONTRACT_ERROR", "message": "err",
                                                  "retryable": False, "details": {}, "artifact_refs": []}})


def _rl_spec(**kwargs: Any) -> tuple[str, str, Any, dict[str, str], Any]:
    defaults: dict[str, Any] = {
        "action": "get_research_loop_capabilities",
        "payload": None,
        "experiment_id": None,
        "resource_id": None,
        "after_sequence": 0,
        "page_size": 100,
        "idempotency_key": None,
    }
    defaults.update(kwargs)
    return dsa._research_loop_spec(**defaults)


# ---------------------------------------------------------------------------
# Spec mapping (§7.1)
# ---------------------------------------------------------------------------


def test_rl_spec_maps_capabilities() -> None:
    assert _rl_spec(action="get_research_loop_capabilities") == (
        "GET",
        "/internal/v1/research-loop/capabilities",
        None,
        {},
        None,
    )


def test_rl_spec_maps_register_experiment_with_idempotency() -> None:
    method, path, body, headers, params = _rl_spec(
        action="register_research_experiment", payload={"experiment_id": "e1"}, idempotency_key="key-1"
    )
    assert method == "POST"
    assert path == "/internal/v1/research-loop/experiments"
    assert body == {"experiment_id": "e1"}
    assert headers["Idempotency-Key"] == "key-1"
    assert params is None


def test_rl_spec_register_candidate_rejects_oversized_source() -> None:
    with pytest.raises(ValueError):
        _rl_spec(action="register_strategy_candidate", payload={"source_code": "x" * 70000},
                 experiment_id="exp_1")


def test_rl_spec_maps_start_execution() -> None:
    method, path, body, headers, params = _rl_spec(
        action="start_research_execution",
        payload={"execution_type": "strategy_sandbox_smoke"},
        experiment_id="exp_1",
        idempotency_key="k",
    )
    assert path == "/internal/v1/research-loop/experiments/exp_1/executions"
    assert headers["Idempotency-Key"] == "k"


def test_rl_spec_maps_poll_events_params() -> None:
    method, path, body, headers, params = _rl_spec(
        action="poll_research_events", experiment_id="exp_1", after_sequence=3, page_size=500
    )
    assert path == "/internal/v1/research-loop/experiments/exp_1/events"
    assert params == {"after_sequence": 3, "page_size": 200}


def test_rl_spec_maps_get_execution() -> None:
    assert _rl_spec(action="get_research_execution", resource_id="exec_1") == (
        "GET",
        "/internal/v1/research-loop/executions/exec_1",
        None,
        {},
        None,
    )


def test_rl_spec_maps_data_tools() -> None:
    method, path, body, headers, params = _rl_spec(action="build_data_snapshot", payload={"layer": "development"})
    assert path == "/internal/v1/research-loop/data-snapshots/build"
    assert headers["Idempotency-Key"]

    assert _rl_spec(action="get_data_snapshot", resource_id="data_dev_001")[1] == (
        "/internal/v1/research-loop/data-snapshots/data_dev_001"
    )

    method, path, body, headers, params = _rl_spec(
        action="export_market_panel", payload={"fields": ["close"]}, resource_id="data_dev_001"
    )
    assert path == "/internal/v1/research-loop/data-snapshots/data_dev_001/panel-exports"

    method, path, body, headers, params = _rl_spec(action="create_artifact_upload", payload={"kind": "x"})
    assert path == "/internal/v1/research-loop/artifact-uploads"

    assert _rl_spec(action="complete_artifact_upload", resource_id="upload_1")[1] == (
        "/internal/v1/research-loop/artifact-uploads/upload_1/complete"
    )

    assert _rl_spec(action="register_factor_snapshot", payload={"manifest": {}})[1] == (
        "/internal/v1/research-loop/factor-snapshots"
    )
    assert _rl_spec(action="get_factor_snapshot", resource_id="fs_1")[1] == (
        "/internal/v1/research-loop/factor-snapshots/fs_1"
    )


def test_rl_spec_rejects_unknown_action() -> None:
    with pytest.raises(ValueError):
        _rl_spec(action="not_real")


def test_rl_spec_rejects_invalid_id() -> None:
    with pytest.raises(ValueError):
        _rl_spec(action="get_research_execution", resource_id="bad/id")


# ---------------------------------------------------------------------------
# Request envelope + retryable classification (§7.1)
# ---------------------------------------------------------------------------


def _request(action: str, fake: _FakeClient, **kwargs: Any) -> str:
    return dsa._research_loop_request(action, client_factory=lambda **kw: fake, **kwargs)


def test_rl_request_forwards_payload_and_idempotency_header() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok_response(data={}))
    result = _request("register_research_experiment", fake, payload={"experiment_id": "e1"}, idempotency_key="k1")
    assert fake.calls[0]["kwargs"]["headers"]["Idempotency-Key"] == "k1"
    assert json.loads(result)["status"] == "ok"


def test_rl_request_marks_429_retryable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_err_response(429, "rate_limited"))
    payload = json.loads(_request("get_research_loop_capabilities", fake))
    assert payload["status"] == "error"
    assert payload["retryable"] is True


def test_rl_request_marks_422_not_retryable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_err_response(422, "source_hash_mismatch"))
    payload = json.loads(
        _request(
            "register_strategy_candidate",
            fake,
            payload={"candidate_id": "c1", "source_code": "def f(): pass", "manifest": {"factor_ids": ["a"]}},
            experiment_id="e1",
        )
    )
    assert payload["retryable"] is False
    assert payload["http_status"] == 422
    assert payload["detail"]["code"] == "source_hash_mismatch"


def test_rl_request_transport_error_retryable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, error=httpx.ConnectError("refused"))
    payload = json.loads(_request("get_research_loop_capabilities", fake))
    assert payload["status"] == "error"
    assert payload["retryable"] is True
    assert payload["http_status"] == 503


def test_rl_request_source_code_too_large_rejected_before_http() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok_response(data={}))
    payload = json.loads(
        _request(
            "register_strategy_candidate",
            fake,
            payload={"candidate_id": "c1", "source_code": "x" * 70000},
            experiment_id="e1",
        )
    )
    assert payload["status"] == "error"
    assert "64 KiB" in payload["error"]
    assert fake.calls == []


def test_rl_request_rejects_non_loopback_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSA_RESEARCH_LOOP_URL", "https://example.com")
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok_response(data={}))
    payload = json.loads(_request("get_research_loop_capabilities", fake))
    assert "must be a loopback HTTP URL" in payload["error"]


# ---------------------------------------------------------------------------
# Integration: real server over stdio + real Mock DSA (HTTP)
# ---------------------------------------------------------------------------


def _make_agent_json(tmp_path: Path, base_url: str) -> Path:
    config: dict[str, Any] = {
        "mcpServers": {
            "dsa_lab": {
                "command": _PYTHON,
                "args": [str(_DSA_SERVER)],
                "env": {
                    "DSA_RESEARCH_LOOP_URL": base_url,
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


def _registry(tmp_path: Path, base_url: str):
    from src.config.loader import load_agent_config
    from src.tools import build_registry

    cfg_path = _make_agent_json(tmp_path, base_url)
    return build_registry(agent_config=load_agent_config(config_path=cfg_path))


def _inner(result: str) -> dict[str, Any]:
    """Extract the DSA envelope from a registry tool result (double-encoded)."""
    outer = json.loads(result)
    for key in ("text", "data"):
        value = outer.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return outer


@pytest.fixture
def mock() -> MockDsaServer:
    server = MockDsaServer("happy_path")
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.integration
def test_new_rl_tools_registered_and_capabilities_work(tmp_path: Path, mock: MockDsaServer) -> None:
    registry = _registry(tmp_path, mock.base_url)
    assert "mcp_dsa_lab_get_research_loop_capabilities" in registry.tool_names
    assert "mcp_dsa_lab_register_research_experiment" in registry.tool_names
    assert "mcp_dsa_lab_poll_research_events" in registry.tool_names
    result = registry.get("mcp_dsa_lab_get_research_loop_capabilities").execute()
    inner = _inner(result)
    assert inner["status"] == "ok"
    advertised = inner["data"]["execution_types"]
    assert "strategy_sandbox_smoke" not in advertised
    assert "portfolio_backtest" not in advertised
    assert "robustness" not in advertised
    assert "gate_challenger" in advertised


@pytest.mark.integration
def test_register_experiment_roundtrip_through_mcp(tmp_path: Path, mock: MockDsaServer) -> None:
    registry = _registry(tmp_path, mock.base_url)
    payload = {
        "experiment_id": "exp_mcp_001",
        "round_id": "round_1",
        "objective": "o",
        "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": ["f"], "factor_ids": ["a"]},
        "research_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
        "gate_policy_version": "gate_champion_v1",
    }
    result = registry.get("mcp_dsa_lab_register_research_experiment").execute(
        payload=payload, idempotency_key="exp_mcp_001"
    )
    inner = _inner(result)
    assert inner["status"] == "ok"
    assert inner["data"]["data"]["experiment_id"] == "exp_mcp_001"

    events = registry.get("mcp_dsa_lab_poll_research_events").execute(
        experiment_id="exp_mcp_001", after_sequence=0, page_size=50
    )
    inner = _inner(events)
    assert inner["status"] == "ok"
    assert any(e["event_type"] == "experiment.created" for e in inner["data"]["data"]["items"])
