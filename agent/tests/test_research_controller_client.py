"""Tests for the DSA research-loop.v1 loopback client.

Unit tests use an injected fake HTTP client; a couple of tests exercise the
real Mock DSA server via uvicorn (``mock_dsa`` fixture) to prove wire
compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.research_controller.client.dsa_client import (
    DsaLoopClient,
    DsaProtocolError,
    DsaUnavailableError,
    http_status_retryable,
    source_code_too_large,
    validate_loopback_base_url,
    validate_resource_id,
)

# ---------------------------------------------------------------------------
# URL / id / limit validation (§7.1 / §4.1)
# ---------------------------------------------------------------------------


def test_loopback_base_url_accepts_loopback_http() -> None:
    assert validate_loopback_base_url("http://127.0.0.1:8012") == "http://127.0.0.1:8012"
    assert validate_loopback_base_url("http://localhost:8012/") == "http://localhost:8012"
    assert validate_loopback_base_url("http://[::1]:8012") == "http://[::1]:8012"


def test_loopback_base_url_rejects_non_loopback() -> None:
    for bad in ("https://127.0.0.1:8012", "http://example.com", "http://user:pass@127.0.0.1:8012"):
        with pytest.raises(ValueError):
            validate_loopback_base_url(bad)


def test_resource_id_charset_and_length() -> None:
    assert validate_resource_id("exp_momentum_001", name="experiment_id") == "exp_momentum_001"
    assert validate_resource_id("exec-001", name="execution_id") == "exec-001"
    for bad in ("", "has space", "a.b", "x" * 129, "bad/id"):
        with pytest.raises(ValueError):
            validate_resource_id(bad, name="id")


def test_source_code_size_limit() -> None:
    assert source_code_too_large("x" * 65536) is False
    assert source_code_too_large("x" * 65537) is True


def test_retryable_status_classification() -> None:
    # §7.1: 429/502/503/504 可重试
    for status in (429, 502, 503, 504):
        assert http_status_retryable(status) is True
    # 400/403/409/422 不自动重试
    for status in (400, 403, 404, 409, 422):
        assert http_status_retryable(status) is False


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------


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


def _ok(status: int = 200, data: Any = None) -> httpx.Response:
    return httpx.Response(status, json={"status": "ok", "action": "x", "data": data})


def _err(status: int, code: str, category: str = "CONTRACT_ERROR", retryable: bool = False) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "protocol_version": "research-loop.v1",
            "status": "error",
            "action": "x",
            "error": {"code": code, "category": category, "message": "err", "retryable": retryable,
                      "details": {}, "artifact_refs": []},
        },
    )


def _client(fake: _FakeClient) -> DsaLoopClient:
    return DsaLoopClient(base_url="http://127.0.0.1:8012", client_factory=lambda **kw: fake)


# ---------------------------------------------------------------------------
# Request mapping / envelope handling
# ---------------------------------------------------------------------------


def test_get_capabilities_maps_path() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok(data={"limits": {}}))
    result = _client(fake).get_capabilities()
    assert fake.calls[0]["path"] == "/internal/v1/research-loop/capabilities"
    assert fake.calls[0]["method"] == "GET"
    assert result["status"] == "ok"


def test_register_experiment_sends_idempotency_key() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok(status=201, data={"experiment_id": "e1"}))
    payload = {"experiment_id": "e1"}
    result = _client(fake).register_experiment(payload, idempotency_key="exp_e1")
    call = fake.calls[0]
    assert call["path"] == "/internal/v1/research-loop/experiments"
    assert call["kwargs"]["headers"]["Idempotency-Key"] == "exp_e1"
    assert result["data"]["experiment_id"] == "e1"


def test_register_experiment_generates_key_when_absent() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok(data={}))
    _client(fake).register_experiment({"experiment_id": "e1"})
    key = fake.calls[0]["kwargs"]["headers"]["Idempotency-Key"]
    assert len(key) >= 5


def test_register_candidate_rejects_oversized_source_before_http() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok(data={}))
    payload = {"candidate_id": "c1", "candidate_version": 1, "source_code": "x" * 70000}
    result = _client(fake).register_candidate("exp_1", payload)
    assert result["status"] == "error"
    assert result["error"]["code"] == "source_code_too_large"
    assert fake.calls == []  # 未发起 HTTP


def test_start_execution_maps_candidate_execution() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=_ok(status=202, data={"execution_id": "exec_1", "status": "queued"}))
    result = _client(fake).start_execution("exp_1", {"execution_type": "strategy_sandbox_smoke"},
                                           idempotency_key="key_1")
    call = fake.calls[0]
    assert call["path"] == "/internal/v1/research-loop/experiments/exp_1/executions"
    assert call["kwargs"]["json"]["execution_type"] == "strategy_sandbox_smoke"
    assert call["kwargs"]["headers"]["Idempotency-Key"] == "key_1"
    assert result["data"]["execution_id"] == "exec_1"


def test_poll_events_builds_query_params() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=_ok(data={"items": [], "next_sequence": 5, "has_more": False}))
    result = _client(fake).poll_events("exp_1", after_sequence=3, page_size=50)
    call = fake.calls[0]
    assert call["path"] == "/internal/v1/research-loop/experiments/exp_1/events"
    assert call["kwargs"]["params"] == {"after_sequence": 3, "page_size": 50}
    assert result["data"]["next_sequence"] == 5


def test_poll_events_caps_page_size() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0, response=_ok(data={}))
    _client(fake).poll_events("exp_1", after_sequence=0, page_size=9999)
    assert fake.calls[0]["kwargs"]["params"]["page_size"] == 200


def test_error_envelope_passthrough_marks_retryable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=_err(429, "rate_limited", retryable=True))
    result = _client(fake).get_capabilities()
    assert result["status"] == "error"
    assert result["http_status"] == 429
    assert result["retryable"] is True
    assert result["error"]["code"] == "rate_limited"


def test_error_envelope_422_not_retryable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=_err(422, "source_hash_mismatch"))
    result = _client(fake).register_candidate("exp_1", {"candidate_id": "c1"})
    assert result["status"] == "error"
    assert result["retryable"] is False
    assert result["error"]["code"] == "source_hash_mismatch"


def test_transport_error_raises_unavailable() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       error=httpx.ConnectError("refused"))
    with pytest.raises(DsaUnavailableError):
        _client(fake).get_capabilities()


def test_oversized_response_raises_protocol_error() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=httpx.Response(200, content=b"x" * (10_000_001)))
    with pytest.raises(DsaProtocolError):
        _client(fake).get_capabilities()


def test_non_json_response_raises_protocol_error() -> None:
    fake = _FakeClient(base_url="http://127.0.0.1:8012", timeout=30.0,
                       response=httpx.Response(200, content=b"not json"))
    with pytest.raises(DsaProtocolError):
        _client(fake).get_capabilities()


# ---------------------------------------------------------------------------
# Wire-level test against the real Mock DSA server
# ---------------------------------------------------------------------------


def test_client_against_mock_happy_path(tmp_path: Path) -> None:
    from tests.research_loop_test_helpers import MockDsaServer

    server = MockDsaServer("happy_path")
    server.start()
    try:
        client = DsaLoopClient(base_url=server.base_url, timeout=30.0)
        caps = client.get_capabilities()
        assert caps["status"] == "ok"
        assert "portfolio_backtest" in caps["data"]["execution_types"]

        payload = {
            "experiment_id": "exp_wire_001",
            "round_id": "round_1",
            "objective": "o",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": ["f"], "factor_ids": ["a"]},
            "research_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
            "gate_policy_version": "gate_champion_v1",
        }
        registered = client.register_experiment(payload, idempotency_key="exp_wire_001")
        assert registered["status"] == "ok"

        # 幂等重放：相同 key + 相同请求返回原响应，不产生第二个实验
        replay = client.register_experiment(payload, idempotency_key="exp_wire_001")
        assert replay["status"] == "ok"

        events = client.poll_events("exp_wire_001", after_sequence=0)
        assert events["status"] == "ok"
        assert any(e["event_type"] == "experiment.created" for e in events["data"]["items"])
    finally:
        server.stop()


def test_client_against_mock_locked_holdout_rejected(tmp_path: Path) -> None:
    from tests.research_loop_test_helpers import MockDsaServer

    server = MockDsaServer("happy_path")
    server.start()
    try:
        client = DsaLoopClient(base_url=server.base_url, timeout=30.0)
        payload = {
            "experiment_id": "exp_holdout_001",
            "round_id": "round_1",
            "objective": "o",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": ["f"], "factor_ids": ["a"]},
            "research_window": {"start_date": "2022-01-01", "end_date": "2025-12-31"},
            "gate_policy_version": "gate_champion_v1",
        }
        result = client.register_experiment(payload, idempotency_key="exp_holdout_001")
        assert result["status"] == "error"
        assert result["http_status"] == 403
        assert result["error"]["code"] == "locked_holdout_forbidden"
        assert result["retryable"] is False
        # 错误包不包含请求源码/凭证（结构固定）
        assert "source_code" not in json.dumps(result["error"])
    finally:
        server.stop()


def test_client_register_candidate_wire(tmp_path: Path) -> None:
    from tests.research_loop_test_helpers import MockDsaServer

    server = MockDsaServer("happy_path")
    server.start()
    try:
        client = DsaLoopClient(base_url=server.base_url, timeout=30.0)
        exp_payload = {
            "experiment_id": "exp_wire_cand_001",
            "round_id": "round_1",
            "objective": "o",
            "hypothesis": {"mechanism": "m", "prediction": "p", "failure_conditions": ["f"], "factor_ids": ["a"]},
            "research_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
            "gate_policy_version": "gate_champion_v1",
        }
        client.register_experiment(exp_payload, idempotency_key="exp_wire_cand_001")
        cand_payload = {
            "candidate_id": "candidate_wire_001",
            "candidate_version": 1,
            "contract_version": "strategy-signal.v1",
            "source_code": "def generate_scores(factors, params):\n    return 0",
            "source_sha256": "a" * 64,
            "manifest": {
                "factor_ids": ["qlib158_mom_20"],
                "top_n": 20,
                "max_holdings": 20,
            },
        }
        result = client.register_candidate("exp_wire_cand_001", cand_payload)
        assert result["status"] == "ok"
        assert result["data"]["status"] == "validated"
    finally:
        server.stop()
