"""Shared fixtures for M4 Research Controller tests.

Provides an in-process Mock DSA Server (uvicorn on an ephemeral loopback port)
so tests exercise the real ``research-loop.v1`` HTTP surface (§18.4). The mock
lives in the DSA repo; tests skip with a clear message when it is not available.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

DSA_BACKTEST_LAB_PATH = Path("/Users/pandeng/Projects/daily_stock_analysis/services/backtest_lab")


def _import_mock():
    if not DSA_BACKTEST_LAB_PATH.is_dir():
        pytest.skip(f"DSA mock server not available at {DSA_BACKTEST_LAB_PATH}")
    if str(DSA_BACKTEST_LAB_PATH) not in sys.path:
        sys.path.insert(0, str(DSA_BACKTEST_LAB_PATH))
    try:
        from research_loop.mocks.mock_lab import MockScenario
        from research_loop.mocks.mock_server import create_mock_app, SCENARIOS
    except Exception as exc:  # pragma: no cover - depends on DSA checkout
        pytest.skip(f"DSA mock server not importable: {exc}")
    return create_mock_app, MockScenario, SCENARIOS


class MockDsaServer:
    """Runs the DSA mock server (uvicorn) on an ephemeral loopback port."""

    def __init__(self, scenario: str = "happy_path") -> None:
        self.scenario = scenario
        self.base_url: str | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        import uvicorn

        create_mock_app, _mock_scenario, _scenarios = _import_mock()
        if self.scenario not in _scenarios:
            pytest.skip(f"unknown mock scenario: {self.scenario}")

        class _Server(uvicorn.Server):
            def install_signal_handlers(self) -> None:
                pass

        app = create_mock_app(self.scenario)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = _Server(config)
        self._server = server
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 15.0
        while not server.started:
            if time.time() > deadline:
                raise RuntimeError("mock DSA server failed to start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> "MockDsaServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


@pytest.fixture
def mock_dsa() -> Iterator[MockDsaServer]:
    """Start a happy_path Mock DSA server for the duration of one test."""
    server = MockDsaServer("happy_path")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def make_mock_dsa(scenario: str) -> MockDsaServer:
    """Start a mock DSA server for a specific scenario (caller stops it)."""
    server = MockDsaServer(scenario)
    server.start()
    return server


def make_controller(
    store,
    base_url: str,
    *,
    queue_target: int = 20,
    refill_threshold: int = 5,
    **kwargs: Any,
):
    """Build a ResearchCampaignController wired to *base_url*."""
    from src.research_controller.client.dsa_client import DsaLoopClient
    from src.research_controller.state_machine.controller import ResearchCampaignController

    client = DsaLoopClient(base_url=base_url, timeout=30.0)
    # The DSA protocol mock intentionally does not claim production runtime
    # readiness.  Existing controller state-machine tests inject an explicit
    # all-green capability document so they remain scoped to transitions; the
    # production assembly never uses this seam.  Dedicated preflight tests
    # exercise missing/degraded fields fail-closed.
    kwargs.setdefault("capabilities_provider", ready_capabilities_provider)
    return ResearchCampaignController(
        store,
        client,
        queue_target=queue_target,
        refill_threshold=refill_threshold,
        **kwargs,
    )


def ready_capabilities_provider() -> dict[str, Any]:
    """Return a complete, deterministic test-only Campaign preflight response."""
    from src.research_controller.contracts import source_bundle_sha256
    from src.research_controller.state_machine.controller import REQUIRED_EXECUTION_TYPES

    return {
        "status": "ok",
        "action": "get_research_loop_capabilities",
        "http_status": 200,
        "retryable": False,
        "data": {
            "contract_bundle_sha256": source_bundle_sha256(),
            "protocol_versions": ["research-loop.v1"],
            "strategy_contracts": ["strategy-signal.v1"],
            "research_sdk_versions": ["research_sdk.v1"],
            "data_snapshot_versions": ["data_snapshot.v1"],
            "universe_snapshot_versions": ["universe_snapshot.v1"],
            "market_panel_versions": ["market_panel.v1"],
            "factor_snapshot_versions": ["factor_snapshot.v1"],
            "evidence_bundle_versions": ["evidence_bundle.v1"],
            "review_report_versions": ["review_report.v1"],
            "markets": ["CN"],
            "frequencies": ["1d"],
            "execution_types": list(REQUIRED_EXECUTION_TYPES),
            "artifact_uploads": {
                "schema_version": "artifact_upload_capabilities.v1",
                "allowlist": {
                    "factor_values": ["application/vnd.apache.parquet", "text/csv"],
                },
            },
            "readiness": {
                name: {
                    "status": "ready",
                    "available": True,
                    "ready": True,
                    "reason": "test-only explicit readiness",
                }
                for name in ("sandbox", "reviewer", "data", "universe", "longbridge")
            },
        },
    }


def sample_campaign_request(**overrides: Any) -> dict[str, Any]:
    """A valid §7.2.1 campaign creation request."""
    request: dict[str, Any] = {
        "objective": "研究 A 股量价因子组合的横截面稳定性",
        "market": "CN",
        "frequency": "1d",
        "development_window": {"start_date": "2010-01-01", "end_date": "2021-12-31"},
        "factor_scope": "a_share_native_357",
        "universe_policy": "pit_liquid_active_v1",
        "target_universe_size": 1200,
        "automation": {
            "generate_hypotheses": True,
            "generate_strategy_code": True,
            "auto_repair": True,
            "parameter_search": True,
            "reinforcement_learning_challengers": False,
            "gate_challengers": False,
            "open_locked_holdout": False,
        },
        "budgets": {
            "llm_calls_rolling_24h": 100,
            "candidate_versions_rolling_24h": 50,
            "sandbox_concurrency": 1,
            "factor_batch_concurrency": 2,
        },
        "report_language": "zh-CN",
    }
    request.update(overrides)
    return request
