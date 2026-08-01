"""Tests for the Research Campaign background driver (R01, Wave C).

覆盖：

- ``CampaignDriver`` 后台线程在 stop 前持续调用 ``run_pipeline_once``，stop 后
  线程退出、无残留；
- 双 driver 实例：第二个实例拿不到 lease 时进入只读观察（不推进）；
- 跨进程文件锁：外部持有 flock 时 driver 进入只读观察（不推进）；
- startup / shutdown 事件：``register_research_campaign`` 装配的 app 在
  TestClient 启动时构造并启动 driver、关闭时停止。

所有 driver 测试使用注入的假 controller 与临时目录 lock 文件，不依赖 DSA mock，
也不写用户 home。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.research_controller.campaign_api.assemble import register_research_campaign
from src.research_controller.campaign_api.routes import (
    get_controller,
    set_controller,
    set_controller_factory,
)
from src.research_controller.state_machine.driver import CampaignDriver

try:  # pragma: no cover - Windows
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

_DRIVER_THREAD_NAME = "research-campaign-driver"
_STATE_LOCK = threading.Lock()


class FakeController:
    """记录调用次数的假 controller，供 driver 测试注入。"""

    def __init__(self, campaign_ids: list[str], interval: float = 0.05) -> None:
        self.campaign_ids = list(campaign_ids)
        self.interval = float(interval)
        self.pipeline_calls: list[str] = []

    def list_active_campaign_ids(self) -> list[str]:
        return list(self.campaign_ids)

    def run_pipeline_once(self, cid: str) -> dict[str, Any]:
        self.pipeline_calls.append(cid)
        return {"campaign_id": cid}

    def poll_interval_seconds(self, cid: str) -> float:
        del cid
        return self.interval

    def list_campaigns(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        del status, limit
        return []


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _make_driver(
    get_controller: Callable[[], Any],
    lock_path: Path,
    *,
    interval_floor: float = 0.05,
    max_interval: float = 0.5,
) -> CampaignDriver:
    return CampaignDriver(
        get_controller,
        interval_floor=interval_floor,
        max_interval=max_interval,
        lock_path=lock_path,
    )


@pytest.fixture(autouse=True)
def _clean_controller_and_lease():
    """每个测试前后重置全局 controller 单例与进程内 lease，避免跨测试污染。"""
    set_controller_factory(None)
    yield
    set_controller_factory(None)
    CampaignDriver._process_advance_holder = False
    lingering = [t for t in threading.enumerate() if t.name == _DRIVER_THREAD_NAME and t.is_alive()]
    assert not lingering, f"lingering driver threads after test: {lingering}"


# ---------------------------------------------------------------------------
# 1. run_forever 持续调用 run_pipeline_once，stop 后退出无残留
# ---------------------------------------------------------------------------


def test_run_forever_advances_until_stop(tmp_path: Path) -> None:
    fake = FakeController(["c1"], interval=0.05)
    driver = _make_driver(lambda: fake, tmp_path / "driver.lock")
    driver.start()
    try:
        assert driver.advance_enabled is True
        assert _wait_until(lambda: len(fake.pipeline_calls) >= 3)
        assert all(cid == "c1" for cid in fake.pipeline_calls)
        assert driver.tick_count >= 3
        assert driver.last_tick_at is not None
        assert driver.is_running()
    finally:
        driver.stop()

    # stop 后：线程退出、无残留推进
    assert driver.is_running() is False
    assert driver._thread is None
    calls_after_stop = len(fake.pipeline_calls)
    time.sleep(0.15)
    assert len(fake.pipeline_calls) == calls_after_stop
    assert not [t for t in threading.enumerate() if t.name == _DRIVER_THREAD_NAME and t.is_alive()]


def test_run_forever_survives_tick_exception(tmp_path: Path) -> None:
    """单次 tick 抛异常不退出线程，且不阻断后续 campaign 推进。"""

    class ExplodingController(FakeController):
        def run_pipeline_once(self, cid: str) -> dict[str, Any]:
            if cid == "boom":
                raise RuntimeError("boom")
            return super().run_pipeline_once(cid)

    fake = ExplodingController(["boom", "ok"], interval=0.05)
    driver = _make_driver(lambda: fake, tmp_path / "driver.lock")
    driver.start()
    try:
        # ok 仍被推进（boom 的异常被捕获并记录）
        assert _wait_until(lambda: fake.pipeline_calls.count("ok") >= 2)
        assert driver.tick_count >= 2
        assert driver.is_running()
    finally:
        driver.stop()


class FakeStore:
    """仅暴露 ``list_campaigns`` 的假 store，用于验证 driver 的 store 回退。"""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = list(rows)

    def list_campaigns(self, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if status is None:
            return list(self._rows)
        return [r for r in self._rows if r["status"] == status]


class NoListerController(FakeController):
    """模拟尚未提供 ``list_active_campaign_ids`` 的 controller：只暴露 ``store``。"""

    list_active_campaign_ids = None  # type: ignore[assignment]

    def __init__(self, store: FakeStore, interval: float = 0.05) -> None:
        super().__init__(campaign_ids=[], interval=interval)
        self.store = store


def test_driver_falls_back_to_store_for_active_ids(tmp_path: Path) -> None:
    """controller 无 ``list_active_campaign_ids`` 时，driver 用 ``store.list_campaigns``
    按活跃状态过滤（与 controller 公开方法语义一致：不轮询 paused / 终态）。"""
    store = FakeStore(
        [
            {"campaign_id": "c-init", "status": "initializing"},
            {"campaign_id": "c-running", "status": "running"},
            {"campaign_id": "c-waiting", "status": "waiting_budget"},
            {"campaign_id": "c-paused", "status": "paused"},
            {"campaign_id": "c-blocked", "status": "blocked"},
            {"campaign_id": "c-completed", "status": "completed"},
            {"campaign_id": "c-cancelled", "status": "cancelled"},
        ]
    )
    fake = NoListerController(store, interval=0.05)
    driver = _make_driver(lambda: fake, tmp_path / "driver.lock")
    driver.start()
    try:
        assert _wait_until(lambda: len(fake.pipeline_calls) >= 3)
        assert sorted(set(fake.pipeline_calls)) == ["c-init", "c-running", "c-waiting"]
    finally:
        driver.stop()


# ---------------------------------------------------------------------------
# 2. 进程内 lease：第二个 driver 实例进入只读观察
# ---------------------------------------------------------------------------


def test_second_driver_instance_read_only(tmp_path: Path) -> None:
    fake1 = FakeController(["c1"], interval=0.05)
    fake2 = FakeController(["c1"], interval=0.05)
    lock = tmp_path / "driver.lock"
    driver1 = _make_driver(lambda: fake1, lock)
    driver1.start()
    try:
        assert driver1.advance_enabled is True
        driver2 = _make_driver(lambda: fake2, lock)
        driver2.start()
        try:
            assert driver2.advance_enabled is False
            # driver1 持续推进
            assert _wait_until(lambda: len(fake1.pipeline_calls) >= 3)
            # driver2 只读观察：仍跑 tick（记录 last_tick_at / tick_count），但不推进
            assert _wait_until(lambda: driver2.tick_count >= 1)
            time.sleep(0.1)
            assert fake2.pipeline_calls == []
            assert driver2.last_tick_at is not None
        finally:
            driver2.stop()
    finally:
        driver1.stop()


# ---------------------------------------------------------------------------
# 3. 跨进程文件锁：外部持有 flock 时 driver 只读观察
# ---------------------------------------------------------------------------


def test_file_lock_held_externally_blocks_advance(tmp_path: Path) -> None:
    if fcntl is None:  # pragma: no cover - Windows
        pytest.skip("fcntl not available")
    lock = tmp_path / "driver.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fake = FakeController(["c1"], interval=0.05)
        # 显式清空进程内 lease，确保本次走文件锁判定
        CampaignDriver._process_advance_holder = False
        driver = _make_driver(lambda: fake, lock)
        driver.start()
        try:
            assert driver.advance_enabled is False
            assert _wait_until(lambda: driver.tick_count >= 2)
            time.sleep(0.1)
            assert fake.pipeline_calls == []
        finally:
            driver.stop()
    finally:
        os.close(fd)


def test_file_lock_released_after_stop_allows_new_driver(tmp_path: Path) -> None:
    if fcntl is None:  # pragma: no cover - Windows
        pytest.skip("fcntl not available")
    lock = tmp_path / "driver.lock"
    fake1 = FakeController(["c1"], interval=0.05)
    driver1 = _make_driver(lambda: fake1, lock)
    driver1.start()
    try:
        assert driver1.advance_enabled is True
    finally:
        driver1.stop()

    # stop 已释放文件锁与进程 lease：新 driver 可重新拿到 lease
    fake2 = FakeController(["c1"], interval=0.05)
    driver2 = _make_driver(lambda: fake2, lock)
    driver2.start()
    try:
        assert driver2.advance_enabled is True
        assert _wait_until(lambda: len(fake2.pipeline_calls) >= 2)
    finally:
        driver2.stop()


# ---------------------------------------------------------------------------
# 4. startup / shutdown 事件接线
# ---------------------------------------------------------------------------


def test_register_research_campaign_startup_shutdown_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 让默认 runtime root 落到 tmp_path，避免测试写用户 home 的 lock 文件
    monkeypatch.setattr("src.config.paths.get_runtime_root", lambda: tmp_path)

    fake = FakeController(["c1"], interval=0.05)
    app = FastAPI()
    register_research_campaign(app, require_auth=lambda: None)
    # 用假 controller 覆盖默认工厂，避免真实 store / DSA client
    set_controller(fake)
    assert get_controller() is fake

    with TestClient(app) as client:
        driver: CampaignDriver = app.state.campaign_driver
        assert driver is not None
        assert driver.is_running()
        assert driver.advance_enabled is True
        assert _wait_until(lambda: len(fake.pipeline_calls) >= 2)
        # 路由可访问（controller 单例与 driver 共享）
        response = client.get("/research-campaigns")
        assert response.status_code == 200
        assert response.json() == []

    # shutdown 已停止 driver
    assert driver.is_running() is False
    calls_after_shutdown = len(fake.pipeline_calls)
    time.sleep(0.15)
    assert len(fake.pipeline_calls) == calls_after_shutdown


def test_driver_not_started_without_lifespan() -> None:
    """不进入 TestClient 上下文（不触发 startup）时不应启动后台线程。

    对应“测试环境不产生后台线程泄漏”：既有 campaign_api 测试用裸 TestClient
    （不进 ``with``），必须不会因此拉起 driver 线程。
    """
    fake = FakeController(["c1"], interval=0.05)
    app = FastAPI()
    register_research_campaign(app, require_auth=lambda: None)
    set_controller(fake)
    client = TestClient(app)  # 未进入 with：startup 不运行
    assert not hasattr(app.state, "campaign_driver")
    client.close()  # 只关闭 transport，不触发 shutdown
    assert not [t for t in threading.enumerate() if t.name == _DRIVER_THREAD_NAME and t.is_alive()]
