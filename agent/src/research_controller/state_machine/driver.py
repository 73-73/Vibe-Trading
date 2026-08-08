"""Research Campaign 内置后台 Worker（R01，设计决策 A：Vibe 内置后台线程 + 进程内 lease）。

在 ``api_server`` 生命周期内常驻一个后台线程，持续推进所有非终态 campaign：
每个 tick 对每个活跃 campaign 调用 ``ResearchCampaignController.run_pipeline_once``，
节奏取自 ``poll_interval_seconds``（§6.6），并按 ``[interval_floor, max_interval]``
收敛。

并发安全：

- 进程内单实例 lease：类级 ``threading.Lock`` + ``_process_advance_holder``
  保证同一进程只有一个 driver 实例推进；其余实例进入只读观察（仅记录 tick，
  不调用 ``run_pipeline_once``）。
- 跨进程文件锁：``fcntl.flock``（runtime root 下 ``campaigns_driver.lock``）防止
  launchd 意外多实例时同机多进程双写；拿不到锁的进程只读观察。

可观测性：每个 tick 输出 INFO 日志（活跃 campaign 数、poll_interval、累计 tick
数），并暴露 ``last_tick_at`` / ``tick_count`` 供健康检查。

本模块不修改 ``controller.py``：活跃 campaign 发现优先复用 controller 公开方法
（``list_active_campaign_ids``），不存在时退化为 ``store.list_campaigns(status=...)``
按活跃状态过滤。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import paths as _paths
from src.research_controller.state_machine.machine import POLL_INTERVAL_IDLE_SECONDS

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 平台差异（Windows 无 fcntl）
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

# 与 ``controller.list_active_campaign_ids``（§13）语义一致的活跃状态集合：
# paused / cancelled / blocked / completed 不轮询（``run_pipeline_once`` 对
# paused 早退，driver 不破坏用户主动暂停）。仅作为 controller 尚未提供公开方法
# 时的防御性回退（避免改 controller.py）。
_ACTIVE_CAMPAIGN_STATUSES: tuple[str, ...] = ("initializing", "running", "waiting_budget")

_DRIVER_THREAD_NAME = "research-campaign-driver"


class CampaignDriver:
    """常驻后台线程，持续推进所有非终态 Research Campaign。

    Args:
        get_controller: 返回 :class:`ResearchCampaignController` 的可调用对象。
            复用 ``campaign_api.routes.get_controller`` 单例，保证 driver 与
            HTTP 层共享同一个 controller / store。
        interval_floor: 两次 tick 的最小间隔（秒），默认 2.0。
        max_interval: 两次 tick 的最大间隔（秒），默认 60.0。
        lock_path: flock 文件锁路径；默认 runtime root 下 ``campaigns_driver.lock``。
            测试可注入临时目录路径，避免写用户 home。
        discovery_limit: 每个 status 查询的 campaign 上限（store 上限 500）。
    """

    # 进程内单实例 lease：同一进程只允许一个 driver 推进
    _process_lease_lock = threading.Lock()
    _process_advance_holder = False

    def __init__(
        self,
        get_controller: Callable[[], Any],
        *,
        interval_floor: float = 2.0,
        max_interval: float = 60.0,
        lock_path: Path | str | None = None,
        discovery_limit: int = 500,
    ) -> None:
        self._get_controller = get_controller
        self.interval_floor = float(interval_floor)
        self.max_interval = float(max_interval)
        if self.max_interval < self.interval_floor:
            self.max_interval = self.interval_floor
        self.discovery_limit = int(discovery_limit)
        self.lock_path = (
            Path(lock_path)
            if lock_path is not None
            else _paths.get_runtime_root() / "campaigns_driver.lock"
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock_fd: int | None = None
        self.advance_enabled = False
        self.tick_count = 0
        self.last_tick_at: float | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> "CampaignDriver":
        """启动后台线程（幂等；已运行时直接返回）。

        只有拿到 lease（进程内 + 跨进程文件锁）的实例才推进；否则进入只读观察。
        """
        if self._thread is not None and self._thread.is_alive():
            return self
        with self._process_lease_lock:
            if self._process_advance_holder:
                self.advance_enabled = False
            else:
                self.advance_enabled = self._try_acquire_file_lock()
                if self.advance_enabled:
                    self._process_advance_holder = True
        self._stop_event.clear()
        thread = threading.Thread(
            target=self.run_forever,
            name=_DRIVER_THREAD_NAME,
            daemon=True,
        )
        self._thread = thread
        thread.start()
        logger.info(
            "research campaign driver started (advance_enabled=%s lock=%s)",
            self.advance_enabled,
            self.lock_path,
        )
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """请求线程退出并释放 lease（幂等）。

        若线程正卡在 ``run_pipeline_once``（DSA 长调用），join 超时后线程仍会
        在下一个 ``stop_event.wait`` 退出；daemon 线程不阻塞进程退出。
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("research campaign driver thread did not stop within %.1fs", timeout)
        self._thread = None
        self._release_file_lock()
        with self._process_lease_lock:
            if self.advance_enabled and self._process_advance_holder:
                self._process_advance_holder = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """后台主循环：持续推进非终态 campaign，直到 stop()。

        单次 tick 的任何异常都会被记录并以 ``max_interval`` 退避，线程不退出。
        """
        logger.info(
            "research campaign driver run_forever entering loop (advance=%s)",
            self.advance_enabled,
        )
        while not self._stop_event.is_set():
            try:
                interval = self._tick()
            except Exception:
                logger.exception("research campaign driver tick failed; backing off")
                interval = self.max_interval
            self.tick_count += 1
            self.last_tick_at = time.time()
            self._stop_event.wait(interval)

    def _tick(self) -> float:
        controller = self._get_controller()
        active_ids = self.list_active_campaign_ids(controller)
        if self.advance_enabled:
            for cid in active_ids:
                try:
                    controller.run_pipeline_once(cid)
                except Exception:
                    logger.exception("research campaign driver failed to advance %s", cid)
        interval = self._next_interval(controller, active_ids)
        logger.info(
            "research campaign driver tick=%d active_campaigns=%d poll_interval=%.1fs advance=%s",
            self.tick_count,
            len(active_ids),
            interval,
            self.advance_enabled,
        )
        return interval

    def list_active_campaign_ids(self, controller: Any | None = None) -> list[str]:
        """返回所有活跃（非终态）campaign id。

        优先复用 controller 公开方法 ``list_active_campaign_ids``（若后续 controller
        新增该方法则直接复用，语义与 ``_ACTIVE_CAMPAIGN_STATUSES`` 一致）；否则用
        ``store.list_campaigns(status=...)`` 按活跃状态过滤，避免修改 controller.py。
        """
        ctl = controller if controller is not None else self._get_controller()
        lister = getattr(ctl, "list_active_campaign_ids", None)
        if callable(lister):
            return list(lister())
        store = getattr(ctl, "store", None)
        if store is None or not hasattr(store, "list_campaigns"):
            raise RuntimeError(
                "research campaign controller exposes neither list_active_campaign_ids "
                "nor store.list_campaigns for active campaign discovery"
            )
        ids: list[str] = []
        for status in _ACTIVE_CAMPAIGN_STATUSES:
            for row in store.list_campaigns(status=status, limit=self.discovery_limit):
                ids.append(row["campaign_id"])
        seen: set[str] = set()
        return [cid for cid in ids if not (cid in seen or seen.add(cid))]

    def _next_interval(self, controller: Any, active_ids: Sequence[str]) -> float:
        """按 §6.6 节奏取下一 tick 间隔，并在 ``[interval_floor, max_interval]`` 内收敛。

        无活跃 campaign 时使用控制器自身的 idle 轮询节奏（15s），保证新建 campaign
        在 ~15s 内被拾起。多个 campaign 取最小值（至少与最活跃的那个同频）。
        """
        if not active_ids:
            return min(max(float(POLL_INTERVAL_IDLE_SECONDS), self.interval_floor), self.max_interval)
        intervals: list[float] = []
        for cid in active_ids:
            try:
                raw = float(controller.poll_interval_seconds(cid))
            except Exception:
                raw = self.max_interval
            intervals.append(min(max(raw, self.interval_floor), self.max_interval))
        return min(intervals)

    # ------------------------------------------------------------------
    # lease
    # ------------------------------------------------------------------

    def _try_acquire_file_lock(self) -> bool:
        if not _HAVE_FCNTL:
            # 无 fcntl 平台（如 Windows）：退化为仅进程内 lease，跨进程双写由上层负责
            logger.warning("fcntl unavailable; campaign driver uses process-only lease")
            return True
        lock_path = self.lock_path
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            logger.warning(
                "cannot open campaign driver lock file %s; using process-only lease",
                lock_path,
            )
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            logger.info(
                "campaign driver file lock %s held elsewhere; read-only observation mode",
                lock_path,
            )
            return False
        self._lock_fd = fd
        return True

    def _release_file_lock(self) -> None:
        fd = self._lock_fd
        self._lock_fd = None
        if fd is None:
            return
        if _HAVE_FCNTL:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass
