"""Research Campaign 装配帮助（§18.5 热点文件接线）。

把 §7.2 Campaign 路由注册压缩为对 ``api_server`` 的单行调用，保持
api_server 为 thin assembler。controller 懒加载（首次 handler 调用时构造），
import 本模块无文件系统副作用。后台推进 Worker（R01）通过 startup/shutdown
事件随 ``app`` 生命周期启停，同样不在 import 时启动。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from src.research_controller.state_machine.driver import CampaignDriver

AuthDep = Callable[..., Awaitable[Any] | Any]


def register_research_campaign(app: FastAPI, require_auth: AuthDep | None = None) -> None:
    """Mount §7.2 Research Campaign routes (lazy controller) + background driver.

    - ``set_controller_factory`` / ``register_research_campaign_routes`` 与之前一致。
    - 额外注册 startup / shutdown 事件处理器，构造并启停 ``CampaignDriver``
      （复用 ``routes.get_controller`` 单例），保证 api_server 生命周期内持续推进
      所有非终态 campaign。事件处理器幂等：重复注册不会重复启动线程。
    """
    from src.research_controller.campaign_api.routes import (
        get_controller,
        register_research_campaign_routes,
        set_controller_factory,
    )
    from src.research_controller.client.dsa_client import DsaLoopClient
    from src.research_controller.state_machine.controller import ResearchCampaignController
    from src.research_controller.store.campaign_store import CampaignStore

    def _factory() -> ResearchCampaignController:
        return ResearchCampaignController(store=CampaignStore(), dsa_client=DsaLoopClient())

    set_controller_factory(_factory)
    register_research_campaign_routes(app, require_auth=require_auth)

    def _start_campaign_driver() -> None:
        existing: CampaignDriver | None = getattr(app.state, "campaign_driver", None)
        if existing is not None and existing.is_running():
            return
        driver = CampaignDriver(get_controller=get_controller)
        driver.start()
        app.state.campaign_driver = driver

    def _stop_campaign_driver() -> None:
        driver: CampaignDriver | None = getattr(app.state, "campaign_driver", None)
        if driver is not None:
            driver.stop()

    app.router.add_event_handler("startup", _start_campaign_driver)
    app.router.add_event_handler("shutdown", _stop_campaign_driver)
