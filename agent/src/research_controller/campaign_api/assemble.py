"""Research Campaign 装配帮助（§18.5 热点文件接线）。

把 §7.2 Campaign 路由注册压缩为对 ``api_server`` 的单行调用，保持
api_server 为 thin assembler。controller 懒加载（首次 handler 调用时构造），
import 本模块无文件系统副作用。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import FastAPI

AuthDep = Callable[..., Awaitable[Any] | Any]


def register_research_campaign(app: FastAPI, require_auth: AuthDep | None = None) -> None:
    """Mount §7.2 Research Campaign routes (lazy controller)."""
    from src.research_controller.campaign_api.routes import (
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
