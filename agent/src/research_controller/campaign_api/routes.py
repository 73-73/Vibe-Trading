"""Research Campaign HTTP routes（§7.2）。

面向用户的唯一高层入口。用户不直接创建 DSA experiment / candidate /
execution；这些由 Controller 内部完成。路由模块只做：

- 请求校验（``campaign_create_request.v1``，含 development_window 与
  ``open_locked_holdout`` 约束）；
- 转发到 :class:`~src.research_controller.state_machine.controller.ResearchCampaignController`。

注册入口由 Integration Lead 在 ``api_server.py`` 接线（§18.5 热点文件）。
"""

from __future__ import annotations

import logging
import sys as _sys
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Query, status

from src.api.helpers import _validate_path_param
from src.research_controller.contracts.generated.campaign_create_request_v1 import CampaignCreateRequest
from src.research_controller.state_machine.controller import (
    CampaignNotFoundError,
    CampaignValidationError,
    ResearchCampaignController,
)

logger = logging.getLogger(__name__)

AuthDep = Callable[..., Awaitable[Any] | Any]

_controller: ResearchCampaignController | None = None
_controller_factory: Callable[[], ResearchCampaignController] | None = None


def set_controller(controller: ResearchCampaignController) -> None:
    """Set the controller singleton used by the routes (test seam).

    Also clears any lazily-registered factory: an explicit instance must take
    precedence over a stale factory (e.g. one registered by ``api_server`` at
    import time), otherwise ``get_controller()`` could rebuild a different
    controller with another store / database.
    """
    global _controller, _controller_factory
    _controller = controller
    _controller_factory = None


def set_controller_factory(factory: Callable[[], ResearchCampaignController]) -> None:
    """Lazily provide the controller singleton.

    Lets ``api_server`` register routes without constructing the controller
    (and its SQLite store) at import time. Replacing the factory also clears
    any already-cached controller instance, so the next ``get_controller()``
    builds from the new factory instead of reusing a stale singleton (which
    would leak state across configs / databases / app instances).
    """
    global _controller, _controller_factory
    _controller = None
    _controller_factory = factory


def get_controller() -> ResearchCampaignController:
    global _controller
    if _controller is None and _controller_factory is not None:
        _controller = _controller_factory()
    if _controller is None:
        raise RuntimeError("research campaign controller not configured")
    return _controller


def register_research_campaign_routes(
    app: FastAPI,
    require_auth: AuthDep | None = None,
) -> None:
    """Mount the Research Campaign routes onto ``app``.

    Resolves ``require_auth`` from the host ``api_server`` module when not
    passed explicitly, mirroring ``src.api.scheduled_routes``.
    """
    host = _sys.modules.get("api_server") or _sys.modules.get("agent.api_server")
    if require_auth is None:
        if host is None:
            raise RuntimeError(
                "register_research_campaign_routes: api_server module not in sys.modules"
            )
        require_auth = host.require_auth

    def _controller_for() -> ResearchCampaignController:
        return get_controller()

    # --- 创建 / 列表 -------------------------------------------------

    @app.post(
        "/research-campaigns",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_auth)],
    )
    async def create_campaign(request: CampaignCreateRequest) -> dict[str, Any]:
        """Create a Research Campaign (§7.2.1)."""
        try:
            return _controller_for().create_campaign(request.model_dump(mode="json"))
        except CampaignValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/research-campaigns", dependencies=[Depends(require_auth)])
    async def list_campaigns(
        status_filter: str | None = Query(None, alias="status"),
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        """List campaigns, optionally filtered by status."""
        return _controller_for().list_campaigns(status=status_filter, limit=limit)

    @app.get("/research-campaigns/{campaign_id}", dependencies=[Depends(require_auth)])
    async def get_campaign(campaign_id: str) -> dict[str, Any]:
        """Return campaign detail (data freshness, snapshots, 357-factor
        progress, queue, current executions, recent errors/repairs, budget,
        Reviewer status and latest report reference)."""
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().get_campaign(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None

    # --- 子资源查询 ---------------------------------------------------

    @app.get("/research-campaigns/{campaign_id}/experiments", dependencies=[Depends(require_auth)])
    async def list_experiments(campaign_id: str) -> list[dict[str, Any]]:
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().get_experiments(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None

    @app.get("/research-campaigns/{campaign_id}/candidates", dependencies=[Depends(require_auth)])
    async def list_candidates(campaign_id: str) -> list[dict[str, Any]]:
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().get_candidates(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None

    @app.get("/research-campaigns/{campaign_id}/repairs", dependencies=[Depends(require_auth)])
    async def list_repairs(campaign_id: str) -> list[dict[str, Any]]:
        """Return the automatic repair lineage for the campaign (§17.5)."""
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().get_repairs(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None

    @app.get("/research-campaigns/{campaign_id}/reports/latest", dependencies=[Depends(require_auth)])
    async def latest_report(campaign_id: str) -> dict[str, Any]:
        """Return the latest Vibe Chinese research report (§15.3)."""
        _validate_path_param(campaign_id, "campaign_id")
        try:
            report = _controller_for().latest_report(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None
        if report is None:
            raise HTTPException(status_code=404, detail="no report generated yet")
        return report

    @app.post("/research-campaigns/{campaign_id}/reports/generate", dependencies=[Depends(require_auth)])
    async def generate_report(campaign_id: str) -> dict[str, Any]:
        """Trigger generation of the final Vibe Chinese report (§15.3 / R13).

        Campaign 不存在 → 404。重复调用幂等：返回已生成的既有最新报告。
        报告对无足够证据 / 缺独立评审做“含缺口”渲染，不抛异常。
        """
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().generate_report(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None

    # --- 暂停 / 恢复 / 取消（§7.2.3）-----------------------------------

    @app.post("/research-campaigns/{campaign_id}/pause", dependencies=[Depends(require_auth)])
    async def pause_campaign(campaign_id: str, cancel_running: bool = False) -> dict[str, Any]:
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().pause_campaign(campaign_id, cancel_running=cancel_running)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None
        except CampaignValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/research-campaigns/{campaign_id}/resume", dependencies=[Depends(require_auth)])
    async def resume_campaign(campaign_id: str) -> dict[str, Any]:
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().resume_campaign(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None
        except CampaignValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/research-campaigns/{campaign_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_campaign(campaign_id: str) -> dict[str, Any]:
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().cancel_campaign(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None
        except CampaignValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/research-campaigns/{campaign_id}/cancellations/reconcile",
        dependencies=[Depends(require_auth)],
    )
    async def reconcile_campaign_cancellations(campaign_id: str) -> dict[str, Any]:
        """Explicitly retry visible blocked remote-cancellation diagnostics."""
        _validate_path_param(campaign_id, "campaign_id")
        try:
            return _controller_for().reconcile_campaign_cancellations(campaign_id)
        except CampaignNotFoundError:
            raise HTTPException(status_code=404, detail="campaign not found") from None
        except CampaignValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
