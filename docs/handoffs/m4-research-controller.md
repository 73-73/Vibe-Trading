# M4 Vibe Research Controller — Handoff（§18.6）

> 模块：`M4-research-controller`
> 仓库：`Vibe-Trading`（Agent 侧）
> 日期：2026-08-01
> 状态：对 Mock DSA 开发完成；真实 DSA（M1）与沙箱（M2）尚未完成，Mock 通过
> 不能代替集成通过（§18.4 / §20）。

## base_commit

- Vibe 基线：`4ed9a6d`（M0：research-loop.v1 contract bundle 副本 + 生成客户端模型）
- DSA 基线：未改动 DSA 仓库代码（M0 冻结 bundle，`bundle_manifest.json` 未漂移）
- 协议 bundle：`source_bundle_sha256()` 保持不变（`verify_local_copy()` 通过）

## changed_files

新增（`agent/src/research_controller/`）：

| 文件 | 职责 |
| --- | --- |
| `client/__init__.py` | 包声明 |
| `client/dsa_client.py` | loopback-only DSA research-loop.v1 HTTP 客户端（§7.1 / §4） |
| `store/__init__.py` | 包声明 |
| `store/canonical.py` | RFC 8785 风格 canonical JSON + SHA-256（§4.3） |
| `store/campaign_store.py` | research_campaign.v1 SQLite 持久化；`apply_bundle` 单事务（§7.2.1） |
| `state_machine/__init__.py` | 包声明 |
| `state_machine/machine.py` | §13.1 阶段常量、§6.6 轮询间隔、§13.3 预算默认值 |
| `state_machine/controller.py` | `ResearchCampaignController`（§7.2 / §8 / §13） |
| `repair/__init__.py` | 包声明 |
| `repair/lineage.py` | 自动修复谱系守卫（§8.3 纯函数） |
| `reporting/__init__.py` | 包声明 |
| `reporting/report.py` | §15.3 九段中文报告纯渲染函数 |
| `campaign_api/__init__.py` | 包声明 |
| `campaign_api/routes.py` | Research Campaign HTTP API 路由（§7.2） |

新增测试（`agent/tests/`）：

| 文件 | 覆盖 |
| --- | --- |
| `research_loop_test_helpers.py` | 进程内 Mock DSA（uvicorn 临时端口）+ 常用 fixture |
| `test_research_controller_client.py` | client 校验 / 映射 / 幂等 / 可重试 / 64KiB / wire |
| `test_research_controller_store.py` | SQLite CRUD / 单事务 bundle / 去重 / 不可变 decision / 预算 |
| `test_research_controller_controller.py` | 状态机 / 生命周期 / 重启恢复 / 队列 / 预算 / 游标过期 |
| `test_research_controller_repair.py` | 修复谱系纯函数 + execution/static failure 闭环 |
| `test_research_controller_report.py` | §15.3 报告结构与哈希内容 |
| `test_research_controller_campaign_api.py` | Campaign HTTP API 端点与校验 |
| `test_research_controller_mcp_tools.py` | MCP research-loop 工具映射 / 约束 / stdio 集成 |
| `test_research_controller_integration.py` | 子进程 uvicorn Mock 端到端 + 重启恢复 |

修改（既有文件）：

- `agent/dsa_lab_mcp_server.py`：追加 18 个 research-loop.v1 MCP 工具（§7.1），
  不改名/不删除一期工具。
- `agent/tests/test_dsa_lab_mcp_server.py`：扩展 `test_dsa_lab_tools_registered`
  期望工具集，纳入新增工具（一期工具集合不变）。

## public_interfaces

### MCP v1 工具（`agent/dsa_lab_mcp_server.py`，§7.1 表）

新增 18 个：`get_research_loop_capabilities`、`register_research_experiment`、
`register_strategy_candidate`、`start_research_execution`、
`get_research_execution`、`poll_research_events`、`get_research_error`、
`get_research_evidence`、`get_research_review`、`record_research_decision`、
`cancel_research_execution`、`build_data_snapshot`、`get_data_snapshot`、
`export_market_panel`、`create_artifact_upload`、`complete_artifact_upload`、
`register_factor_snapshot`、`get_factor_snapshot`。
现有工具（`health` / `refresh_catalog` / `list_strategies` / `preflight` /
`create_batch` / `list_batches` / `get_batch` / `list_runs` / `get_run` /
`cancel_batch` / `run_factor_research`）保持兼容。

新工具使用环境变量 `DSA_RESEARCH_LOOP_URL`（默认 `http://127.0.0.1:8012`，
Mock 端口）；一期工具仍用 `DSA_LAB_URL`（默认 8011）。均 loopback-only。

### Research Campaign API（§7.2）

`register_research_campaign_routes(app, require_auth=None)`：
`POST /research-campaigns`、`GET /research-campaigns`、
`GET /research-campaigns/{id}`、`GET /research-campaigns/{id}/experiments`、
`GET /research-campaigns/{id}/candidates`、
`GET /research-campaigns/{id}/repairs`、
`GET /research-campaigns/{id}/reports/latest`、
`POST /research-campaigns/{id}/pause?cancel_running=`、
`POST /research-campaigns/{id}/resume`、
`POST /research-campaigns/{id}/cancel`。

由 Integration Lead 在 `api_server.py` 接线（§18.5 热点文件，本模块未修改）。

### Controller（`src.research_controller.state_machine.controller`）

`ResearchCampaignController(store, dsa_client, *, hypothesis_generator,
code_generator, queue_target=20, refill_threshold=5, snapshot_id_resolver,
max_repair_versions=3, max_consecutive_fingerprint=2)`。

公开方法：`create_campaign` / `get_campaign` / `list_campaigns` /
`pause_campaign` / `resume_campaign` / `cancel_campaign` / `get_experiments` /
`get_candidates` / `get_repairs` / `get_decisions` / `latest_report` /
`generate_report` / `run_pipeline_once` / `poll_events_once` /
`reconcile_executions_once` / `poll_interval_seconds` / `reset_availability`。

### DSA 客户端（`src.research_controller.client.dsa_client`）

`DsaLoopClient(base_url=None, timeout=None, client_factory=None)` 覆盖全部
research-loop 端点；`validate_loopback_base_url` / `validate_resource_id` /
`http_status_retryable` / `source_code_too_large` / `DsaUnavailableError` /
`DsaProtocolError`。

## tests_run

运行命令（验收命令 + 相关既有测试）：

```bash
/Users/pandeng/Projects/Vibe-Trading/.venv/bin/python -m pytest \
  agent/tests/test_research_controller_client.py \
  agent/tests/test_research_controller_store.py \
  agent/tests/test_research_controller_controller.py \
  agent/tests/test_research_controller_repair.py \
  agent/tests/test_research_controller_report.py \
  agent/tests/test_research_controller_campaign_api.py \
  agent/tests/test_research_controller_mcp_tools.py \
  agent/tests/test_research_controller_integration.py \
  agent/tests/test_research_controller_contracts.py \
  agent/tests/test_dsa_lab_mcp_server.py \
  -q
```

也执行：`python -m ruff check <changed_files>`、`python -m py_compile <changed_files>`。

## test_results

- 验收测试命令：**140 passed**（含 4 个 contracts 测试与既有 `test_dsa_lab_mcp_server.py`）。
- `ruff check`（E/F/W，line-length 120 忽略）：`All checks passed!`
- `py_compile` 全部改动 Python 文件：通过。
- 端到端行为（对 Mock DSA）：
  - happy_path：create campaign → data snapshot → experiment → candidate →
    sandbox → backtest → review → accept_result → robustness → completed；
    事件按 message_id 去重落库。
  - execution_failure：v1 失败 → Reviewer(code_bug) → decision
    `submit_candidate_revision` → v2 提交（parent=1, repair_of_error_id,
    change_summary）→ 相同指纹连续 2 次 → 提前停止 → abandon → rejected。
  - candidate_static_failure：CONTRACT_ERROR/python_syntax_error →
    v2 修复版本 → 连续指纹停止 → rejected。
  - reviewer_failure：无 review.completed；报告标记“缺少 DSA 独立评审”。
  - duplicate_events：重复 message_id 不重复落库。
  - cursor_expired：410 → campaign `blocked` + blocked_reason。
  - 重启恢复：新 store/controller 实例复用同一 db → 不重复提交候选/执行，
    不把 RUNNING 重置为 PENDING。
  - pause/resume/cancel 语义符合 §7.2.3。

## migrations

- Vibe runtime root 下独立 SQLite：`~/.vibe-trading/research_controller/campaigns.db`
  （`get_runtime_root()/research_controller/campaigns.db`），schema 由
  `CampaignStore._init_schema()` 自动创建（`CREATE TABLE IF NOT EXISTS`）。
- 不修改 DSA SQLite、不修改 scheduled job JSON 语义。
- 无手动数据迁移步骤；新增表不影响既有 store。

## known_gaps

- **真实 DSA（M1）未完成**：Controller 只对 Mock DSA 验证；`DsaLoopClient`
  base_url 可配置，最终可切换真实 DSA，但切换后需重跑 M1 的幂等/游标语义。
- **Docker/Colima 沙箱（M2）未实现**：策略代码只做契约级静态校验，Mock 不执行
  真实沙箱；真实执行隔离在 M2 验收范围内。
- **357 因子漏斗（WP5）未实现**：`FACTOR_INVENTORY` / `FACTOR_COMPUTE` 阶段只做
  记账（target=357, completed=0），真实因子计算由 WP5 填充。
- **`snapshot_id_resolver` Mock 兜底**：Mock 的 `build_data_snapshot` 不返回
  `data_snapshot_id`，Controller 默认调用 `get_data_snapshot("data_dev_001")`
  解析 golden id；真实 DSA 应在 build 返回或事件中携带 snapshot id（§6.12）。
- **Reviewer 超时降级**：review.failed 时 Controller 按 gate 结果自行决定并在
  报告标记“缺少 DSA 独立评审”；5 分钟等待窗口未做（Mock 同步返回）。
- **Campaign API 未接入 `api_server.py`**：路由注册由 Integration Lead 在热点
  文件接线（§18.5）；测试通过 `register_research_campaign_routes` 自行挂载验证。
- **对话层工具（§7.2.4）未实现**：`create_research_campaign` 等 9 个用户对话工具
  需要上层 Agent 工具注册，本模块只提供 Controller/API 能力。
- **单实例 lease / launchd（§16）未实现**：属于 WP7。
- **MCP 新工具未在 agent.json 默认配置启用**：`DSA_RESEARCH_LOOP_URL` 需要由
  Integration Lead 在部署配置中指向真实 DSA。

## security_assumptions

- 仅 loopback HTTP（`127.0.0.1`/`::1`/`localhost`）；拒绝非 loopback、带凭证、
  查询串和 fragment 的 URL。
- ID 全部通过 `^[A-Za-z0-9_-]{1,128}$` 校验（§4.1）。
- `source_code` ≤ 64 KiB，超限在发起 HTTP 前拒绝（§7.1）。
- 日志与错误包不回显 source_code、请求凭证、token 或 DSA 绝对路径
  （`_fetch_error` 只存规范化 error 字段；MCP `_error` 固定结构）。
- 429/502/503/504 标记可重试；400/403/409/422 不自动重试（§7.1）。
- 网络超时不等同提交失败：`DsaUnavailableError` 由 Controller 退避处理，
  创建/启动请求携带稳定 Idempotency-Key，先按幂等键/资源 ID 查询再重提（§4.3）。
- 决策在创建候选/启动任务前本地持久化（不可变 decision，§6.10）。
- Campaign 创建校验 `development_window.end_date < 2022-01-01` 且
  `open_locked_holdout=false`；自动队列永远不携带 locked-holdout 授权（§7.2.5）。
- 测试使用 `pytest`；Mock DSA 来自 DSA 仓库，不可用时测试 skip（不影响默认运行）。

## rollback

- 回滚 = 删除本分支合并即可：新增文件全部位于 `agent/src/research_controller/`
  与 `agent/tests/`（新增文件），不会影响既有模块。
- 既有文件仅改两处：`agent/dsa_lab_mcp_server.py`（追加工具，可安全移除）与
  `agent/tests/test_dsa_lab_mcp_server.py`（期望工具集扩展）。
- 运行时数据：`~/.vibe-trading/research_controller/campaigns.db` 为纯新增 SQLite，
  删除即可清理，不影响既有 `scheduled_research`、session、memory 存储。
- 没有修改 DSA 仓库、`mcp_server.py`（主 MCP）、`api_server.py`（热点文件）、
  认证、调度或发布路径。

## 验收对照（§20 / §18.7 WP4）

- [x] MCP v1 工具可对 Mock 工作，一期工具不破坏（`test_dsa_lab_mcp_server.py` 通过）。
- [x] Campaign 创建/查询/暂停/恢复/取消不产生重复任务。
- [x] 事件去重、幂等、恢复有测试覆盖。
- [x] 修复谱系：execution_failure 下 v1 失败 → Reviewer → decision → v2 提交可演示。
- [x] 报告生成函数对 mock evidence 输出中文报告（9 段结构 + 哈希）。
- [x] 验收命令通过（140 passed）。
- [ ] 真实 DSA 集成、沙箱隔离、357 因子漏斗、launchd 常驻 —— 依赖 M1/M2/WP5/WP7。
