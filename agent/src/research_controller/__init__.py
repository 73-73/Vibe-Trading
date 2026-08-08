"""Vibe Research Controller（M4）。

面向用户的 Research Campaign 入口；通过 dsa_lab MCP bridge / 受限 HTTP client
驱动 DSA research-loop.v1。契约模型与 golden fixtures 由 DSA contract_bundle
生成，禁止手工维护第二套 Schema（技术规格 §18.4）。
"""
