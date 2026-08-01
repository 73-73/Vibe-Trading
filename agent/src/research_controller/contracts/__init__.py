"""research_controller.contracts：由 DSA contract_bundle.v1 生成的副本。

- ``bundle_manifest.json``：DSA bundle manifest，含 bundle_sha256 / source_commit。
- ``schemas/``：research-loop.v1 各契约的 JSON Schema（bundle 副本）。
- ``golden/``：golden fixtures（bundle 副本）。
- ``generated/``：datamodel-code-generator 生成的 Pydantic 客户端模型。

任何 schema 改动都必须重新生成并更新 bundle_sha256，禁止手工编辑 generated/。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

_CONTRACTS_DIR = Path(__file__).resolve().parent


def source_bundle_sha256() -> str:
    """返回 DSA contract bundle 的 bundle_sha256（drift 检测基准）。"""
    return load_bundle_manifest()["bundle_sha256"]


def load_bundle_manifest() -> dict:
    with (_CONTRACTS_DIR / "bundle_manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_copy() -> list[str]:
    """校验本地副本与 bundle manifest 一致；返回问题列表（空列表表示通过）。"""
    problems: list[str] = []
    manifest = load_bundle_manifest()
    schemas_dir = _CONTRACTS_DIR / "schemas"
    for name, expected in manifest.get("schemas", {}).items():
        target = schemas_dir / f"{name}.json"
        if not target.exists():
            problems.append(f"missing schema copy: {name}")
        elif sha256_file(target) != expected:
            problems.append(f"schema copy drift: {name}")
    return problems
