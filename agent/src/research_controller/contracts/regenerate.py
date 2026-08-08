"""从 DSA contract_bundle 的 JSON Schema 重新生成 Pydantic 客户端模型。

用法（Vibe 仓库根目录）：
    .venv/bin/python agent/src/research_controller/contracts/regenerate.py

生成的代码落在 ``generated/``，由 datamodel-code-generator 产出；任何手工修改
都会在下次重新生成时丢失。bundle 更新后必须重跑本脚本并更新 bundle_manifest。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_CONTRACTS_DIR = Path(__file__).resolve().parent
_SCHEMAS_DIR = _CONTRACTS_DIR / "schemas"
_GENERATED_DIR = _CONTRACTS_DIR / "generated"


def _slug(filename: str) -> str:
    return filename.removesuffix(".json").replace(".", "_").replace("-", "_")


def main() -> int:
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    schema_files = sorted(_SCHEMAS_DIR.glob("*.json"))
    failures: list[str] = []
    for schema in schema_files:
        module = _slug(schema.name)
        output = _GENERATED_DIR / f"{module}.py"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "datamodel_code_generator",
                "--input",
                str(schema),
                "--output",
                str(output),
                "--input-file-type",
                "jsonschema",
                "--base-class",
                "pydantic.BaseModel",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failures.append(f"{schema.name}: {proc.stderr.strip()[-500:]}")
        else:
            print(f"generated {output.name}")
    ( _GENERATED_DIR / "__init__.py").write_text(
        "# 自动生成，禁止手工编辑；运行 regenerate.py 重新生成。\n", encoding="utf-8"
    )
    if failures:
        print("FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"generated {len(schema_files)} modules into {_GENERATED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
