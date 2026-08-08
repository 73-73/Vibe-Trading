#!/usr/bin/env python3
"""Export/check the deterministic VIBE-owned 357-factor manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.research_controller.factors.authority import build_authoritative_manifest  # noqa: E402

DEFAULT_OUTPUT = AGENT_ROOT / "src/research_controller/factors/a_share_native_357.v1.json"


def _render() -> str:
    return json.dumps(build_authoritative_manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"factor manifest drift: {args.output}", file=sys.stderr)
            return 1
        print(f"factor manifest OK: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
