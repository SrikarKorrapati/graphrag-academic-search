#!/usr/bin/env python3
"""Emit only: query + top-5 fused papers per sample run (minimal JSON).

Reads query_logs/sample_query_runs/q*.json.

Usage (from graphrag/):
  python3 scripts/summarize_top5_by_query.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _paper_min(p: dict[str, Any]) -> dict[str, str]:
    meta = p.get("metadata") or {}
    return {
        "paper_id": str(meta.get("paper_id") or p.get("id") or ""),
        "title": str(meta.get("title") or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        default=str(_ROOT / "query_logs" / "sample_query_runs"),
    )
    ap.add_argument(
        "--output",
        default=str(_ROOT / "query_logs" / "sample_query_runs" / "top5_by_query_summary.json"),
    )
    ap.add_argument("--top-k", type=int, default=5, dest="top_k")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    paths = sorted(runs_dir.glob("q*.json"), key=lambda p: p.name)
    paths = [p for p in paths if p.name.startswith("q") and p.stem[1:].isdigit()]

    out: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        qtext = str(data.get("user_query", "")).strip()
        final = data.get("final_output") or {}
        hybrid = data.get("hybrid_fusion") or {}
        papers = final.get("papers_fused") or hybrid.get("papers") or []
        papers = papers[: int(args.top_k)]
        out.append(
            {
                "query": qtext,
                "papers": [_paper_min(p) for p in papers],
            }
        )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {outp} ({len(out)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
