#!/usr/bin/env python3
"""Pretty-print ``query_logs/pipeline_trace.jsonl`` (one JSON object per line)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "query_logs" / "pipeline_trace.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=_default_path(),
        help="Path to pipeline_trace.jsonl (default: graphrag/query_logs/pipeline_trace.jsonl)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write formatted JSON here; default: print to stdout",
    )
    ap.add_argument(
        "--array",
        action="store_true",
        help="Emit a single JSON array of all rows (default: one pretty object after another)",
    )
    args = ap.parse_args()

    path: Path = args.path
    if not path.is_file():
        print(f"No file: {path}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"Line {i}: invalid JSON: {exc}", file=sys.stderr)
            return 1

    if args.array:
        text = json.dumps(rows, ensure_ascii=False, indent=2)
    else:
        chunks = [json.dumps(r, ensure_ascii=False, indent=2) for r in rows]
        text = "\n\n".join(chunks)

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
