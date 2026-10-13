#!/usr/bin/env python3
"""Run standard query eval cases (JSON fixtures) against a live POST /query API.

Usage:
  cd graphrag && python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
  # other shell:
  cd graphrag && python3 scripts/eval_query_cases.py --base-url http://127.0.0.1:8000

Fixture format: tests/fixtures/query_eval_cases.json
  cases[].input.query          — user string
  cases[].expect               — all keys optional; missing keys are not checked
    intent_one_of: [str, ...] — parsed_query.intent must be one of these
    min_keywords: int         — len(parsed_query.keywords) >=
    min_papers: int           — len(papers) >=
    answer_contains_any: [str] — at least one substring (case-insensitive) in answer
    latency_ms_max: float     — latency_ms must be <= this
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load eval fixture and validate top-level shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("fixture must contain a 'cases' array")
    return cases


def _post_query(base: str, query: str, timeout_s: float) -> dict[str, Any]:
    """Send POST /query and decode JSON response."""
    url = base.rstrip("/") + "/query"
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check(resp: dict[str, Any], exp: dict[str, Any]) -> list[str]:
    """Validate response against optional expectations from the fixture."""
    errs: list[str] = []
    if "detail" in resp:
        return [f"HTTP error body: {resp.get('detail')}"]

    pq = resp.get("parsed_query") or {}
    intent = str(pq.get("intent", ""))
    kws = pq.get("keywords") or []
    papers = resp.get("papers") or []
    answer = str(resp.get("answer", ""))
    lat = float(resp.get("latency_ms") or 0.0)

    if "intent_one_of" in exp:
        allowed = set(exp["intent_one_of"])
        if intent not in allowed:
            errs.append(f"intent={intent!r} not in {sorted(allowed)}")

    if "min_keywords" in exp:
        need = int(exp["min_keywords"])
        if len(kws) < need:
            errs.append(f"keywords count {len(kws)} < min_keywords {need}")

    if "min_papers" in exp:
        need = int(exp["min_papers"])
        if len(papers) < need:
            errs.append(f"papers count {len(papers)} < min_papers {need}")

    subs = exp.get("answer_contains_any")
    if subs:
        low = answer.lower()
        # Case-insensitive substring checks make fixtures resilient to casing changes.
        if not any(str(s).lower() in low for s in subs):
            errs.append(f"answer missing all of answer_contains_any={subs!r}")

    if "latency_ms_max" in exp and lat > float(exp["latency_ms_max"]):
        errs.append(f"latency_ms={lat} > latency_ms_max={exp['latency_ms_max']}")

    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "query_eval_cases.json",
        help="Path to query_eval_cases.json",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="GraphRAG API base URL")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds)")
    ap.add_argument("--case-id", default="", help="Run only this case id (substring match ok)")
    args = ap.parse_args()

    try:
        cases = _load_cases(args.fixture)
    except Exception as exc:
        print(f"ERROR loading fixture: {exc}", file=sys.stderr)
        return 1

    filt = (args.case_id or "").strip().lower()
    ran = 0
    failed = 0
    for case in cases:
        cid = str(case.get("id", "<no-id>"))
        if filt and filt not in cid.lower():
            continue
        inp = case.get("input") or {}
        query = str(inp.get("query", "")).strip()
        exp = case.get("expect") or {}
        if not query:
            print(f"SKIP {cid}: empty input.query")
            continue
        ran += 1
        print(f"RUN {cid} …", flush=True)
        try:
            resp = _post_query(args.base_url, query, args.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            print(f"FAIL {cid}: HTTP {exc.code} {body}", file=sys.stderr)
            failed += 1
            continue
        except Exception as exc:
            print(f"FAIL {cid}: {exc}", file=sys.stderr)
            failed += 1
            continue

        errs = _check(resp, exp)
        if errs:
            failed += 1
            print(f"FAIL {cid}:", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
        else:
            n = len(resp.get("papers") or [])
            lat = resp.get("latency_ms")
            print(f"OK   {cid}  papers={n}  latency_ms={lat}")

    if ran == 0:
        print("No cases ran (check --case-id or fixture).", file=sys.stderr)
        return 1
    print(f"Summary: {ran - failed}/{ran} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
