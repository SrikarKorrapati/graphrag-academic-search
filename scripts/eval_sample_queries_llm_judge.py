#!/usr/bin/env python3
"""LLM-as-judge: score each sample_query_runs/*.json for relevance to the user query (1–10).

Reads final_output.answer (+ light context). Writes a single JSON report with per-query
scores, rationale, and aggregate ranking.

Usage (from graphrag/):
  python3 scripts/eval_sample_queries_llm_judge.py
  python3 scripts/eval_sample_queries_llm_judge.py --runs-dir query_logs/sample_query_runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.llm_backend import complete_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_judge_eval")

_JUDGE_SYSTEM = """You are an expert evaluator for an academic search assistant (GraphRAG).
Your task: judge how relevant the ASSISTANT OUTPUT is to the USER QUERY.

Scoring rubric for relevance_score (integer 1–10, 10 = most relevant):
- 9–10: Output directly addresses the query with on-topic papers/findings; main concepts match; useful specifics (titles, methods, domains) align with the query.
- 7–8: Mostly on-topic with minor gaps, tangents, or missing one sub-aspect of a multi-part query.
- 5–6: Partially relevant (right general area) but misses key constraints (e.g., wrong modality, wrong task, too generic).
- 3–4: Weak relevance: vague, mostly off-topic, or cites material that does not match the query intent.
- 1–2: Irrelevant, empty, error-like, or contradicts the query.

Do NOT score based on writing style alone. Focus on topical match between query and what the output claims or summarizes.
If the output is empty or says it could not answer, assign a low score (1–3) unless the query truly cannot be addressed.

Respond with ONE JSON object only (no markdown fences). Keys:
- relevance_score: integer 1–10
- rationale: string, 2–5 sentences explaining the score
- strengths: array of 0–4 short strings (what matched the query well)
- gaps: array of 0–4 short strings (what was missing or off)
"""


def _safe_int(value: Any, default: int = 0) -> int:
    """Parse a numeric value into int; return default on malformed input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strip_json_fences(raw: str) -> str:
    """Remove optional markdown code fences around a JSON payload."""
    s = raw.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse judge response JSON, including tolerant extraction from mixed text."""
    s = _strip_json_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Some providers occasionally prepend/explain around JSON; recover first object.
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
        raise


def _clamp_score(v: Any) -> int:
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return 0
    return max(1, min(10, n))


def _build_user_block(
    user_query: str,
    answer: str,
    *,
    intent: str | None,
    pipeline_error: str | None,
    n_kg: int,
    n_dense: int,
    max_answer_chars: int,
) -> str:
    """Build structured judge input with lightweight retrieval diagnostics."""
    ans = (answer or "").strip()
    truncated = False
    if len(ans) > max_answer_chars:
        ans = ans[:max_answer_chars] + "\n…[truncated for evaluation context]"
        truncated = True
    parts = [
        f"USER_QUERY:\n{user_query}\n",
        f"PARSED_INTENT (hint): {intent or 'unknown'}\n",
        f"PIPELINE_ERROR (if any): {pipeline_error or 'none'}\n",
        f"RETRIEVAL_COUNTS: kg_hits={n_kg}, dense_hits={n_dense}\n",
        f"ASSISTANT_OUTPUT:\n{ans if ans else '[empty]'}\n",
    ]
    if truncated:
        parts.append("(Assistant output was truncated only for this evaluation prompt.)\n")
    return "\n".join(parts)


async def _judge_one(
    user_query: str,
    answer: str,
    *,
    intent: str | None,
    pipeline_error: str | None,
    n_kg: int,
    n_dense: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    """Call the LLM judge once and normalize output to a strict schema."""
    user = _build_user_block(
        user_query,
        answer,
        intent=intent,
        pipeline_error=pipeline_error,
        n_kg=n_kg,
        n_dense=n_dense,
        max_answer_chars=max_answer_chars,
    )
    raw = await asyncio.to_thread(
        complete_chat,
        _JUDGE_SYSTEM,
        user,
        json_mode=True,
    )
    data = _parse_judge_json(raw)
    score = _clamp_score(data.get("relevance_score"))
    if score == 0:
        score = 1
    return {
        "relevance_score": score,
        "rationale": str(data.get("rationale", "")).strip(),
        "strengths": [str(x) for x in (data.get("strengths") or []) if str(x).strip()][:4],
        "gaps": [str(x) for x in (data.get("gaps") or []) if str(x).strip()][:4],
        "raw_judge_response": raw[:4000],
    }


async def _async_main(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        logger.error("Runs directory not found: %s", runs_dir)
        return 1

    paths = sorted(runs_dir.glob("q*.json"), key=lambda p: p.name)
    if not paths:
        logger.error("No q*.json files in %s", runs_dir)
        return 1

    results: list[dict[str, Any]] = []
    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        qid = rec.get("query_id") or path.stem
        user_query = str(rec.get("user_query", "")).strip()
        final_out = rec.get("final_output") or {}
        answer = str(final_out.get("answer", ""))
        intent = final_out.get("intent") or (rec.get("parsed_query") or {}).get("intent")
        pipeline_error = rec.get("pipeline_error")
        kg = rec.get("kg_traversal") or {}
        dense = rec.get("dense_retrieval") or {}
        n_kg = _safe_int(kg.get("n_hits"), default=0)
        n_dense = _safe_int(dense.get("n_hits"), default=0)

        logger.info("Judging %s …", qid)
        t0 = time.perf_counter()
        try:
            judge = await _judge_one(
                user_query,
                answer,
                intent=str(intent) if intent else None,
                pipeline_error=str(pipeline_error) if pipeline_error else None,
                n_kg=n_kg,
                n_dense=n_dense,
                max_answer_chars=max(1, _safe_int(args.max_answer_chars, default=12000)),
            )
            judge_error = None
        except Exception as exc:
            logger.exception("Judge failed for %s", qid)
            judge = {
                "relevance_score": 0,
                "rationale": "",
                "strengths": [],
                "gaps": [],
                "raw_judge_response": "",
            }
            judge_error = str(exc)

        ms = round((time.perf_counter() - t0) * 1000, 2)
        results.append(
            {
                "query_id": qid,
                "source_file": path.name,
                "user_query": user_query,
                "latency_ms": ms,
                "judge_error": judge_error,
                **judge,
            }
        )
        if args.sleep_seconds > 0:
            await asyncio.sleep(args.sleep_seconds)

    # Aggregate only successful judge calls with valid 1-10 scores.
    scored = [r for r in results if not r.get("judge_error") and int(r.get("relevance_score") or 0) >= 1]
    scores = [int(r["relevance_score"]) for r in scored]
    order = sorted(
        range(len(results)),
        key=lambda i: (
            -(int(results[i].get("relevance_score") or 0)),
            str(results[i].get("query_id") or ""),
        ),
    )
    for rank, idx in enumerate(order, start=1):
        results[idx]["relevance_rank"] = rank

    ranking_rows = [
        {
            "rank": results[i]["relevance_rank"],
            "query_id": results[i].get("query_id"),
            "relevance_score": results[i].get("relevance_score"),
            "user_query": results[i].get("user_query"),
        }
        for i in order
    ]

    report = {
        "schema_version": 1,
        "description": "LLM-as-judge relevance (1–10) of final_output.answer vs user_query",
        "judge_system_prompt_excerpt": _JUDGE_SYSTEM[:400] + "…",
        "aggregate": {
            "n_judged": len(results),
            "n_ok": len(scored),
            "mean_relevance": round(sum(scores) / len(scores), 3) if scores else None,
            "median_relevance": float(median(scores)) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
        },
        "ranking": ranking_rows,
        "per_query": sorted(results, key=lambda x: str(x.get("query_id"))),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM judge relevance 1–10 for sample query runs")
    ap.add_argument(
        "--runs-dir",
        default=str(_ROOT / "query_logs" / "sample_query_runs"),
        help="Directory containing q01.json … q20.json",
    )
    ap.add_argument(
        "--output",
        default=str(_ROOT / "query_logs" / "sample_query_runs" / "llm_eval_relevance.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--max-answer-chars",
        type=int,
        default=12000,
        help="Max characters of assistant answer sent to the judge (truncation)",
    )
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.5,
        help="Pause between judge calls (rate limits). 0 disables.",
    )
    args = ap.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
