"""On-disk query/answer history (JSONL)."""

from .writer import append_query_log, build_log_record

__all__ = ["append_query_log", "build_log_record"]
