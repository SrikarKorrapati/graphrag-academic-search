"""Answer synthesis via configured LLM and intent-specific system prompts."""

from __future__ import annotations

import logging

from pipeline.llm_backend import complete_chat

logger = logging.getLogger(__name__)

_PAPER_INTENTS = frozenset(
    {
        "find_papers",
        "find_by_topic",
        "find_by_venue",
        "find_by_citation",
    }
)
_AUTHOR_INTENTS = frozenset({"find_authors"})
_OVERVIEW_INTENTS = frozenset({"comparative", "exploratory", "find_by_institution"})


class LLMGenerator:
    """Generates the final natural-language answer grounded in context."""

    def __init__(self) -> None:
        """Generator is stateless; backend reads ``settings`` per call."""

    def _system(self, intent: str) -> str:
        """Pick the system prompt for the classified intent."""
        if intent in _PAPER_INTENTS:
            return (
                "You are an academic paper search assistant. Given context about "
                "research papers, answer the user's query by listing the most "
                "relevant papers. For each paper mention: title, authors, year, "
                "venue, and why it is relevant to the query. Cite sources as "
                "[1], [2], etc."
            )
        if intent in _AUTHOR_INTENTS:
            return (
                "You are an academic search assistant. Based on the context, "
                "identify the most relevant researchers for the user's query. "
                "For each author mention their name, institution, and key papers."
            )
        if intent in _OVERVIEW_INTENTS:
            return (
                "You are a research assistant. Provide a comprehensive overview "
                "based on the academic papers in the context. Group by subtopic "
                "where relevant. Cite sources as [1], [2], etc."
            )
        return (
            "You are an academic paper search assistant. Use the context to answer "
            "the query clearly and cite sources as [1], [2], etc."
        )

    def generate(self, original_query: str, context: str, intent: str) -> str:
        """Call the LLM once with intent-conditioned instructions."""
        try:
            user = f"Context:\n{context}\n\nQuery:\n{original_query}"
            text = complete_chat(self._system(intent), user, json_mode=False)
            return text.strip() or "No answer generated."
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            raise RuntimeError(f"Answer generation failed: {exc}") from exc
