"""Turn fused hits into a single context block for the generator LLM."""

from __future__ import annotations

from typing import Any

from models.schemas import FusedResult


class ContextAggregator:
    """Merges overlapping KG/dense rows and formats numbered context."""

    def aggregate(self, fused_results: list[FusedResult]) -> str:
        """Deduplicate by paper/author id, format blocks, cap total length."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for fr in fused_results:
            mid = fr.id
            if mid not in merged:
                order.append(mid)
                merged[mid] = dict(fr.metadata)
            else:
                a, b = merged[mid], fr.metadata
                merged[mid] = {**a, **b}
                for k in ("authors", "topics", "keywords"):
                    if k in a or k in b:
                        xs = []
                        if isinstance(a.get(k), list):
                            xs.extend(a[k])
                        if isinstance(b.get(k), list):
                            xs.extend(b[k])
                        if xs:
                            merged[mid][k] = list(dict.fromkeys(xs))
        blocks: list[str] = []
        for i, mid in enumerate(order, start=1):
            m = merged[mid]
            kind = m.get("kind")
            if kind == "author":
                blocks.append(self._author_block(i, m))
            else:
                blocks.append(self._paper_block(i, m))
        text = "\n\n".join(blocks)
        if len(text) > 6000:
            text = text[:6000] + "\n…[truncated]"
        return text

    def _paper_block(self, n: int, m: dict[str, Any]) -> str:
        """Render one numbered paper block for the LLM context."""
        title = m.get("title", "Unknown title")
        year = m.get("year", "")
        authors = m.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        venue = m.get("venue", "")
        topics = m.get("topics") or m.get("keywords") or []
        if isinstance(topics, str):
            topics = [topics]
        ab = str(m.get("abstract", "")).strip()
        if len(ab) > 200:
            ab = ab[:200] + "…"
        au = ", ".join(str(a) for a in authors if a) or "N/A"
        to = ", ".join(str(t) for t in topics if t) or "N/A"
        lines = [
            f"[{n}] {title} ({year})",
            f"    Authors: {au}",
            f"    Venue: {venue or 'N/A'}",
            f"    Topics: {to}",
        ]
        if ab:
            lines.append(f"    {ab}")
        return "\n".join(lines)

    def _author_block(self, n: int, m: dict[str, Any]) -> str:
        """Render one numbered author block for the LLM context."""
        name = m.get("name", "Unknown")
        inst = m.get("institution", "N/A")
        papers = m.get("papers") or []
        if isinstance(papers, str):
            papers = [papers]
        plist = ", ".join(str(p) for p in papers[:5])
        return (
            f"[{n}] Author: {name}\n"
            f"    Institution: {inst}\n"
            f"    Sample papers: {plist}"
        )
