"""Semantic chunking: split markdown documents along structural boundaries.

Strategy (in priority order):
1. Split on markdown headings  -> each section is one candidate chunk.
2. If a section exceeds MAX_CHUNK_CHARS, split it on paragraph boundaries,
   greedily packing paragraphs until the size cap, with a small overlap so
   sentences near a boundary are not lost.

Every chunk keeps metadata (source file + section title) so answers can cite
their origin.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CHUNK_OVERLAP_CHARS, MAX_CHUNK_CHARS


@dataclass
class Chunk:
    text: str
    source: str          # e.g. "03_hybrid_search.md"
    section: str         # e.g. "Fusion methods"
    chunk_id: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown document into (section_title, section_text) pairs."""
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return [("", markdown.strip())]

    sections: list[tuple[str, str]] = []
    # Text before the first heading (rare, but keep it)
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _split_long_text(text: str) -> list[str]:
    """Greedily pack paragraphs into chunks under MAX_CHUNK_CHARS, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Overlap: carry the tail of the previous chunk into the next one
            tail = current[-CHUNK_OVERLAP_CHARS:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
            # Pathological case: a single paragraph longer than the cap
            while len(current) > MAX_CHUNK_CHARS:
                chunks.append(current[:MAX_CHUNK_CHARS])
                current = current[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS:]
    if current:
        chunks.append(current)
    return chunks


def chunk_document(markdown: str, source: str) -> list[Chunk]:
    """Chunk one markdown document into semantically coherent pieces."""
    chunks: list[Chunk] = []
    for title, body in _split_sections(markdown):
        # Prefix the section title so the embedding carries the topic signal
        for piece in _split_long_text(body):
            text = f"{title}\n{piece}" if title else piece
            chunks.append(Chunk(text=text, source=source, section=title))
    return chunks


def chunk_directory(docs_dir: Path) -> list[Chunk]:
    """Chunk every .md file in a directory; assigns global chunk ids."""
    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name))
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = i
    return all_chunks
