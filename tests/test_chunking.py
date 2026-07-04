"""Unit tests for semantic chunking — no model downloads required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import Chunk, chunk_document
from rag.config import MAX_CHUNK_CHARS


SAMPLE_MD = """# Title

Intro paragraph before any section.

## Section One

First paragraph of section one.

Second paragraph of section one.

## Section Two

Content of section two.
"""


def test_splits_on_headings():
    chunks = chunk_document(SAMPLE_MD, source="sample.md")
    sections = {c.section for c in chunks}
    assert "Section One" in sections
    assert "Section Two" in sections


def test_chunk_carries_metadata():
    chunks = chunk_document(SAMPLE_MD, source="sample.md")
    assert all(c.source == "sample.md" for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_section_title_prefixed_into_text():
    chunks = chunk_document(SAMPLE_MD, source="sample.md")
    sec_one = next(c for c in chunks if c.section == "Section One")
    assert sec_one.text.startswith("Section One")


def test_long_sections_respect_size_cap():
    long_body = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    md = f"# Big\n\n## Huge Section\n\n{long_body}"
    chunks = chunk_document(md, source="big.md")
    assert len(chunks) > 1
    # Allow small slack for the prefixed section title
    assert all(len(c.text) <= MAX_CHUNK_CHARS + 100 for c in chunks)


def test_document_without_headings_still_chunks():
    chunks = chunk_document("Just a plain paragraph with no headings.", source="plain.md")
    assert len(chunks) == 1
    assert chunks[0].section == ""
