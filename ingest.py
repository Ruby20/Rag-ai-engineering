"""
Usage:
    python ingest.py path/to/paper.pdf
    python ingest.py path/to/paper.txt
"""

import re
import sys
from pathlib import Path

import pypdf

from db import init_schema, insert_chunks, paper_already_ingested
from embedder import embed_documents

CHUNK_SIZE = 800
OVERLAP = 150
MIN_CHUNK = 200

EARLY_THRESHOLD = 0.15
LATE_THRESHOLD = 0.75


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def strip_references(text: str) -> str:
    # Drop everything from a "References" heading onward
    match = re.search(r"\n\s*References\s*\n", text, re.IGNORECASE)
    if match:
        return text[: match.start()]
    return text


def make_chunks(text: str) -> list[str]:
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append(chunk)
        start += step
    return chunks


def section_hint(chunk_index: int, total_chunks: int) -> str:
    if total_chunks == 0:
        return "early"
    ratio = chunk_index / total_chunks
    if ratio < EARLY_THRESHOLD:
        return "early"
    if ratio < LATE_THRESHOLD:
        return "middle"
    return "late"


def ingest(path: Path):
    paper_title = path.stem  # filename without extension as the paper key

    if paper_already_ingested(paper_title):
        print(f"Skipping '{paper_title}' — already ingested.")
        return

    print(f"Ingesting '{paper_title}' ...")
    raw = extract_text(path)
    raw = strip_references(raw)
    chunks = make_chunks(raw)
    total = len(chunks)
    print(f"  {total} chunks created")

    embeddings = embed_documents(chunks)

    rows = [
        {
            "paper_title": paper_title,
            "content": chunk,
            "embedding": embeddings[i],
            "chunk_index": i,
            "section_hint": section_hint(i, total),
        }
        for i, chunk in enumerate(chunks)
    ]

    insert_chunks(rows)
    print(f"  Stored {len(rows)} chunks in pgvector.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file.pdf|file.txt>")
        sys.exit(1)

    init_schema()
    for arg in sys.argv[1:]:
        ingest(Path(arg))
