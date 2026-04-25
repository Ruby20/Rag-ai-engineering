"""
Usage:
    python query.py "What is the attention mechanism?"
"""

import sys

from openai import OpenAI

from cache import get_cached, set_cached
from db import search_chunks
from embedder import embed_query
from expander import expand_query

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM_PROMPT = """\
You are a research assistant. Answer the user's question using ONLY the \
provided context chunks. If the context does not contain enough information, \
say so explicitly. Do not fabricate citations or facts."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] (paper: {c['paper_title']}, section: {c['section_hint']}, "
            f"similarity: {c['similarity']:.3f})\n{c['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def ask(question: str) -> dict:
    cached = get_cached(question)
    if cached:
        print("(cache hit)")
        return cached

    expanded = expand_query(question)
    if expanded != question:
        print(f"(expanded: {expanded})")

    query_vec = embed_query(expanded)
    chunks = search_chunks(query_vec, top_k=5)

    if not chunks:
        return {"answer": "No relevant chunks found in the database.", "sources": []}

    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="llama3.2",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content
    sources = list({c["paper_title"] for c in chunks})
    result = {"answer": answer, "sources": sources}
    set_cached(question, result)
    return result


def _print_result(result: dict) -> None:
    print("\nAnswer:\n", result["answer"])
    print("\nSources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print_result(ask(" ".join(sys.argv[1:])))
    else:
        print("RAG query loop — type 'quit' to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                break
            _print_result(ask(question))
            print()
