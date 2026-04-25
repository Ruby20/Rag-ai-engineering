"""
Usage:
    python eval.py
"""

from dotenv import load_dotenv
load_dotenv()

from embedder import embed_query
from expander import expand_query
from db import search_chunks
from query import ask

FAMA_PAPERS = [
    "Fama-French_JFE93",
    "The Cross-Section of Expected Stock Returns",
    "The Journal of Finance - March 1996 - FAMA - Multifactor Explanations of Asset Pricing Anomalies",
]

EVALS = [
    # --- NIPS attention paper ---
    {
        "question": "What is the attention mechanism?",
        "must_contain": ["quer", "key", "value"],   # "quer" matches query/queries
        "expected_sources": ["NIPS-2017-attention-is-all-you-need-Paper"],
        "min_similarity": 0.65,
    },
    {
        "question": "What is multi-head attention?",
        "must_contain": ["head", "parallel"],
        "expected_sources": ["NIPS-2017-attention-is-all-you-need-Paper"],
        "min_similarity": 0.65,
    },
    {
        "question": "Why does the Transformer model avoid recurrent neural networks?",
        "must_contain": ["parallel", "attention"],
        "expected_sources": ["NIPS-2017-attention-is-all-you-need-Paper"],
        "min_similarity": 0.65,
    },

    # --- Fama-French finance papers ---
    {
        "question": "Why does HML exist?",
        "must_contain": ["distress", "human", "capital"],   # split phrase into tokens
        "expected_sources": FAMA_PAPERS,
        "min_similarity": 0.55,
    },
    {
        "question": "What is the SMB factor?",
        "must_contain": ["small", "size"],
        "expected_sources": FAMA_PAPERS,
        "min_similarity": 0.60,
    },
    {
        "question": "What three factors explain stock returns in the Fama-French model?",
        "must_contain": ["market", "size", "book"],
        "expected_sources": FAMA_PAPERS,
        "min_similarity": 0.65,
    },
    {
        "question": "What is the relationship between book-to-market ratio and stock returns?",
        "must_contain": ["book", "market", "return"],
        "expected_sources": FAMA_PAPERS,
        "min_similarity": 0.65,
    },

    # --- Known limitation: cross-domain should NOT confuse papers ---
    {
        "question": "What is the transformer architecture?",
        "must_contain": ["encod", "decod", "attention"],   # "encod" matches encoder/encoding
        "expected_sources": ["NIPS-2017-attention-is-all-you-need-Paper"],
        "min_similarity": 0.70,
    },
]


def run_eval(ev: dict) -> dict:
    question = ev["question"]

    result = ask(question, skip_cache=True)
    answer = result["answer"].lower()
    sources = result["sources"]

    expanded = expand_query(question)
    vec = embed_query(expanded)
    chunks = search_chunks(vec, top_k=5)
    top_similarity = chunks[0]["similarity"] if chunks else 0.0

    keyword_results = {kw: kw.lower() in answer for kw in ev["must_contain"]}
    keywords_pass = all(keyword_results.values())
    source_pass = any(s in sources for s in ev["expected_sources"])
    similarity_pass = top_similarity >= ev["min_similarity"]

    passed = keywords_pass and source_pass and similarity_pass

    return {
        "question": question,
        "passed": passed,
        "keywords": keyword_results,
        "keywords_pass": keywords_pass,
        "source_pass": source_pass,
        "expected_sources": ev["expected_sources"],
        "actual_sources": sources,
        "similarity_pass": similarity_pass,
        "top_similarity": top_similarity,
        "min_similarity": ev["min_similarity"],
    }


def print_result(r: dict):
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{status}] {r['question']}")

    kw_parts = " ".join(
        f"✓ {k}" if v else f"✗ {k}" for k, v in r["keywords"].items()
    )
    print(f"  keywords:   {kw_parts}")

    src_icon = "✓" if r["source_pass"] else "✗"
    print(f"  source:     {src_icon} got={r['actual_sources']}")
    if not r["source_pass"]:
        print(f"              expected one of={r['expected_sources']}")

    sim_icon = "✓" if r["similarity_pass"] else "✗"
    print(f"  similarity: {sim_icon} {r['top_similarity']:.3f} (min={r['min_similarity']})")


if __name__ == "__main__":
    print(f"Running {len(EVALS)} evals...\n{'─' * 50}")
    results = [run_eval(ev) for ev in EVALS]
    for r in results:
        print_result(r)

    passed = sum(r["passed"] for r in results)
    print(f"\n{'─' * 50}")
    print(f"Results: {passed}/{len(results)} passed")
