# Naive RAG Pipeline

A from-scratch RAG (Retrieval-Augmented Generation) pipeline in Python. No LangChain, no LlamaIndex — raw psycopg2, sentence-transformers, and OpenAI-compatible API only.

Built to understand where naive RAG breaks before reaching for abstractions.

## Stack

| Component | Tool |
|---|---|
| Vector store | pgvector (Postgres) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (local) |
| LLM | Ollama (`llama3.2`, local) |
| Cache | Redis |
| PDF extraction | pypdf |
| Infrastructure | Docker Compose |

## Project Structure

```
db.py             — connection, schema init, insert_chunks, search_chunks
embedder.py       — BGE model singleton, embed_documents, embed_query
ingest.py         — extract text, chunk, embed, store in pgvector
query.py          — embed question, retrieve top-5, generate answer, interactive loop
expander.py       — LLM-based query expansion to bridge terminology gaps
cache.py          — Redis query cache (SHA256 key, 1hr TTL)
docker-compose.yml — Postgres (pgvector) + Redis
resources/        — ingested PDFs
```

## Schema

```sql
chunks: id, paper_title, content, embedding vector(384),
        chunk_index, section_hint (early/middle/late), created_at

Index: ivfflat cosine on embedding
Index: btree on paper_title
```

## Setup

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ingest a paper
python ingest.py resources/your_paper.pdf

# Query — interactive loop
python query.py

# Query — single question
python query.py "What is the attention mechanism?"
```

## Chunking Strategy

Fixed-size with overlap: 800 chars per chunk, 150 char overlap, 200 char minimum.
Section hints assigned by relative position: first 15% = early, 15-75% = middle, rest = late.

## Query Pipeline

1. Expand query via LLM (bridges acronyms and terminology gaps)
2. Check Redis cache — return immediately on hit
3. Embed expanded query locally via BGE
4. Cosine search top-5 chunks in pgvector
5. Build prompt with retrieved context
6. Generate answer via Ollama at temperature 0
7. Cache result, return answer + sources

## Known Limitations (intentional)

1. **Multi-hop retrieval fails** — cross-paper questions return chunks from one paper only
2. **No conflict resolution** — contradicting facts both get stored and retrieved
3. **No entity/relationship modeling** — graph traversal queries fail entirely
4. **Fixed-size chunking** — splits on character count, not semantic boundaries

## Roadmap

- [x] Core pipeline (ingest → embed → retrieve → generate)
- [x] Redis query cache
- [x] Interactive query loop
- [x] Query expansion (LLM-based terminology bridging)
- [ ] Eval harness (keyword match + retrieval scoring)
- [ ] Deliberately trigger and document known limitations
