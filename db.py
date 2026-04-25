import os
import psycopg2
from psycopg2.extras import execute_values

DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_pass@localhost:5432/rag_db",
)


def get_conn():
    return psycopg2.connect(DSN)


def init_schema():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id            SERIAL PRIMARY KEY,
                paper_title   TEXT        NOT NULL,
                content       TEXT        NOT NULL,
                embedding     vector(384) NOT NULL,
                chunk_index   INTEGER     NOT NULL,
                section_hint  TEXT        NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        # ivfflat index for approximate cosine search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_paper_title_idx
            ON chunks (paper_title)
        """)


def insert_chunks(rows: list[dict]):
    """rows: list of {paper_title, content, embedding, chunk_index, section_hint}"""
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO chunks (paper_title, content, embedding, chunk_index, section_hint)
            VALUES %s
            """,
            [
                (
                    r["paper_title"],
                    r["content"],
                    r["embedding"],
                    r["chunk_index"],
                    r["section_hint"],
                )
                for r in rows
            ],
        )


def search_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Returns top_k chunks ordered by cosine similarity (highest first)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT paper_title, content, section_hint,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def paper_already_ingested(paper_title: str) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM chunks WHERE paper_title = %s LIMIT 1",
            (paper_title,),
        )
        return cur.fetchone() is not None
