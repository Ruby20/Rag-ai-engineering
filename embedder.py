from __future__ import annotations

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks. No prefix — BGE convention for passages."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query. BGE requires a prefix for query-side encoding."""
    model = _get_model()
    prefixed = _BGE_QUERY_PREFIX + text
    return model.encode(prefixed, normalize_embeddings=True).tolist()
