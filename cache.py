import hashlib
import json
import os
from typing import Optional

import redis
from dotenv import load_dotenv

load_dotenv()

CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # seconds

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
    return _client


def _key(question: str) -> str:
    return "rag:" + hashlib.sha256(question.encode()).hexdigest()


def get_cached(question: str) -> Optional[dict]:
    try:
        val = _get_client().get(_key(question))
        return json.loads(val) if val else None
    except redis.RedisError:
        return None


def set_cached(question: str, result: dict) -> None:
    try:
        _get_client().setex(_key(question), CACHE_TTL, json.dumps(result))
    except redis.RedisError:
        pass
