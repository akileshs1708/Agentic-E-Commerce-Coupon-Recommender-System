"""
Redis cache utilities shared across services.
Used for user session caching, rate limiting, and fast coupon lookups.
"""
import os
import json
import redis
from typing import Any, Optional
from shared.logger import get_logger

logger = get_logger(__name__, "cache")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Global Redis client (connection pooled)
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Returns a Redis client, creating one if necessary."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """
    Store a value in Redis with optional TTL (seconds).
    Value is JSON-serialized automatically.
    """
    try:
        r = get_redis()
        serialized = json.dumps(value, default=str)
        r.setex(key, ttl, serialized)
        return True
    except Exception as e:
        logger.warning(f"Cache set failed for key {key}: {e}")
        return False


def cache_get(key: str) -> Optional[Any]:
    """
    Retrieve and deserialize a value from Redis.
    Returns None on cache miss or error.
    """
    try:
        r = get_redis()
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache get failed for key {key}: {e}")
        return None


def cache_delete(key: str) -> bool:
    """Remove a key from Redis."""
    try:
        r = get_redis()
        r.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Cache delete failed for key {key}: {e}")
        return False


def cache_increment(key: str, amount: int = 1, ttl: int = 86400) -> int:
    """Atomically increment a counter in Redis. Returns new value."""
    try:
        r = get_redis()
        value = r.incr(key, amount)
        if ttl > 0:
            r.expire(key, ttl)
        return value
    except Exception as e:
        logger.warning(f"Cache increment failed for key {key}: {e}")
        return 0


def cache_hset(name: str, mapping: dict, ttl: int = 3600) -> bool:
    """Store a hash (dict) in Redis."""
    try:
        r = get_redis()
        r.hset(name, mapping=mapping)
        if ttl > 0:
            r.expire(name, ttl)
        return True
    except Exception as e:
        logger.warning(f"Cache hset failed for {name}: {e}")
        return False


def cache_hgetall(name: str) -> dict:
    """Retrieve all fields of a Redis hash."""
    try:
        r = get_redis()
        return r.hgetall(name) or {}
    except Exception as e:
        logger.warning(f"Cache hgetall failed for {name}: {e}")
        return {}
