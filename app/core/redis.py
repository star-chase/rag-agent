"""Redis 异步客户端。"""
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None


def get_redis() -> Redis:
    """返回共享的异步 Redis 客户端（惰性初始化）。"""
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(settings.redis_url, decode_responses=True)
    return Redis(connection_pool=_pool)


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
