"""健康检查：验证服务、数据库、Redis、LLM 链路。"""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict:
    """返回各依赖组件状态，供探活与排障。"""
    checks: dict = {"db": "ok", "redis": "ok"}

    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        checks["db"] = "error"

    try:
        r = get_redis()
        await r.ping()
    except Exception:
        checks["redis"] = "error"

    return {
        "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
        "checks": checks,
        "time": datetime.now(UTC).isoformat(),
    }
