"""pytest 配置：使用本机 PostgreSQL 测试库 + fake 模式（不依赖外部 API）。"""
import os

# 必须在导入 app 之前设置
os.environ["TESTING"] = "1"
TEST_DB_HOST = os.environ.get("TEST_DB_HOST", "127.0.0.1")
TEST_DB_PORT = os.environ.get("TEST_DB_PORT", "5433")
TEST_DB_PASSWORD = os.environ.get("TEST_DB_PASSWORD", "")
TEST_DB_USER = os.environ.get("TEST_DB_USER", "rag")
os.environ["DATABASE_URL"] = (
    f"postgresql+asyncpg://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/rag_agent_test"
)
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
os.environ["EMBEDDING_MODE"] = "fake"
os.environ["LLM_MODE"] = "fake"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["JWT_SECRET"] = "test-secret-0123456789abcdef0123456789abcdef"


import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.base import Base
from app.db.session import AsyncSessionLocal, engine

TEST_DB_NAME = "rag_agent_test"


async def _ensure_test_db() -> None:
    """若测试库不存在则创建（连接 postgres 默认库）。"""
    import asyncpg

    conn = await asyncpg.connect(
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD or None,
        host=TEST_DB_HOST,
        port=int(TEST_DB_PORT),
        database="postgres",
    )
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME)
    if not exists:
        await conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    await conn.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    await _ensure_test_db()
    async with engine.begin() as conn:
        from sqlalchemy import text

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict:
    """注册并登录一个测试用户，返回 Bearer 头。"""
    email = "u1@test.com"
    await client.post("/api/v1/auth/register", json={"email": email, "password": "password123", "name": "u1"})
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "password123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
