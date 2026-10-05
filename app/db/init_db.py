"""初始化数据库：创建 pgvector 扩展、建表、建向量索引。

用法：
    PYTHONPATH=.deps python3 -m app.db.init_db
"""
import asyncio

from sqlalchemy import text

from app.db.base import Base
from app.db.models import (  # noqa: F401  确保模型注册
    Chunk,
    Conversation,
    Document,
    KnowledgeBase,
    Message,
    User,
)
from app.db.session import AsyncSessionLocal, engine


async def init() -> None:
    async with engine.begin() as conn:
        # pgvector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # 建表
        await conn.run_sync(Base.metadata.create_all)

        # 向量检索索引：HNSW（适合中等规模，召回好）
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"))
        # 关键词检索辅助索引
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_kb_id ON chunks (knowledge_base_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_kb_id ON documents (knowledge_base_id)"))

    async with AsyncSessionLocal() as db:
        await db.execute(text("SELECT 1"))
    print("数据库初始化完成: pgvector 扩展 + 全部表 + HNSW 索引")


if __name__ == "__main__":
    asyncio.run(init())
