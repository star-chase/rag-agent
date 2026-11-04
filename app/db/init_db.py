"""初始化数据库：创建 pgvector 扩展、建表、建向量索引。

用法：
    PYTHONPATH=.deps python3 -m app.db.init_db
"""
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
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


def embedding_storage(dim: int | None = None) -> tuple[str, str]:
    """返回列类型和 HNSW 操作符类。超过 2000 维时 vector 索引建不了。"""
    size = settings.embedding_dim if dim is None else dim
    if size > 2000:
        return f"halfvec({size})", "halfvec_cosine_ops"
    return f"vector({size})", "vector_cosine_ops"


async def ensure_embedding_column(conn: AsyncConnection) -> None:
    """把已有的 chunks.embedding 改成当前维度。旧向量宽度对不上，只能清空后重嵌。"""
    type_sql, ops = embedding_storage()
    current = await conn.scalar(
        text(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute AS a
            JOIN pg_class AS c ON c.oid = a.attrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relname = 'chunks'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
    )
    if current is None:
        return
    normalized = current.replace(" ", "")
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    if normalized != type_sql:
        await conn.execute(text("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw"))
        await conn.execute(text(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE {type_sql} USING NULL"))
    await conn.execute(text("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw"))
    await conn.execute(
        text(f"CREATE INDEX idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding {ops})")
    )


async def init() -> None:
    async with engine.begin() as conn:
        # pgvector 扩展；pg_jieba 提供 jiebacfg，关键词路对原文做中文分词。
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_jieba"))
        # 建表
        await conn.run_sync(Base.metadata.create_all)
        await ensure_embedding_column(conn)

        # 向量检索索引由 ensure_embedding_column 重建。这里不再重复创建。
        # 关键词检索辅助索引
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_kb_id ON chunks (knowledge_base_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_kb_id ON documents (knowledge_base_id)"))
        await conn.execute(text("DROP INDEX IF EXISTS idx_chunks_token_tsv"))
        await conn.execute(text("ALTER TABLE chunks DROP COLUMN IF EXISTS token_text"))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_jieba "
            "ON chunks USING gin (to_tsvector('jiebacfg', content))"
        ))

    async with AsyncSessionLocal() as db:
        await db.execute(text("SELECT 1"))
    print("数据库初始化完成: pgvector + pg_jieba + 全部表 + HNSW/GIN 索引")


if __name__ == "__main__":
    asyncio.run(init())
