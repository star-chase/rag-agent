"""知识库业务逻辑。"""
from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, KnowledgeBase
from app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseUpdate


async def list_knowledge_bases(db: AsyncSession, user_id: int) -> list[dict]:
    """返回知识库列表，附带文档数与就绪数。"""
    rows = await db.execute(
        select(
            KnowledgeBase,
            func.count(Document.id).label("document_count"),
            func.count(Document.id).filter(Document.status == "ready").label("ready_count"),
        )
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .where(KnowledgeBase.owner_id == user_id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.id.desc())
    )
    return [
        {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "owner_id": kb.owner_id,
            "created_at": kb.created_at,
            "updated_at": kb.updated_at,
            "document_count": doc_count or 0,
            "ready_count": ready_count or 0,
        }
        for kb, doc_count, ready_count in rows.all()
    ]


async def get_knowledge_base(db: AsyncSession, kb_id: int, user_id: int) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == user_id)
    )
    if kb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    return kb


async def create_knowledge_base(
    db: AsyncSession, user_id: int, data: KnowledgeBaseCreate
) -> KnowledgeBase:
    kb = KnowledgeBase(name=data.name, description=data.description, owner_id=user_id)
    db.add(kb)
    await db.flush()
    return kb


async def update_knowledge_base(
    db: AsyncSession, kb: KnowledgeBase, data: KnowledgeBaseUpdate
) -> KnowledgeBase:
    if data.name is not None:
        kb.name = data.name
    if data.description is not None:
        kb.description = data.description
    await db.flush()
    return kb


async def delete_knowledge_base(db: AsyncSession, kb: KnowledgeBase) -> None:
    await db.delete(kb)
    await db.flush()
