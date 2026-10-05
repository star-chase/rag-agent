"""知识库 CRUD 路由（需登录，多租户隔离）。"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_owned_knowledge_base
from app.db.models import KnowledgeBase, User
from app.db.session import get_db
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
    KnowledgeBaseWithStats,
)
from app.services import knowledge_base_service

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.get("", response_model=list[KnowledgeBaseWithStats])
async def list_kbs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """注意：返回体含 document_count/ready_count，用 dict 而非直接 ORM 序列化。"""
    return await knowledge_base_service.list_knowledge_bases(db, user.id)


@router.post("", response_model=KnowledgeBaseOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    data: KnowledgeBaseCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_base_service.create_knowledge_base(db, user.id, data)


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
async def get_kb(
    kb: KnowledgeBase = Depends(get_owned_knowledge_base),
):
    return kb


@router.patch("/{kb_id}", response_model=KnowledgeBaseOut)
async def update_kb(
    data: KnowledgeBaseUpdate,
    kb: KnowledgeBase = Depends(get_owned_knowledge_base),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_base_service.update_knowledge_base(db, kb, data)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb: KnowledgeBase = Depends(get_owned_knowledge_base),
    db: AsyncSession = Depends(get_db),
):
    await knowledge_base_service.delete_knowledge_base(db, kb)
