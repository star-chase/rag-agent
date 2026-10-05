"""对话路由：会话管理 + 非流式/流式问答（SSE）。"""

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.deps import get_current_user
from app.db.models import Conversation, User
from app.db.session import get_db
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationOut,
    MessageOut,
)
from app.services import chat_service

router = APIRouter(tags=["conversations"])


@router.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Conversation:
    return await chat_service.create_conversation(db, user, data.knowledge_base_id, data.title)


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Conversation]:
    return await chat_service.list_conversations(db, user.id)


@router.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conv_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = await chat_service.get_conversation(db, conv_id, user.id)
    return await chat_service.list_messages(db, conv)


@router.post("/conversations/{conv_id}/ask", response_model=AskResponse)
async def ask(
    conv_id: int,
    data: AskRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await chat_service.check_rate_limit(user.id, settings.rate_limit_per_minute)
    conv = await chat_service.get_conversation(db, conv_id, user.id)
    return await chat_service.ask(db, user, conv, data.question)


@router.post("/conversations/{conv_id}/ask-stream")
async def ask_stream(
    conv_id: int,
    data: AskRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """SSE 流式问答：事件流为 sources -> delta* -> done。"""
    await chat_service.check_rate_limit(user.id, settings.rate_limit_per_minute)
    conv = await chat_service.get_conversation(db, conv_id, user.id)

    async def event_gen():
        async for event in chat_service.ask_stream(db, user, conv, data.question):
            # 客户端断开时停止生成
            if await request.is_disconnected():
                break
            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/conversations/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conv_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    conv = await chat_service.get_conversation(db, conv_id, user.id)
    await db.delete(conv)
    await db.flush()
