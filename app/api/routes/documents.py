"""文档路由：上传 / 列表 / 详情 / 删除 / 状态。"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.deps import get_owned_document, get_owned_knowledge_base
from app.core.storage import get_storage, guess_file_type
from app.db.models import Chunk, Document, KnowledgeBase
from app.db.session import get_db
from app.schemas.document import ChunkOut, DocumentOut, DocumentStatusOut
from app.services.ingest_service import make_storage_key, process_document
from app.workers.ingest_tasks import process_document_task

router = APIRouter(tags=["documents"])

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    kb: KnowledgeBase = Depends(get_owned_knowledge_base),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """上传文档：先存对象存储，建记录后投递异步摄入任务。"""
    filename = file.filename or "unnamed"
    try:
        file_type = guess_file_type(filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from None

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空")
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件超过 {MAX_UPLOAD_SIZE // 1024 // 1024}MB 限制",
        )

    storage_key = make_storage_key(kb.id, filename)
    await get_storage().upload(storage_key, data, content_type=file.content_type)

    doc = Document(
        knowledge_base_id=kb.id,
        filename=filename,
        storage_key=storage_key,
        file_type=file_type,
        file_size=len(data),
    )
    db.add(doc)
    await db.flush()
    await db.commit()  # 先提交，让异步任务/新会话能看到该文档

    # 投递异步任务；本地调试（always_eager）时直接 await，避免事件循环嵌套
    if settings.celery_task_always_eager:
        await process_document(doc.id)
    else:
        process_document_task.delay(doc.id)

    return doc


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentOut])
async def list_documents(
    kb_id: int,
    kb: KnowledgeBase = Depends(get_owned_knowledge_base),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    rows = await db.scalars(
        select(Document)
        .where(Document.knowledge_base_id == kb.id)
        .order_by(Document.id.desc())
    )
    return list(rows)


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(doc: Document = Depends(get_owned_document)) -> Document:
    return doc


@router.get("/documents/{doc_id}/chunks", response_model=list[ChunkOut])
async def list_document_chunks(
    doc: Document = Depends(get_owned_document),
    db: AsyncSession = Depends(get_db),
) -> list[Chunk]:
    """查看文档解析出的分块内容（演示台「查看内容」/排障用）。"""
    rows = await db.scalars(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.seq))
    return list(rows)


@router.get("/documents/{doc_id}/status", response_model=DocumentStatusOut)
async def get_document_status(doc: Document = Depends(get_owned_document)) -> DocumentStatusOut:
    return DocumentStatusOut(
        id=doc.id,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=doc.chunk_count,
    )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc: Document = Depends(get_owned_document),
    db: AsyncSession = Depends(get_db),
) -> None:
    """删除文档：先删对象存储文件，再删数据库记录（级联删除 chunks）。"""
    storage = get_storage()
    try:
        await storage.delete(doc.storage_key)
    except Exception:  # noqa: BLE001 对象不存在等不阻塞删除
        pass
    await db.delete(doc)
    await db.flush()
