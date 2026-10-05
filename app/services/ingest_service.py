"""文档摄入流水线：下载 -> 解析 -> 分块 -> 向量化 -> 入库。

刻意不用 LangChain/LlamaIndex，自己实现核心逻辑，便于讲清原理。
"""
import re
import uuid
from collections.abc import Iterable

from loguru import logger
from pypdf import PdfReader
from sqlalchemy import delete

from app.config import settings
from app.core.storage import get_storage
from app.db.models import Chunk, Document
from app.db.models.document import (
    DOC_STATUS_FAILED,
    DOC_STATUS_PROCESSING,
    DOC_STATUS_READY,
)
from app.db.session import AsyncSessionLocal
from app.services import embedding_service

# ---------- 解析 ----------

async def parse_document(data: bytes, file_type: str) -> str:
    """把原始文件解析为纯文本。"""
    if file_type == "pdf":
        return _parse_pdf(data)
    if file_type in ("md", "txt"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"不支持的解析类型: {file_type}")


def _parse_pdf(data: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"--- page {i + 1} ---\n{text}")
    return "\n".join(pages)


# ---------- 分块 ----------

def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """段落感知的文本分块。

    - 先按空行切段落，尽量在段落边界合并，避免切断语义；
    - 超长段落按 chunk_size 硬切，硬切处保留 overlap 字符衔接上下文。
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # 段落合并进当前块
        if current and len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}"
            continue
        if current:
            chunks.append(current)
            current = ""
        # 超长段落硬切
        while len(para) > chunk_size:
            chunks.append(para[:chunk_size])
            para = para[chunk_size - overlap :]
        current = para
    if current:
        chunks.append(current)
    return chunks


# ---------- 主流程 ----------

async def process_document(doc_id: int) -> None:
    """执行单篇文档的完整摄入流程（由 Celery worker 异步调用）。"""
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, doc_id)
        if doc is None:
            logger.warning("document {} not found, skip", doc_id)
            return

        try:
            doc.status = DOC_STATUS_PROCESSING
            await db.flush()
            logger.info("start processing document {} ({})", doc.id, doc.filename)

            # 1. 下载原始文件
            storage = get_storage()
            raw = await storage.download(doc.storage_key)

            # 2. 解析为文本
            text = await parse_document(raw, doc.file_type)
            if not text.strip():
                raise ValueError("文档解析结果为空")

            # 3. 分块
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("文档无法分块")

            # 4. 向量化（分批）
            vectors = await embedding_service.embed_texts(chunks)

            # 5. 删除旧块（支持重新处理）并写入新块
            await db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            for i, (content, vec) in enumerate(zip(chunks, vectors, strict=True)):
                db.add(
                    Chunk(
                        document_id=doc.id,
                        knowledge_base_id=doc.knowledge_base_id,
                        seq=i,
                        content=content,
                        embedding=vec,
                        meta={"filename": doc.filename, "chunk_index": i, "chunk_count": len(chunks)},
                    )
                )

            doc.status = DOC_STATUS_READY
            doc.chunk_count = len(chunks)
            doc.error_message = None
            await db.commit()
            logger.info("document {} ready, {} chunks", doc.id, len(chunks))
        except Exception as exc:  # noqa: BLE001
            doc.status = DOC_STATUS_FAILED
            doc.error_message = str(exc)[:1000]
            await db.commit()
            logger.error("document {} failed: {}", doc.id, exc)
            raise


def make_storage_key(kb_id: int, filename: str) -> str:
    """生成对象存储键：kb/<kb_id>/<uuid>/<filename>。"""
    return f"kb/{kb_id}/{uuid.uuid4().hex}/{filename}"


def split_into_batches(items: Iterable, size: int) -> Iterable[list]:
    """通用分批工具。"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
