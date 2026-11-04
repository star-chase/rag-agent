"""文本分块模型：携带 embedding 向量，是向量检索的基本单元。"""
from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.db.base import Base, TimestampMixin


def embedding_column():
    """vector 的 HNSW 最多 2000 维。本项目 1024，用 vector。"""
    dim = settings.embedding_dim
    if dim > 2000:
        return HALFVEC(dim)
    return Vector(dim)


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True, nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 块在文档中的顺序
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(embedding_column(), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # 页码/标题等元数据

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk {self.id} doc={self.document_id} seq={self.seq}>"
