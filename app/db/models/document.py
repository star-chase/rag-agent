"""文档模型：上传的原始文件，状态机管理解析流程。"""
from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

DOC_STATUS_PENDING = "pending"       # 已上传，等待处理
DOC_STATUS_PROCESSING = "processing"  # 解析/分块/向量化中
DOC_STATUS_READY = "ready"           # 完成，可检索
DOC_STATUS_FAILED = "failed"         # 处理失败


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)  # S3/MinIO 中的对象键
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)     # pdf/md/txt/...
    file_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=DOC_STATUS_PENDING, index=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document {self.id} {self.filename} [{self.status}]>"
