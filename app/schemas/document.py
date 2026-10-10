"""文档 Pydantic 模型。"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    error_message: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentStatusOut(BaseModel):
    id: int
    status: str
    error_message: str | None
    chunk_count: int


class ChunkOut(BaseModel):
    """文档分块（供演示台查看解析结果，不含向量）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    seq: int
    content: str
