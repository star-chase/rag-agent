"""ORM 模型统一入口：导入即注册到 Base.metadata。"""
from app.db.models.chunk import Chunk
from app.db.models.conversation import Conversation, Message
from app.db.models.document import Document
from app.db.models.knowledge_base import KnowledgeBase
from app.db.models.user import User

__all__ = ["User", "KnowledgeBase", "Document", "Chunk", "Conversation", "Message"]
