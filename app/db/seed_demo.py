"""灌入演示数据：demo 用户 + 演示知识库 + evals 语料（幂等，可重复执行）。

用法：
    PYTHONPATH=.deps python3 -m app.db.seed_demo
    PYTHONPATH=.deps python3 -m app.db.seed_demo --email demo@example.com --password demo12345

语料复用 evals 的 12 篇企业制度文档，摄入走与上传接口一致的链路
（对象存储 → Document → process_document），无需 worker（不依赖 Celery）。
"""
import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger
from sqlalchemy import func, select

from app.core.security import hash_password
from app.core.storage import get_storage, guess_file_type
from app.db.init_db import init as init_schema
from app.db.models import Chunk, Document, KnowledgeBase, User
from app.db.session import AsyncSessionLocal
from app.services.ingest_service import make_storage_key, process_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "evals" / "golden" / "corpus"
QUERIES_FILE = PROJECT_ROOT / "evals" / "golden" / "queries.jsonl"


async def _get_or_create_user(db, email: str, name: str, password: str) -> tuple[User, bool]:
    user = await db.scalar(select(User).where(User.email == email))
    if user is not None:
        return user, True
    user = User(email=email, name=name, hashed_password=hash_password(password))
    db.add(user)
    await db.flush()
    return user, False


async def _get_or_create_kb(db, user: User, name: str) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(KnowledgeBase.owner_id == user.id, KnowledgeBase.name == name)
    )
    if kb is not None:
        return kb
    kb = KnowledgeBase(owner_id=user.id, name=name, description="演示用企业制度知识库（seed_demo 生成）")
    db.add(kb)
    await db.flush()
    return kb


async def _ingest_corpus(db, kb: KnowledgeBase, corpus_dir: Path) -> tuple[int, int]:
    """上传并摄入语料；已存在且 ready 的文档跳过。返回 (新增数, 跳过数)。"""
    storage = get_storage()
    added = skipped = 0

    for path in sorted(corpus_dir.glob("*")):
        if not path.is_file():
            continue
        existing = await db.scalar(
            select(Document).where(Document.knowledge_base_id == kb.id, Document.filename == path.name)
        )
        if existing is not None and existing.status == "ready":
            skipped += 1
            continue

        data = path.read_bytes()
        file_type = guess_file_type(path.name)
        if existing is not None:  # 上次失败/中断：重跑摄入，不重复建记录
            doc = existing
        else:
            storage_key = make_storage_key(kb.id, path.name)
            await storage.upload(storage_key, data, content_type="text/markdown" if file_type == "md" else None)
            doc = Document(
                knowledge_base_id=kb.id,
                filename=path.name,
                storage_key=storage_key,
                file_type=file_type,
                file_size=len(data),
            )
            db.add(doc)
        await db.commit()
        await process_document(doc.id)
        await db.refresh(doc)
        if doc.status != "ready":
            logger.error("{} 摄入失败: {}", path.name, doc.error_message)
            continue
        added += 1

    return added, skipped


def _sample_questions(limit: int = 6) -> list[str]:
    if not QUERIES_FILE.exists():
        return []
    questions = []
    for line in QUERIES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("answerable"):
            questions.append(item["question"])
        if len(questions) >= limit:
            break
    return questions


async def seed(email: str, password: str, name: str, kb_name: str, corpus_dir: Path) -> None:
    await init_schema()

    async with AsyncSessionLocal() as db:
        user, existed = await _get_or_create_user(db, email, name, password)
        kb = await _get_or_create_kb(db, user, kb_name)
        await db.commit()

        added, skipped = await _ingest_corpus(db, kb, corpus_dir)

        doc_count = await db.scalar(
            select(func.count()).select_from(Document).where(Document.knowledge_base_id == kb.id)
        )
        chunk_total = await db.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.knowledge_base_id == kb.id)
        )

    print("=" * 56)
    print("演示数据就绪" + ("（用户已存在，复用）" if existed else ""))
    print(f"  登录账号：{email} / {password}")
    print(f"  知识库  ：#{kb.id} {kb.name}")
    print(f"  文档    ：{doc_count} 篇（本次新摄入 {added}，跳过已就绪 {skipped}）")
    print(f"  向量块  ：{chunk_total} chunks")
    print("  打开演示台：http://127.0.0.1:8000/demo/")
    questions = _sample_questions()
    if questions:
        print("  可试提问：")
        for q in questions:
            print(f"    - {q}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="灌入 RAG Agent 演示数据（幂等）")
    parser.add_argument("--email", default="demo@example.com")
    parser.add_argument("--password", default="demo12345")
    parser.add_argument("--name", default="演示用户")
    parser.add_argument("--kb", default="企业制度演示库")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    if not args.corpus.is_dir():
        raise SystemExit(f"语料目录不存在：{args.corpus}")
    asyncio.run(seed(args.email, args.password, args.name, args.kb, args.corpus))


if __name__ == "__main__":
    main()
