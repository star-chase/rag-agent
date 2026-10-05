"""文档摄入的 Celery 任务。"""
import asyncio

from loguru import logger

from app.services.ingest_service import process_document
from app.workers.celery_app import celery_app


@celery_app.task(
    name="ingest.process_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def process_document_task(self, doc_id: int) -> None:
    """异步处理单篇文档；失败自动退避重试，重试耗尽则任务失败（文档已标记 failed）。"""
    logger.info("celery task start doc_id={}", doc_id)
    try:
        asyncio.run(process_document(doc_id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("celery task failed doc_id={}", doc_id)
        raise self.retry(exc=exc) from None  # type: ignore[misc]
