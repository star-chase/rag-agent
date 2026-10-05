"""Celery 应用：broker/backend 均用 Redis。"""
from celery import Celery

from app.config import settings

celery_app = Celery(
    "rag_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,          # 任务执行完再 ack，避免崩溃丢任务
    worker_prefetch_multiplier=1, # 一次只取一个任务，避免长任务占坑
    task_always_eager=settings.celery_task_always_eager,
)

# 发现任务模块
celery_app.autodiscover_tasks(["app.workers"])
