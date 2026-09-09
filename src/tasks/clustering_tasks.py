"""
Celery Tasks for ML Clustering

Periodic tasks for clustering users into psychographic archetypes.
"""

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from datetime import datetime

from src.tasks.celery_app import app
from src.database.connection import get_db_session, reset_async_engine
from src.api.services.clustering import ClusteringService
from src.config.logging import get_logger

logger = get_logger(__name__)


class ClusteringTask(Task):
    """Base task for clustering operations"""
    _clustering_service = None

    @property
    def clustering_service(self):
        if self._clustering_service is None:
            self._clustering_service = ClusteringService()
        return self._clustering_service


@app.task(base=ClusteringTask, bind=True)
def cluster_all_topics_task(self):
    """
    Cluster users for all active topics.

    This task runs every 6 hours to update user archetypes
    based on their latest behavioral data.
    """
    import asyncio

    reset_async_engine()
    logger.info("Starting clustering for all topics")

    async def _cluster():
        try:
            async with get_db_session() as db:
                result = await self.clustering_service.cluster_all_topics(db)

                logger.info(f"Clustering complete: {result}")
                return result
        except SoftTimeLimitExceeded:
            logger.warning("cluster_all_topics_task soft time limit exceeded")
            return {"error": "time_limit_exceeded"}

    return asyncio.run(_cluster())


@app.task(base=ClusteringTask, bind=True)
def cluster_topic_task(self, topic_id: int):
    """
    Cluster users for a specific topic.

    Args:
        topic_id: Topic ID to cluster

    Can be called manually or via API for on-demand clustering.
    """
    import asyncio

    reset_async_engine()
    logger.info(f"Clustering topic_id={topic_id}")

    async def _cluster():
        async with get_db_session() as db:
            result = await self.clustering_service.cluster_users(topic_id, db)

            logger.info(f"Clustering complete: {result}")
            return result

    return asyncio.run(_cluster())
