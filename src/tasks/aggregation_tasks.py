"""
Celery Tasks for Metrics Aggregation

Periodic tasks for calculating user metrics from enriched posts.
"""

from celery import Task

from src.tasks.celery_app import app
from src.database.connection import reset_async_engine
from src.processing.metrics_pipeline import MetricsAggregationPipeline
from src.config.logging import get_logger

logger = get_logger(__name__)


class AggregationTask(Task):
    """Base task for aggregation operations"""
    _metrics_pipeline = None

    @property
    def metrics_pipeline(self):
        if self._metrics_pipeline is None:
            self._metrics_pipeline = MetricsAggregationPipeline()
        return self._metrics_pipeline


@app.task(base=AggregationTask, bind=True)
def aggregate_user_metrics_task(self):
    """
    Aggregate user metrics for all active topics.

    This task runs hourly to calculate psychological metrics
    for all users based on their recent posts.
    """
    import asyncio

    reset_async_engine()
    logger.info("Starting user metrics aggregation")

    async def _aggregate():
        result = await self.metrics_pipeline.aggregate_all_topics(
            hours_back=24
        )

        overall_result = {
            "topics_processed": len(result),
            "total_users_updated": sum(
                stats.get("succeeded", 0) for stats in result.values()
            ),
            "errors": []
        }

        for topic_id, stats in result.items():
            if stats.get("failed", 0) > 0:
                overall_result["errors"].append(
                    f"topic {topic_id}: {stats['failed']} users failed"
                )

        logger.info(f"Aggregation complete: {overall_result}")
        return overall_result

    return asyncio.run(_aggregate())


@app.task(name="refresh_materialized_views")
def refresh_materialized_views():
    """
    Refreshes materialized views for analytics.
    Runs concurrently to avoid locking the tables.
    """
    from src.database.connection import sync_db_session
    from sqlalchemy import text

    logger.info("refresh_views_started")
    try:
        with sync_db_session() as session:
            # Refresh hourly sentiment
            session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hourly_sentiment"))

            # Refresh topic metrics
            session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_topic_metrics"))

            logger.info("refresh_views_completed")
            return {"status": "success", "views_refreshed": 2}

    except Exception as e:
        logger.error("refresh_views_failed", error=str(e))
        raise
