"""
Celery Tasks for Maintenance

Periodic tasks for database cleanup and maintenance.
"""

from celery import Task
from datetime import datetime, timedelta

from src.tasks.celery_app import app, HEARTBEAT_REDIS_KEY
from src.database.connection import get_db_session, reset_async_engine
from src.config.logging import get_logger

logger = get_logger(__name__)


@app.task
def cleanup_old_jobs_task():
    """
    Clean up old processing jobs from the database.

    This task runs daily at 2 AM to remove old job records.
    Keeps jobs from the last 30 days.
    """
    import asyncio

    reset_async_engine()
    logger.info("Starting cleanup of old processing jobs")

    async def _cleanup():
        from sqlalchemy import delete
        from src.database.models import ProcessingJob

        async with get_db_session() as db:
            # Delete jobs older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)

            stmt = delete(ProcessingJob).where(
                ProcessingJob.created_at < cutoff_date
            )

            result = await db.execute(stmt)
            await db.commit()

            deleted_count = result.rowcount
            logger.info(f"Deleted {deleted_count} old processing jobs")

            return {"deleted_jobs": deleted_count}

    return asyncio.run(_cleanup())


@app.task
def update_database_stats_task():
    """
    Update database statistics for query optimization.

    Runs weekly to keep query planner statistics fresh.
    """
    import asyncio

    reset_async_engine()
    logger.info("Updating database statistics")

    async def _update_stats():
        from src.database.connection import engine

        async with engine.begin() as conn:
            await conn.execute("ANALYZE;")

        logger.info("Database statistics updated")
        return {"status": "completed"}

    return asyncio.run(_update_stats())


@app.task
def prune_raw_data_task():
    """
    Prune raw data older than 1 Year (365 days).
    
    We retain 1 year of data to support long-term analytical assessment, 
    training, and validation. Data older than this is removed to control costs.
    Ideally, this deleted data should be archived to Cold Storage (S3) first.
    """
    import asyncio
    
    reset_async_engine()
    logger.info("Starting raw data pruning")

    async def _prune():
        from sqlalchemy import delete
        from src.database.models import PostRaw
        
        async with get_db_session() as db:
            # Delete posts older than 365 days (1 Year)
            cutoff_date = datetime.utcnow() - timedelta(days=365)
            
            stmt = delete(PostRaw).where(
                PostRaw.posted_at < cutoff_date
            )
            
            # Note: This might be slow on a huge table without indices on posted_at.
            # Assuming posted_at is indexed.
            result = await db.execute(stmt)
            await db.commit()
            
            deleted_count = result.rowcount
            logger.info(f"Pruned {deleted_count} old raw posts")
            
            return {"deleted_posts": deleted_count}

    return asyncio.run(_prune())


@app.task(ignore_result=True)
def worker_heartbeat_task():
    """
    Lightweight heartbeat: writes current timestamp to Redis every minute.

    The /health/cron endpoint reads this key to detect worker death
    independently of the Celery inspect API (which hangs when Redis
    is unreachable or workers are down).
    """
    try:
        redis_conn = app.connection_for_write().default_channel.client
        redis_conn.set(
            HEARTBEAT_REDIS_KEY,
            datetime.utcnow().isoformat(),
            ex=300,  # Expire after 5 minutes — if not refreshed, worker is dead
        )
    except Exception as e:
        logger.warning("worker_heartbeat_failed", error=str(e))
