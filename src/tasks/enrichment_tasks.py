"""
Celery Tasks for Emotion Enrichment

Periodic tasks for processing posts through Kaiko EQ+ emotion analysis.
"""

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from datetime import datetime

from src.tasks.celery_app import app
from src.database.connection import get_db_session, reset_async_engine
from src.database.models import ProcessingJob
from src.api.services.enrichment import EnrichmentService
from src.integrations.kaiko_client import _rate_limiter
from src.config.logging import get_logger

logger = get_logger(__name__)


class EnrichmentTask(Task):
    """Base task for enrichment operations.

    Centralizes async lifecycle management: each Celery task invocation
    gets a fresh event loop via asyncio.run(), so we must reset the
    SQLAlchemy async engine (which caches connections bound to the
    previous loop) before each run.
    """

    def create_enrichment_service(self):
        """Create a fresh EnrichmentService (new KaikoClient session per task)."""
        return EnrichmentService()

    def run_async(self, coro_fn):
        """Run an async coroutine with proper engine lifecycle.

        Resets the async engine before creating a new event loop to avoid
        'Future attached to a different loop' errors, then runs the coroutine.
        """
        import asyncio
        reset_async_engine()
        return asyncio.run(coro_fn())


@app.task(base=EnrichmentTask, bind=True, max_retries=3)
def enrich_pending_posts_task(self, limit: int = 0):
    """
    Enrich posts that haven't been processed yet.

    This task runs every 5 minutes to process pending posts.
    Batch size adapts to the current Kaiko API rate limit state.
    When limit=0 (default from beat schedule), the rate limiter decides the batch size.

    Args:
        limit: Maximum posts to process. 0 = auto-size from rate limiter (recommended).
    """
    # Adaptive batch sizing: let the rate limiter decide based on current throughput
    if limit <= 0:
        limit = _rate_limiter.get_recommended_batch_size()

    logger.info(
        f"Starting enrichment task (limit={limit})",
        extra={"rate_limiter": _rate_limiter.get_stats()}
    )

    async def _enrich():
        service = self.create_enrichment_service()
        async with get_db_session() as db:
            job = ProcessingJob(
                job_type="enrichment",
                status="running",
                started_at=datetime.utcnow()
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)

            try:
                result = await service.enrich_pending_posts(db, limit=limit)

                if result.get("posts_enriched", 0) == 0 and result.get("posts_failed", 0) == 0:
                    logger.warning(
                        "enrichment_zero_records",
                        message="Enrichment task completed but processed 0 posts"
                    )

                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.records_processed = result.get("posts_enriched", 0)
                job.records_failed = result.get("posts_failed", 0)
                if result.get("errors"):
                    job.error_message = "; ".join(result["errors"][:5])
                await db.commit()

                logger.info(f"Enrichment complete: {result}")
                return result
            except SoftTimeLimitExceeded:
                logger.warning("enrich_pending_posts_task soft time limit exceeded")
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.error_message = "Task exceeded soft time limit"
                await db.commit()
                return {"error": "time_limit_exceeded", "posts_enriched": job.records_processed}

            except Exception as e:
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.error_message = str(e)
                await db.commit()
                logger.error(f"Enrichment task failed: {e}", exc_info=True)
                raise

    return self.run_async(_enrich)


@app.task(base=EnrichmentTask, bind=True)
def enrich_specific_posts_task(self, post_ids: list):
    """
    Enrich specific posts by ID.

    Args:
        post_ids: List of post IDs to enrich

    Can be called manually or via API for on-demand enrichment.
    """
    logger.info(f"Enriching {len(post_ids)} specific posts")

    async def _enrich():
        service = self.create_enrichment_service()
        async with get_db_session() as db:
            result = await service.enrich_posts(post_ids, db)

            logger.info(f"Enrichment complete: {result}")
            return result

    return self.run_async(_enrich)
