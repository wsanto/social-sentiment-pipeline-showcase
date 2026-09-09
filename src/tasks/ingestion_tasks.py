"""
Celery Tasks for Data Ingestion

Periodic tasks for fetching data from LunarCrush API.

Rate Limiting:
- LunarCrush allows 10 requests/minute on Individual plan
- We add 7-second delays between topic fetches to stay under limit
- This means ~18 topics takes ~2 minutes to process
"""

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy import select
import asyncio

from src.tasks.celery_app import app
from src.database.connection import get_db_session, reset_async_engine
from src.database.models import Topic, ProcessingJob
from src.integrations.lunarcrush_client import LunarCrushClient, REQUEST_DELAY_SECONDS
from src.api.services.ingestion import IngestionService
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# Rate limiting: slightly longer than client delay to be safe
TOPIC_FETCH_DELAY_SECONDS = REQUEST_DELAY_SECONDS + 1  # 7 seconds


class DatabaseTask(Task):
    """Base task with database session management"""
    _db = None
    _ingestion_service = None

    def create_lunarcrush_client(self):
        """Create a fresh LunarCrush client for each task invocation.

        Must not be cached across tasks because asyncio.run() creates a new
        event loop each time, and the client's asyncio.Lock would be bound
        to the previous (now-closed) loop, causing silent failures.
        """
        return LunarCrushClient(api_key=settings.lunarcrush_api_key)

    @property
    def ingestion_service(self):
        if self._ingestion_service is None:
            self._ingestion_service = IngestionService()
        return self._ingestion_service


@app.task(base=DatabaseTask, bind=True, max_retries=3)
def fetch_posts_task(self):
    """
    Fetch posts from LunarCrush for all active topics.

    This task runs every 15 minutes to collect new social media posts.
    """
    import asyncio

    reset_async_engine()
    logger.info("Starting post fetching task")

    async def _fetch():
        overall_stats = {
            "topics_processed": 0,
            "total_posts_fetched": 0,
            "total_posts_ingested": 0,
            "errors": []
        }

        async with get_db_session() as db:
            # Get active topics
            result = await db.execute(
                select(Topic).where(Topic.is_active == True)
            )
            topics = result.scalars().all()

            logger.info(f"Found {len(topics)} active topics")
            logger.info(f"Rate limiting: {TOPIC_FETCH_DELAY_SECONDS}s delay between topics, ~{len(topics) * TOPIC_FETCH_DELAY_SECONDS / 60:.1f} min total")

            # Create a fresh client for this event loop
            async with self.create_lunarcrush_client() as client:
                for i, topic in enumerate(topics):
                    # Rate limiting: wait between topics to avoid 429 errors
                    if i > 0:
                        logger.debug(f"Rate limit delay: waiting {TOPIC_FETCH_DELAY_SECONDS}s before next topic")
                        await asyncio.sleep(TOPIC_FETCH_DELAY_SECONDS)
                    try:
                        # Create processing job
                        job = ProcessingJob(
                            job_type="ingestion",
                            topic_id=topic.id,
                            status="running",
                            started_at=datetime.utcnow()
                        )
                        db.add(job)
                        await db.commit()
                        await db.refresh(job)

                        # Fetch posts from LunarCrush
                        logger.info(f"Fetching posts for topic: {topic.topic_name}")

                        # Last 1 hour of data
                        end = int(datetime.utcnow().timestamp())
                        start = int((datetime.utcnow() - timedelta(hours=1)).timestamp())

                        posts_data = await client.get_topic_posts(
                            topic=topic.topic_name.lower(),
                            start=start,
                            end=end,
                            limit=500
                        )

                        logger.info(f"Fetched {len(posts_data)} posts for {topic.topic_name}")

                        if posts_data:
                            # Ingest posts
                            ingest_result = await self.ingestion_service.ingest_batch(
                                posts_data=posts_data,
                                topic_id=topic.id,
                                db=db
                            )

                            # Update job
                            job.status = "completed"
                            job.completed_at = datetime.utcnow()
                            job.records_processed = ingest_result["posts_ingested"]
                            job.records_failed = ingest_result["posts_skipped"]
                            await db.commit()

                            overall_stats["total_posts_fetched"] += len(posts_data)
                            overall_stats["total_posts_ingested"] += ingest_result["posts_ingested"]
                        else:
                            job.status = "completed"
                            job.completed_at = datetime.utcnow()
                            job.records_processed = 0
                            await db.commit()

                        overall_stats["topics_processed"] += 1

                    except SoftTimeLimitExceeded:
                        logger.warning(f"fetch_posts_task soft time limit exceeded at topic {topic.topic_name}")
                        job.status = "failed"
                        job.error_message = "Task exceeded soft time limit"
                        job.completed_at = datetime.utcnow()
                        await db.commit()
                        overall_stats["errors"].append("time_limit_exceeded")
                        return overall_stats

                    except Exception as e:
                        logger.error(f"Error fetching posts for topic {topic.topic_name}: {e}", exc_info=True)
                        overall_stats["errors"].append(str(e))

                        # Update job as failed
                        job.status = "failed"
                        job.error_message = str(e)
                        job.completed_at = datetime.utcnow()
                        await db.commit()

        logger.info(f"Post fetching complete: {overall_stats}")
        return overall_stats

    # Run async function
    return asyncio.run(_fetch())


@app.task(base=DatabaseTask, bind=True)
def fetch_topic_posts_task(self, topic_id: int, hours_back: int = 1, limit: int = 500):
    """
    Fetch posts for a specific topic.

    Args:
        topic_id: Topic ID to fetch posts for
        hours_back: How many hours of data to fetch
        limit: Maximum number of posts

    This can be called manually or via API to fetch specific topic data.
    """
    import asyncio

    reset_async_engine()
    logger.info(f"Fetching posts for topic_id={topic_id}")

    async def _fetch():
        async with get_db_session() as db:
            # Get topic
            result = await db.execute(
                select(Topic).where(Topic.id == topic_id)
            )
            topic = result.scalar_one_or_none()

            if not topic:
                logger.error(f"Topic {topic_id} not found")
                return {"error": f"Topic {topic_id} not found"}

            # Create processing job
            job = ProcessingJob(
                job_type="ingestion",
                topic_id=topic_id,
                status="running",
                started_at=datetime.utcnow()
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)

            try:
                # Fetch posts
                end = int(datetime.utcnow().timestamp())
                start = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())

                async with self.create_lunarcrush_client() as client:
                    posts_data = await client.get_topic_posts(
                        topic=topic.topic_name.lower(),
                        start=start,
                        end=end,
                        limit=limit
                    )

                logger.info(f"Fetched {len(posts_data)} posts")

                # Ingest posts
                if posts_data:
                    ingest_result = await self.ingestion_service.ingest_batch(
                        posts_data=posts_data,
                        topic_id=topic_id,
                        db=db
                    )

                    # Update job
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.records_processed = ingest_result["posts_ingested"]
                    job.records_failed = ingest_result["posts_skipped"]
                    await db.commit()

                    return ingest_result
                else:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.records_processed = 0
                    await db.commit()

                    return {"posts_ingested": 0, "posts_skipped": 0}

            except Exception as e:
                logger.error(f"Error fetching posts: {e}", exc_info=True)

                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await db.commit()

                raise

    return asyncio.run(_fetch())
