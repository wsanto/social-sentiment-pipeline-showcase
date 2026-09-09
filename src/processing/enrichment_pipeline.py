"""
Emotion Enrichment Pipeline

Analyzes posts with Kaiko EQ+ and stores enriched emotion data.
Processes posts in batches for efficiency.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from src.integrations.kaiko_client import KaikoClient
from src.database.connection import get_db_session
from src.database.utils import (
    get_unenriched_posts,
    create_processing_job,
    update_job_status
)
from src.database.models import PostRaw, PostEnriched, ProcessingJob

logger = structlog.get_logger(__name__)


class EnrichmentPipeline:
    """
    Pipeline for enriching posts with emotion analysis

    Workflow:
    1. Fetch unenriched posts from database
    2. Analyze emotions using Kaiko EQ+
    3. Store enriched data in posts_enriched table
    4. Track processing job status
    """

    def __init__(self, kaiko_api_key: Optional[str] = None):
        """
        Initialize enrichment pipeline

        Args:
            kaiko_api_key: Kaiko API key (defaults to env var)
        """
        self.kaiko_client = KaikoClient(api_key=kaiko_api_key)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.kaiko_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.kaiko_client.__aexit__(exc_type, exc_val, exc_tb)

    async def enrich_post(
        self,
        db: AsyncSession,
        post: PostRaw
    ) -> Optional[PostEnriched]:
        """
        Enrich a single post with emotion analysis

        Args:
            db: Database session
            post: PostRaw object to enrich

        Returns:
            PostEnriched object if successful, None if error
        """
        try:
            # Analyze emotions
            analysis = await self.kaiko_client.analyze_text(post.content)

            # Create enriched post record
            enriched = PostEnriched(
                post_id=post.id,
                valence=analysis["valence"],
                arousal=analysis["arousal"],
                emotion_joy=analysis["emotions"]["joy"],
                emotion_trust=analysis["emotions"]["trust"],
                emotion_fear=analysis["emotions"]["fear"],
                emotion_surprise=analysis["emotions"]["surprise"],
                emotion_sadness=analysis["emotions"]["sadness"],
                emotion_disgust=analysis["emotions"]["disgust"],
                emotion_anger=analysis["emotions"]["anger"],
                emotion_anticipation=analysis["emotions"]["anticipation"],
                dominant_emotion=analysis["dominant_emotion"],
                toxicity_score=analysis["toxicity_score"],
                # V2 metrics
                intensity=analysis.get("intensity", 0.0),
                intensity_level=analysis.get("intensity_level"),
                emotional_complexity=analysis.get("emotional_complexity"),
                wonder_index=analysis.get("wonder_index", 0.0),
                discovery_level=analysis.get("discovery_level"),
                # P0: Safety & Hostility
                safety_concern_score=analysis.get("safety_concern_score", 0.0),
                hostility_level=analysis.get("hostility_level", "none"),
                hostility_score=analysis.get("hostility_score", 0.0),
                hostility_escalation_count=analysis.get("hostility_escalation_count", 0),
                de_escalation_detected=analysis.get("de_escalation_detected", False),
                is_breakthrough=analysis.get("is_breakthrough", False),
                # P1: Empathy & Meta-Emotional
                empathic_concern=analysis.get("empathic_concern", 0.0),
                personal_distress=analysis.get("personal_distress", 0.0),
                meta_emotional_score=analysis.get("meta_emotional_score", 0.0),
                emotional_vector=analysis.get("emotional_vector") or None,
                emotional_tags=analysis.get("emotional_tags") or None,
                active_labels=analysis.get("active_labels") or None,
                raw_27_emotions=analysis.get("raw_27_emotions") or None,
                # P2: Forward-compat
                emotional_signature=analysis.get("emotional_signature") or None,
                pattern_emotion_score=analysis.get("pattern_emotion_score", 0.0),
                dimensional_source=analysis.get("dimensional_source") or None,
                is_fallback=analysis.get("is_fallback", False),
                # Raw data
                raw_v2_emotions=analysis.get("raw_v2_emotions", {}),
                processed_at=datetime.utcnow()
            )

            db.add(enriched)

            logger.info(
                "post_enriched",
                post_id=post.id,
                dominant_emotion=analysis["dominant_emotion"],
                valence=analysis["valence"],
                safety_concern=analysis.get("safety_concern_score", 0.0),
                empathic_concern=analysis.get("empathic_concern", 0.0),
                is_breakthrough=analysis.get("is_breakthrough", False)
            )

            return enriched

        except Exception as e:
            logger.error(
                "enrichment_error",
                post_id=post.id,
                error=str(e)
            )
            return None

    async def enrich_batch(
        self,
        db: AsyncSession,
        posts: List[PostRaw],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Enrich multiple posts using Kaiko batch API

        Args:
            db: Database session
            posts: List of PostRaw objects to enrich
            batch_size: Number of posts to process per batch

        Returns:
            Dict with stats: {processed, succeeded, failed}
        """
        stats = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0
        }

        # Process in batches
        for i in range(0, len(posts), batch_size):
            batch = posts[i:i + batch_size]
            texts = [post.content for post in batch]

            try:
                # Batch analyze
                analyses = await self.kaiko_client.analyze_batch(texts)

                # Create enriched records
                for post, analysis in zip(batch, analyses):
                    try:
                        enriched = PostEnriched(
                            post_id=post.id,
                            valence=analysis["valence"],
                            arousal=analysis["arousal"],
                            emotion_joy=analysis["emotions"]["joy"],
                            emotion_trust=analysis["emotions"]["trust"],
                            emotion_fear=analysis["emotions"]["fear"],
                            emotion_surprise=analysis["emotions"]["surprise"],
                            emotion_sadness=analysis["emotions"]["sadness"],
                            emotion_disgust=analysis["emotions"]["disgust"],
                            emotion_anger=analysis["emotions"]["anger"],
                            emotion_anticipation=analysis["emotions"]["anticipation"],
                            dominant_emotion=analysis["dominant_emotion"],
                            toxicity_score=analysis["toxicity_score"],
                            # V2 metrics
                            intensity=analysis.get("intensity", 0.0),
                            intensity_level=analysis.get("intensity_level"),
                            emotional_complexity=analysis.get("emotional_complexity"),
                            wonder_index=analysis.get("wonder_index", 0.0),
                            discovery_level=analysis.get("discovery_level"),
                            # P0: Safety & Hostility
                            safety_concern_score=analysis.get("safety_concern_score", 0.0),
                            hostility_level=analysis.get("hostility_level", "none"),
                            hostility_score=analysis.get("hostility_score", 0.0),
                            hostility_escalation_count=analysis.get("hostility_escalation_count", 0),
                            de_escalation_detected=analysis.get("de_escalation_detected", False),
                            is_breakthrough=analysis.get("is_breakthrough", False),
                            # P1: Empathy & Meta-Emotional
                            empathic_concern=analysis.get("empathic_concern", 0.0),
                            personal_distress=analysis.get("personal_distress", 0.0),
                            meta_emotional_score=analysis.get("meta_emotional_score", 0.0),
                            emotional_vector=analysis.get("emotional_vector") or None,
                            emotional_tags=analysis.get("emotional_tags") or None,
                            active_labels=analysis.get("active_labels") or None,
                            raw_27_emotions=analysis.get("raw_27_emotions") or None,
                            # P2: Forward-compat
                            emotional_signature=analysis.get("emotional_signature") or None,
                            pattern_emotion_score=analysis.get("pattern_emotion_score", 0.0),
                            dimensional_source=analysis.get("dimensional_source") or None,
                            is_fallback=analysis.get("is_fallback", False),
                            # Raw data
                            raw_v2_emotions=analysis.get("raw_v2_emotions", {}),
                            processed_at=datetime.utcnow()
                        )

                        db.add(enriched)
                        stats["succeeded"] += 1

                    except Exception as e:
                        logger.error(
                            "enriched_record_creation_error",
                            post_id=post.id,
                            error=str(e)
                        )
                        stats["failed"] += 1

                stats["processed"] += len(batch)

                # Commit batch
                await db.commit()

                logger.info(
                    "batch_enriched",
                    batch_size=len(batch),
                    succeeded=stats["succeeded"],
                    failed=stats["failed"]
                )

            except Exception as e:
                logger.error(
                    "batch_enrichment_error",
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(e)
                )
                stats["failed"] += len(batch)
                await db.rollback()

        return stats

    async def enrich_unenriched_posts(
        self,
        limit: int = 100,
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Find and enrich posts that haven't been analyzed yet

        Args:
            limit: Max posts to process
            batch_size: Batch size for Kaiko API

        Returns:
            Stats dict
        """
        async with get_db_session() as db:
            # Create processing job
            job = await create_processing_job(
                db,
                job_type="enrich",
                status="running"
            )
            job_id = job.id
            await db.commit()

            try:
                # Get unenriched posts
                posts = await get_unenriched_posts(db, limit=limit)

                logger.info(
                    "found_unenriched_posts",
                    count=len(posts)
                )

                if not posts:
                    await update_job_status(
                        db,
                        job_id=job_id,
                        status="completed",
                        records_processed=0
                    )
                    await db.commit()
                    return {"processed": 0, "succeeded": 0, "failed": 0}

                # Enrich in batches
                stats = await self.enrich_batch(db, posts, batch_size=batch_size)

                # Update job status
                await update_job_status(
                    db,
                    job_id=job_id,
                    status="completed",
                    records_processed=stats["succeeded"],
                    records_failed=stats["failed"]
                )
                await db.commit()

                logger.info(
                    "enrichment_complete",
                    stats=stats
                )

                return stats

            except Exception as e:
                logger.error(
                    "enrichment_pipeline_failed",
                    error=str(e)
                )

                # Update job as failed
                await update_job_status(
                    db,
                    job_id=job_id,
                    status="failed",
                    error_message=str(e)
                )
                await db.commit()

                raise

    async def enrich_topic_posts(
        self,
        topic_id: int,
        hours_back: int = 24,
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Enrich posts for a specific topic within a time window

        Args:
            topic_id: Topic ID to filter posts
            hours_back: Time window in hours
            batch_size: Batch size for processing

        Returns:
            Stats dict
        """
        async with get_db_session() as db:
            # Create processing job
            period_start = datetime.utcnow() - timedelta(hours=hours_back)
            period_end = datetime.utcnow()

            job = await create_processing_job(
                db,
                job_type="enrich",
                status="running",
                topic_id=topic_id,
                period_start=period_start,
                period_end=period_end
            )
            job_id = job.id
            await db.commit()

            try:
                # Get posts for topic in time window
                # Note: We need to add a topic relationship to posts
                # For now, get unenriched posts in time window
                query = (
                    select(PostRaw)
                    .outerjoin(PostEnriched)
                    .where(PostEnriched.post_id.is_(None))
                    .where(PostRaw.posted_at >= period_start)
                    .where(PostRaw.posted_at <= period_end)
                    .limit(1000)
                )

                result = await db.execute(query)
                posts = result.scalars().all()

                logger.info(
                    "found_topic_posts",
                    topic_id=topic_id,
                    count=len(posts),
                    period_start=period_start,
                    period_end=period_end
                )

                if not posts:
                    await update_job_status(
                        db,
                        job_id=job_id,
                        status="completed",
                        records_processed=0
                    )
                    await db.commit()
                    return {"processed": 0, "succeeded": 0, "failed": 0}

                # Enrich in batches
                stats = await self.enrich_batch(db, posts, batch_size=batch_size)

                # Update job
                await update_job_status(
                    db,
                    job_id=job_id,
                    status="completed",
                    records_processed=stats["succeeded"],
                    records_failed=stats["failed"]
                )
                await db.commit()

                logger.info(
                    "topic_enrichment_complete",
                    topic_id=topic_id,
                    stats=stats
                )

                return stats

            except Exception as e:
                logger.error(
                    "topic_enrichment_failed",
                    topic_id=topic_id,
                    error=str(e)
                )

                await update_job_status(
                    db,
                    job_id=job_id,
                    status="failed",
                    error_message=str(e)
                )
                await db.commit()

                raise

    async def close(self):
        """Close the Kaiko client"""
        await self.kaiko_client.close()


# Convenience function for testing
async def test_pipeline():
    """Test the enrichment pipeline"""
    async with EnrichmentPipeline() as pipeline:
        # Enrich unenriched posts
        stats = await pipeline.enrich_unenriched_posts(limit=10, batch_size=5)
        print(f"Enrichment stats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pipeline())
