"""
Data Ingestion Pipeline

Fetches social media posts from LunarCrush and stores them in the database.
Handles deduplication, user creation, and error recovery.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.integrations.lunarcrush_client import LunarCrushClient
from src.database.connection import get_db_session
from langdetect import detect as detect_language, LangDetectException
from src.database.models import InfluencerRegistry
from src.database.utils import (
    get_or_create_user,
    create_post,
    get_topic_by_name,
    create_processing_job,
    update_job_status
)
from src.database.models import ProcessingJob

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    Pipeline for ingesting social media posts from LunarCrush

    Workflow:
    1. Fetch posts for specified topics from LunarCrush
    2. Create/update user records
    3. Create post records (with deduplication)
    4. Track processing job status
    """

    def __init__(self, lunarcrush_api_key: Optional[str] = None):
        """
        Initialize ingestion pipeline

        Args:
            lunarcrush_api_key: LunarCrush API key (defaults to env var)
        """
        self.lc_client = LunarCrushClient(api_key=lunarcrush_api_key)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.lc_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.lc_client.__aexit__(exc_type, exc_val, exc_tb)

    def _extract_post_data(self, raw_post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize post data from LunarCrush response

        Args:
            raw_post: Raw post object from LunarCrush API

        Returns:
            Normalized post dict for database insertion
        """
        # LunarCrush post structure varies by platform
        # Common fields: id, text/content, created, creator, interactions, etc.

        external_id = raw_post.get("id") or raw_post.get("post_id")
        content = raw_post.get("text") or raw_post.get("content") or ""

        # Creator/user info
        creator = raw_post.get("creator", {})
        creator_id = creator.get("id") or creator.get("creator_id", "unknown")
        creator_name = creator.get("name") or creator.get("username", "unknown")

        # Platform detection
        platform = raw_post.get("platform", "unknown")
        post_type = raw_post.get("type", "")

        # Map post_type to platform if not specified
        if platform == "unknown" and post_type:
            platform_map = {
                "tweet": "twitter",
                "reddit-post": "reddit",
                "youtube-video": "youtube",
                "tiktok-video": "tiktok",
                "news": "news"
            }
            platform = platform_map.get(post_type, "unknown")

        # Posted timestamp
        posted_at = raw_post.get("created") or raw_post.get("time")
        if isinstance(posted_at, (int, float)):
            posted_at = datetime.fromtimestamp(posted_at)
        elif isinstance(posted_at, str):
            try:
                posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
            except Exception:
                posted_at = datetime.utcnow()
        else:
            posted_at = datetime.utcnow()

        # Engagement metrics
        interactions = raw_post.get("interactions", 0)
        engagement_score = float(interactions) if interactions else 0.0

        # Hashtags and mentions
        hashtags = raw_post.get("hashtags", [])
        mentions = raw_post.get("mentions", [])

        # Store full metadata
        post_metadata = {
            "interactions": interactions,
            "post_type": post_type,
            "url": raw_post.get("url"),
            "source_data": raw_post  # Keep original for reference
        }

        # Detect language (fast, <1ms per text)
        language = None
        if content and len(content) > 10:
            try:
                language = detect_language(content)
            except LangDetectException:
                language = None

        return {
            "external_id": str(external_id),
            "user_id": str(creator_id),
            "username": str(creator_name),
            "content": content,
            "platform": platform,
            "posted_at": posted_at,
            "hashtags": hashtags if hashtags else None,
            "mentions": mentions if mentions else None,
            "engagement_score": engagement_score,
            "language": language,
            "post_metadata": post_metadata
        }

    def _extract_user_data(self, creator: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Extract user data from LunarCrush creator object

        Args:
            creator: Creator/user object from LunarCrush
            platform: Platform name

        Returns:
            User dict for database insertion
        """
        user_id = creator.get("id") or creator.get("creator_id", "unknown")
        username = creator.get("name") or creator.get("username", "unknown")

        followers = creator.get("followers_count") or creator.get("followers", 0)
        following = creator.get("following_count") or creator.get("following", 0)
        verified = creator.get("verified", False)

        # User metadata
        user_metadata = {
            "profile_url": creator.get("url"),
            "bio": creator.get("bio") or creator.get("description"),
            "source_data": creator
        }

        return {
            "user_id": str(user_id),
            "username": str(username),
            "platform": platform,
            "followers_count": int(followers) if followers else 0,
            "following_count": int(following) if following else 0,
            "verified": bool(verified),
            "profile_post_metadata": user_metadata
        }

    async def ingest_topic(
        self,
        db: AsyncSession,
        topic_name: str,
        hours_back: int = 24,
        limit: int = 100
    ) -> Dict[str, int]:
        """
        Ingest posts for a single topic

        Args:
            db: Database session
            topic_name: Topic to fetch posts for
            hours_back: How many hours back to fetch
            limit: Max posts to fetch

        Returns:
            Dict with stats: {posts_fetched, posts_created, users_created, errors}
        """
        stats = {
            "posts_fetched": 0,
            "posts_created": 0,
            "posts_skipped": 0,
            "users_created": 0,
            "errors": 0
        }

        # Get topic from database
        topic = await get_topic_by_name(db, topic_name)
        if not topic:
            logger.error("topic_not_found", topic=topic_name)
            return stats

        logger.info(
            "ingesting_topic",
            topic=topic_name,
            hours_back=hours_back,
            limit=limit
        )

        try:
            # Fetch posts from LunarCrush
            end = int(datetime.utcnow().timestamp())
            start = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())

            raw_posts = await self.lc_client.get_topic_posts(
                topic=topic_name.lower(),
                start=start,
                end=end,
                limit=limit
            )

            stats["posts_fetched"] = len(raw_posts)

            logger.info(
                "fetched_posts",
                topic=topic_name,
                count=len(raw_posts)
            )

            # Process each post
            for raw_post in raw_posts:
                try:
                    # Extract post data
                    post_data = self._extract_post_data(raw_post)

                    # Create or get user
                    creator = raw_post.get("creator", {})
                    user_data = self._extract_user_data(creator, post_data["platform"])

                    user, user_created = await get_or_create_user(
                        db,
                        user_id=user_data["user_id"],
                        username=user_data["username"],
                        platform=user_data["platform"],
                        **{k: v for k, v in user_data.items() if k not in ["user_id", "username", "platform"]}
                    )

                    if user_created:
                        stats["users_created"] += 1

                    # Check if this user is a known influencer or should be auto-registered
                    try:
                        existing_inf = await db.execute(
                            select(InfluencerRegistry.id).where(
                                InfluencerRegistry.user_id == user_data["user_id"]
                            ).limit(1)
                        )
                        is_influencer = existing_inf.scalar() is not None

                        if not is_influencer:
                            followers = user_data.get("followers_count", 0) or 0
                            if followers >= 10000:
                                # Auto-register high-follower users as influencers
                                tier = "mega" if followers >= 1000000 else "macro" if followers >= 100000 else "micro"
                                db.add(InfluencerRegistry(
                                    user_id=user_data["user_id"],
                                    topic_id=topic.id if topic else None,
                                    tier=tier,
                                    current_follower_count=followers,
                                    is_active=True,
                                    monitoring_enabled=True,
                                ))
                                is_influencer = True
                                logger.info("influencer_auto_registered",
                                           user_id=user_data["user_id"], tier=tier, followers=followers)

                        if is_influencer:
                            post_data.setdefault("post_metadata", {})["is_influencer"] = True
                    except Exception as inf_err:
                        logger.debug("influencer_check_skipped", error=str(inf_err))

                    # Create post (with deduplication)
                    post_created = await create_post(
                        db,
                        external_id=post_data["external_id"],
                        user_id=post_data["user_id"],
                        content=post_data["content"],
                        platform=post_data["platform"],
                        posted_at=post_data["posted_at"],
                        hashtags=post_data.get("hashtags"),
                        mentions=post_data.get("mentions"),
                        engagement_score=post_data.get("engagement_score"),
                        language=post_data.get("language"),
                        post_metadata=post_data.get("post_metadata")
                    )

                    if post_created:
                        stats["posts_created"] += 1
                    else:
                        stats["posts_skipped"] += 1

                except Exception as e:
                    logger.error(
                        "post_processing_error",
                        topic=topic_name,
                        post_id=raw_post.get("id"),
                        error=str(e)
                    )
                    stats["errors"] += 1

            # Commit all changes
            await db.commit()

            logger.info(
                "topic_ingestion_complete",
                topic=topic_name,
                stats=stats
            )

        except Exception as e:
            logger.error(
                "topic_ingestion_failed",
                topic=topic_name,
                error=str(e)
            )
            await db.rollback()
            stats["errors"] += 1

        return stats

    async def ingest_multiple_topics(
        self,
        topic_names: List[str],
        hours_back: int = 24,
        limit_per_topic: int = 100
    ) -> Dict[str, Dict[str, int]]:
        """
        Ingest posts for multiple topics

        Args:
            topic_names: List of topics to ingest
            hours_back: How many hours back to fetch
            limit_per_topic: Max posts per topic

        Returns:
            Dict mapping topic -> stats
        """
        results = {}

        async with get_db_session() as db:
            # Create processing job
            job = await create_processing_job(
                db,
                job_type="fetch",
                status="running",
                period_start=datetime.utcnow() - timedelta(hours=hours_back),
                period_end=datetime.utcnow()
            )
            job_id = job.id

            await db.commit()

            total_records = 0
            total_errors = 0

            try:
                # Process each topic
                for topic_name in topic_names:
                    stats = await self.ingest_topic(
                        db,
                        topic_name=topic_name,
                        hours_back=hours_back,
                        limit=limit_per_topic
                    )

                    results[topic_name] = stats
                    total_records += stats["posts_created"]
                    total_errors += stats["errors"]

                # Update job status
                await update_job_status(
                    db,
                    job_id=job_id,
                    status="completed",
                    records_processed=total_records,
                    records_failed=total_errors
                )

                await db.commit()

            except Exception as e:
                logger.error(
                    "multi_topic_ingestion_failed",
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

        logger.info(
            "multi_topic_ingestion_complete",
            topics=len(topic_names),
            total_posts=total_records,
            total_errors=total_errors
        )

        return results

    async def close(self):
        """Close the LunarCrush client"""
        await self.lc_client.close()


# Convenience function for testing
async def test_pipeline():
    """Test the ingestion pipeline"""
    async with IngestionPipeline() as pipeline:
        # Test ingesting a single topic
        topics = ["politics", "crypto"]

        results = await pipeline.ingest_multiple_topics(
            topic_names=topics,
            hours_back=24,
            limit_per_topic=10
        )

        for topic, stats in results.items():
            print(f"{topic}: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pipeline())
