"""
User Metrics Aggregation Pipeline

Calculates 27 psychological features per user per topic.
Aggregates emotions from enriched posts into user profiles.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import structlog

from src.database.connection import get_db_session
from src.database.models import (
    PostRaw, PostEnriched, User, Topic, UserMetrics, ProcessingJob
)
from src.database.utils import (
    create_processing_job,
    update_job_status
)

logger = structlog.get_logger(__name__)


class MetricsAggregationPipeline:
    """
    Pipeline for aggregating user metrics from enriched posts

    Calculates 27 psychological features:
    - Emotion balance (2): avg_valence, avg_arousal
    - Emotion distribution (1): emotion_distribution dict
    - Volatility (4): valence_stddev, arousal_stddev, emotion_stability, topic_volatility
    - Dominant patterns (2): primary_emotion, secondary_emotion
    - Toxicity/Hostility (2): mean_toxicity, mean_hostility
    - Engagement (2): avg_engagement_score, reaction_bias
    - Temporal patterns (2): circadian_pattern, peak_activity_hour
    - Social influence (2): influence_score, follower_log
    - Cognitive complexity (3): topic_diversity, entropy_score, avg_content_length
    - Trust/Alignment (2): trust_level, agreement_rate
    - Post statistics (1): post_count
    """

    def __init__(self):
        """Initialize metrics aggregation pipeline"""
        pass

    def _calculate_stddev(self, values: List[float]) -> float:
        """
        Calculate standard deviation

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of emotion distribution

        Higher entropy = more diverse/unpredictable emotions
        Lower entropy = consistent/predictable emotional state

        Args:
            distribution: Dict mapping emotion -> probability

        Returns:
            Entropy score (0 to ~3.0 for 8 emotions)
        """
        if not distribution:
            return 0.0

        # Normalize to probabilities
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        probs = [v / total for v in distribution.values() if v > 0]

        # Calculate Shannon entropy: H = -Σ(p * log2(p))
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return entropy

    def _calculate_emotion_stability(
        self,
        valence_list: List[float],
        arousal_list: List[float]
    ) -> float:
        """
        Calculate emotional stability (inverse of volatility)

        Combines valence and arousal standard deviations.
        Higher = more stable/consistent emotions
        Lower = more volatile/erratic emotions

        Args:
            valence_list: List of valence values
            arousal_list: List of arousal values

        Returns:
            Stability score (0 to 1)
        """
        if not valence_list or not arousal_list:
            return 0.5  # Neutral

        valence_std = self._calculate_stddev(valence_list)
        arousal_std = self._calculate_stddev(arousal_list)

        # Combined volatility (0 to ~2, since std of -1 to 1 range is max 1)
        combined_volatility = (valence_std + arousal_std) / 2

        # Convert to stability (inverse)
        # Cap at 1.0 for normalization
        stability = max(0.0, min(1.0, 1.0 - combined_volatility))

        return stability

    def _calculate_circadian_pattern(
        self,
        posted_times: List[datetime]
    ) -> Tuple[Dict[str, float], Optional[int]]:
        """
        Calculate posting activity by hour of day

        Args:
            posted_times: List of post timestamps

        Returns:
            (circadian_pattern dict, peak_activity_hour)
            - circadian_pattern: {hour_0: 0.05, hour_1: 0.08, ...}
            - peak_activity_hour: Hour with most posts (0-23)
        """
        if not posted_times:
            return {}, None

        # Count posts by hour
        hour_counts = Counter(dt.hour for dt in posted_times)

        # Total posts
        total = len(posted_times)

        # Calculate probabilities
        circadian_pattern = {
            f"hour_{hour}": count / total
            for hour, count in hour_counts.items()
        }

        # Find peak hour
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]

        return circadian_pattern, peak_hour

    def _calculate_influence_score(
        self,
        follower_count: int,
        avg_engagement: float,
        post_count: int
    ) -> float:
        """
        Calculate social influence score

        Combines:
        - Follower count (reach)
        - Engagement rate (resonance)
        - Post frequency (activity)

        Args:
            follower_count: Number of followers
            avg_engagement: Average engagement per post
            post_count: Number of posts in period

        Returns:
            Influence score (0 to 1)
        """
        # Log-scale follower count (handles large ranges)
        follower_log = math.log10(max(1, follower_count))
        follower_score = min(1.0, follower_log / 6)  # Cap at 1M followers

        # Engagement score (normalized)
        engagement_score = min(1.0, avg_engagement / 1000)  # Cap at 1000 engagements

        # Activity score (posts per day, cap at 10/day)
        activity_score = min(1.0, post_count / 10)

        # Weighted combination
        influence = (
            0.5 * follower_score +
            0.3 * engagement_score +
            0.2 * activity_score
        )

        return round(influence, 4)

    async def aggregate_user_metrics(
        self,
        db: AsyncSession,
        user_id: str,
        topic_id: int,
        period_start: datetime,
        period_end: datetime
    ) -> Optional[UserMetrics]:
        """
        Calculate all 27 metrics for a user-topic combination

        Args:
            db: Database session
            user_id: User to aggregate
            topic_id: Topic to aggregate
            period_start: Start of time window
            period_end: End of time window

        Returns:
            UserMetrics object with all features calculated
        """
        try:
            # Query all enriched posts for user in time window
            query = (
                select(PostRaw, PostEnriched, User)
                .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
                .join(User, PostRaw.user_id == User.user_id)
                .where(
                    and_(
                        PostRaw.user_id == user_id,
                        PostRaw.posted_at >= period_start,
                        PostRaw.posted_at <= period_end
                    )
                )
            )

            result = await db.execute(query)
            rows = result.all()

            if not rows:
                logger.info(
                    "no_posts_for_user",
                    user_id=user_id,
                    topic_id=topic_id
                )
                return None

            # Extract data from rows
            posts = [row[0] for row in rows]
            enriched = [row[1] for row in rows]
            user = rows[0][2]  # Same user for all rows

            post_count = len(posts)

            # Extract all emotion values
            valence_list = [e.valence for e in enriched if e.valence is not None]
            arousal_list = [e.arousal for e in enriched if e.arousal is not None]
            toxicity_list = [e.toxicity_score for e in enriched if e.toxicity_score is not None]

            # P0/P1: Extract safety, empathy, and meta-emotional values
            safety_list = [e.safety_concern_score for e in enriched if e.safety_concern_score is not None]
            empathic_list = [e.empathic_concern for e in enriched if e.empathic_concern is not None]
            distress_list = [e.personal_distress for e in enriched if e.personal_distress is not None]
            meta_emotional_list = [e.meta_emotional_score for e in enriched if e.meta_emotional_score is not None]
            hostility_incidents = sum(1 for e in enriched if e.hostility_level in ("medium", "high"))
            breakthroughs = sum(1 for e in enriched if e.is_breakthrough)

            # P1: Aggregate active_labels distribution across all posts
            all_labels: Dict[str, int] = {}
            for e in enriched:
                if e.active_labels:
                    for label in e.active_labels:
                        all_labels[label] = all_labels.get(label, 0) + 1
            total_labels = sum(all_labels.values()) or 1
            active_labels_distribution = {k: round(v / total_labels, 4) for k, v in all_labels.items()}

            # Emotion distributions
            emotion_sums = {
                "joy": sum(e.emotion_joy for e in enriched if e.emotion_joy is not None),
                "trust": sum(e.emotion_trust for e in enriched if e.emotion_trust is not None),
                "fear": sum(e.emotion_fear for e in enriched if e.emotion_fear is not None),
                "surprise": sum(e.emotion_surprise for e in enriched if e.emotion_surprise is not None),
                "sadness": sum(e.emotion_sadness for e in enriched if e.emotion_sadness is not None),
                "disgust": sum(e.emotion_disgust for e in enriched if e.emotion_disgust is not None),
                "anger": sum(e.emotion_anger for e in enriched if e.emotion_anger is not None),
                "anticipation": sum(e.emotion_anticipation for e in enriched if e.emotion_anticipation is not None)
            }

            # Normalize emotion distribution
            total_emotions = sum(emotion_sums.values())
            emotion_distribution = {
                k: round(v / total_emotions, 4) if total_emotions > 0 else 0.0
                for k, v in emotion_sums.items()
            }

            # Find primary and secondary emotions
            sorted_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
            primary_emotion = sorted_emotions[0][0] if sorted_emotions else None
            secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None

            # Calculate all 27 features
            metrics = UserMetrics(
                user_id=user_id,
                topic_id=topic_id,
                period_start=period_start,
                period_end=period_end,

                # Emotion balance (2 features)
                avg_valence=round(sum(valence_list) / len(valence_list), 4) if valence_list else None,
                avg_arousal=round(sum(arousal_list) / len(arousal_list), 4) if arousal_list else None,

                # Emotion distribution (1 feature)
                emotion_distribution=emotion_distribution,

                # Volatility (4 features)
                valence_stddev=round(self._calculate_stddev(valence_list), 4) if valence_list else None,
                arousal_stddev=round(self._calculate_stddev(arousal_list), 4) if arousal_list else None,
                emotion_stability=round(
                    self._calculate_emotion_stability(valence_list, arousal_list), 4
                ),
                topic_volatility=round(
                    self._calculate_stddev([e.valence for e in enriched if e.valence is not None]), 4
                ) if enriched else None,

                # Dominant patterns (2 features)
                primary_emotion=primary_emotion,
                secondary_emotion=secondary_emotion,

                # Toxicity/Hostility (2 features)
                mean_toxicity=round(sum(toxicity_list) / len(toxicity_list), 4) if toxicity_list else None,
                mean_hostility=round(sum(toxicity_list) / len(toxicity_list), 4) if toxicity_list else None,  # Same as toxicity for now

                # P0: Safety & Hostility Aggregates
                avg_safety_concern=round(sum(safety_list) / len(safety_list), 4) if safety_list else None,
                hostility_incident_count=hostility_incidents,
                breakthrough_count=breakthroughs,

                # P1: Empathy & Meta-Emotional Aggregates
                avg_empathic_concern=round(sum(empathic_list) / len(empathic_list), 4) if empathic_list else None,
                avg_personal_distress=round(sum(distress_list) / len(distress_list), 4) if distress_list else None,
                avg_meta_emotional_score=round(sum(meta_emotional_list) / len(meta_emotional_list), 4) if meta_emotional_list else None,
                active_labels_distribution=active_labels_distribution if all_labels else None,

                # Engagement (2 features)
                avg_engagement_score=round(
                    sum(p.engagement_score for p in posts if p.engagement_score) / post_count, 4
                ) if any(p.engagement_score for p in posts) else 0.0,
                reaction_bias=round(
                    sum(1 for v in valence_list if v > 0) / len(valence_list), 4
                ) if valence_list else 0.5,  # Percentage positive reactions

                # Temporal patterns (2 features)
                circadian_pattern=self._calculate_circadian_pattern([p.posted_at for p in posts])[0],
                peak_activity_hour=self._calculate_circadian_pattern([p.posted_at for p in posts])[1],

                # Social influence (2 features)
                influence_score=self._calculate_influence_score(
                    follower_count=user.followers_count,
                    avg_engagement=sum(p.engagement_score for p in posts if p.engagement_score) / post_count if post_count > 0 else 0,
                    post_count=post_count
                ),
                follower_log=round(math.log10(max(1, user.followers_count)), 4),

                # Cognitive complexity (3 features)
                topic_diversity=1,  # Will be calculated across topics in future
                entropy_score=round(self._calculate_entropy(emotion_distribution), 4),
                avg_content_length=round(
                    sum(len(p.content) for p in posts) / post_count, 2
                ) if post_count > 0 else 0.0,

                # Trust/Alignment (2 features)
                trust_level=emotion_distribution.get("trust", 0.0),
                agreement_rate=round(
                    sum(1 for v in valence_list if abs(v) > 0.5) / len(valence_list), 4
                ) if valence_list else 0.0,  # Strong opinions rate

                # Post statistics (1 feature)
                post_count=post_count,

                # Metadata
                calculated_at=datetime.utcnow()
            )

            logger.info(
                "metrics_calculated",
                user_id=user_id,
                topic_id=topic_id,
                post_count=post_count,
                primary_emotion=primary_emotion,
                avg_valence=metrics.avg_valence
            )

            return metrics

        except Exception as e:
            logger.error(
                "metrics_calculation_error",
                user_id=user_id,
                topic_id=topic_id,
                error=str(e)
            )
            return None

    async def aggregate_topic_users(
        self,
        topic_id: int,
        period_start: datetime,
        period_end: datetime,
        limit: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Aggregate metrics for all users posting about a topic

        Args:
            topic_id: Topic to aggregate
            period_start: Start of time window
            period_end: End of time window
            limit: Optional limit on users to process

        Returns:
            Stats dict: {processed, succeeded, failed}
        """
        stats = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0
        }

        async with get_db_session() as db:
            # Create processing job
            job = await create_processing_job(
                db,
                job_type="aggregate",
                status="running",
                topic_id=topic_id,
                period_start=period_start,
                period_end=period_end
            )
            job_id = job.id
            await db.commit()

            try:
                # Get all users who posted about this topic in the period
                query = (
                    select(PostRaw.user_id)
                    .distinct()
                    .where(
                        and_(
                            PostRaw.posted_at >= period_start,
                            PostRaw.posted_at <= period_end
                        )
                    )
                )

                if limit:
                    query = query.limit(limit)

                result = await db.execute(query)
                user_ids = [row[0] for row in result.all()]

                logger.info(
                    "found_users_for_topic",
                    topic_id=topic_id,
                    user_count=len(user_ids)
                )

                # Aggregate metrics for each user
                for user_id in user_ids:
                    try:
                        metrics = await self.aggregate_user_metrics(
                            db,
                            user_id=user_id,
                            topic_id=topic_id,
                            period_start=period_start,
                            period_end=period_end
                        )

                        if metrics:
                            # Check if metrics already exist
                            existing_query = (
                                select(UserMetrics)
                                .where(
                                    and_(
                                        UserMetrics.user_id == user_id,
                                        UserMetrics.topic_id == topic_id,
                                        UserMetrics.period_start == period_start
                                    )
                                )
                            )
                            existing_result = await db.execute(existing_query)
                            existing = existing_result.scalar_one_or_none()

                            if existing:
                                # Update existing
                                for key, value in metrics.__dict__.items():
                                    if not key.startswith('_') and key not in ['id']:
                                        setattr(existing, key, value)
                            else:
                                # Create new
                                db.add(metrics)

                            stats["succeeded"] += 1

                        stats["processed"] += 1

                        # Commit every 10 users
                        if stats["processed"] % 10 == 0:
                            await db.commit()

                    except Exception as e:
                        logger.error(
                            "user_metrics_error",
                            user_id=user_id,
                            topic_id=topic_id,
                            error=str(e)
                        )
                        stats["failed"] += 1

                # Final commit
                await db.commit()

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
                    "topic_aggregation_complete",
                    topic_id=topic_id,
                    stats=stats
                )

            except Exception as e:
                logger.error(
                    "topic_aggregation_failed",
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

        return stats

    async def aggregate_all_topics(
        self,
        hours_back: int = 24,
        limit_per_topic: Optional[int] = None
    ) -> Dict[int, Dict[str, int]]:
        """
        Aggregate metrics for all active topics

        Args:
            hours_back: Time window in hours
            limit_per_topic: Optional limit on users per topic

        Returns:
            Dict mapping topic_id -> stats
        """
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=hours_back)

        results = {}

        async with get_db_session() as db:
            # Get all active topics
            query = select(Topic).where(Topic.is_active == True)
            result = await db.execute(query)
            topics = result.scalars().all()

            logger.info(
                "aggregating_all_topics",
                topic_count=len(topics),
                period_start=period_start,
                period_end=period_end
            )

            # Process each topic
            for topic in topics:
                stats = await self.aggregate_topic_users(
                    topic_id=topic.id,
                    period_start=period_start,
                    period_end=period_end,
                    limit=limit_per_topic
                )

                results[topic.id] = stats

        logger.info(
            "all_topics_aggregation_complete",
            results=results
        )

        return results


# Convenience function for testing
async def test_pipeline():
    """Test the metrics aggregation pipeline"""
    pipeline = MetricsAggregationPipeline()

    # Aggregate last 24 hours
    results = await pipeline.aggregate_all_topics(
        hours_back=24,
        limit_per_topic=10
    )

    for topic_id, stats in results.items():
        print(f"Topic {topic_id}: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pipeline())
