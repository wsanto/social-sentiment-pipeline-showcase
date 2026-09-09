"""
Topics API Endpoints

Provides topic listing, filtering, and detail views.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, text
from datetime import datetime, timedelta
import structlog

from src.database.connection import get_db
from src.database.models import Topic, PostRaw, PostEnriched, User
from src.api.auth import require_read_permission, APIKey
from src.api.time_utils import get_time_range

logger = structlog.get_logger(__name__)

router = APIRouter()

# Keyword definitions per topic name (for sub-topic breakdown)
# Keys should be lowercase to match normalized topic names
# Standardized topic-to-subtopic mapping
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    # Core topics with standardized sub-topics
    "politics": [
        "elections_voting",
        "government_policy",
        "international_relations",
        "geopolitics",
        "republicans",
        "democrats",
        "legislation_bills",
        "political_scandals",
        "human_rights",
        "national_security"
    ],
    "crypto": [
        "bitcoin",
        "ethereum",
        "altcoins",
        "defi",
        "nfts",
        "web3",
        "crypto_regulation",
        "XRP",
        "base",
        "solana"
    ],
    "cryptocurrency": [
        "bitcoin",
        "ethereum",
        "altcoins",
        "defi",
        "nfts",
        "web3",
        "crypto_regulation",
        "XRP",
        "base",
        "solana"
    ],
    "sports": [
        "football_soccer",
        "basketball",
        "baseball",
        "hockey",
        "tennis",
        "motorsports",
        "esports",
        "olympics",
        "sports_business",
        "archery"
    ],
    "entertainment": [
        "movies",
        "tv_streaming",
        "music",
        "celebrities",
        "gaming",
        "pop_culture",
        "awards_festivals",
        "influencers",
        "fan_communities",
        "trailers_releases"
    ],
    "technology": [
        "artificial_intelligence",
        "startups",
        "big_tech",
        "software_apps",
        "hardware",
        "cybersecurity",
        "data_analytics",
        "cloud_computing",
        "robotics",
        "emerging_tech"
    ],
    "tech": [
        "artificial_intelligence",
        "startups",
        "big_tech",
        "software_apps",
        "hardware",
        "cybersecurity",
        "data_analytics",
        "cloud_computing",
        "robotics",
        "emerging_tech"
    ],
    "health": [
        "mental_health",
        "fitness",
        "nutrition",
        "public_health",
        "medicine",
        "healthcare_policy",
        "wellness",
        "disease_research",
        "digital_health",
        "longevity"
    ],
    "healthcare": [
        "mental_health",
        "fitness",
        "nutrition",
        "public_health",
        "medicine",
        "healthcare_policy",
        "wellness",
        "disease_research",
        "digital_health",
        "longevity"
    ],
    "education": [
        "higher_education",
        "k12",
        "online_learning",
        "edtech",
        "skills_careers",
        "research_academia",
        "student_life",
        "policy_reform",
        "teaching_methods",
        "lifelong_learning"
    ],
    "economy": [
        "inflation",
        "interest_rates",
        "markets_stocks",
        "employment_jobs",
        "trade",
        "economic_policy",
        "recession_growth",
        "corporate_earnings",
        "housing_real_estate",
        "global_economy"
    ],
    "economic": [
        "inflation",
        "interest_rates",
        "markets_stocks",
        "employment_jobs",
        "trade",
        "economic_policy",
        "recession_growth",
        "corporate_earnings",
        "housing_real_estate",
        "global_economy"
    ],
    "environment": [
        "climate_change",
        "sustainability",
        "renewable_energy",
        "conservation",
        "biodiversity",
        "pollution",
        "climate_policy",
        "natural_disasters",
        "environmental_tech",
        "carbon_markets"
    ],
    "social issues": [
        "inequality",
        "social_justice",
        "gender_issues",
        "race_identity",
        "indigenous_issues",
        "labor_workers",
        "housing_homelessness",
        "migration_refugees",
        "digital_rights",
        "community_culture"
    ],
    "social_issues": [
        "inequality",
        "social_justice",
        "gender_issues",
        "race_identity",
        "indigenous_issues",
        "labor_workers",
        "housing_homelessness",
        "migration_refugees",
        "digital_rights",
        "community_culture"
    ],
    "social": [
        "inequality",
        "social_justice",
        "gender_issues",
        "race_identity",
        "indigenous_issues",
        "labor_workers",
        "housing_homelessness",
        "migration_refugees",
        "digital_rights",
        "community_culture"
    ],
}


@router.get("/topics")
async def get_topics(
    sort_by: str = Query("mentions", description="Sort field: mentions, sentiment, name"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(50, ge=1, le=100, description="Number of topics to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    search: Optional[str] = Query(None, description="Search query for topic name"),
    days_back: int = Query(7, ge=1, le=90, description="Days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of topics with statistics.

    Returns paginated list of topics with mention counts, sentiment,
    and emotion data computed from posts within the requested time range.
    """
    start_date, end_date = get_time_range(days_back)

    # Also compute the previous period for trend calculation
    prev_start = start_date - timedelta(days=days_back)
    prev_end = start_date

    try:
        # Get all active topics
        topics_query = select(Topic).where(Topic.is_active == True)
        if search:
            topics_query = topics_query.where(
                Topic.topic_name.ilike(f"%{search}%")
            )
        topics_result = await db.execute(topics_query)
        all_topics = topics_result.scalars().all()

        # Get posts in the current time range
        posts_query = (
            select(PostRaw, PostEnriched)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(and_(
                PostRaw.posted_at >= start_date,
                PostRaw.posted_at <= end_date
            ))
        )
        posts_result = await db.execute(posts_query)
        posts_with_enrichment = posts_result.all()

        # Get posts in the previous period for trend calculation
        prev_posts_query = (
            select(PostRaw, PostEnriched)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(and_(
                PostRaw.posted_at >= prev_start,
                PostRaw.posted_at <= prev_end
            ))
        )
        prev_posts_result = await db.execute(prev_posts_query)
        prev_posts = prev_posts_result.all()

        # Match posts to topics by keyword (same logic as dashboard)
        topic_stats: Dict[int, Dict[str, Any]] = {}
        prev_topic_counts: Dict[int, int] = {}

        for topic in all_topics:
            topic_stats[topic.id] = {
                "topic": topic,
                "count": 0,
                "total_valence": 0.0,
                "emotions": {},
            }
            prev_topic_counts[topic.id] = 0

        def match_post_to_topics(post, enriched, stats_dict, count_dict=None):
            content_lower = (post.content or "").lower()
            hashtags_lower = [h.lower() for h in (post.hashtags or [])]
            for topic in all_topics:
                matched = False
                for keyword in (topic.keywords or []):
                    for term in keyword.lower().split('_'):
                        if len(term) >= 3 and term in content_lower:
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    for tag in (topic.hashtags or []):
                        if tag.lower() in hashtags_lower or tag.lower().replace('#', '') in content_lower:
                            matched = True
                            break
                if matched:
                    if count_dict is not None:
                        count_dict[topic.id] = count_dict.get(topic.id, 0) + 1
                    else:
                        stats = stats_dict[topic.id]
                        stats["count"] += 1
                        stats["total_valence"] += float(enriched.valence or 0)
                        dom_emotion = enriched.dominant_emotion or "neutral"
                        stats["emotions"][dom_emotion] = stats["emotions"].get(dom_emotion, 0) + 1

        for post, enriched in posts_with_enrichment:
            match_post_to_topics(post, enriched, topic_stats)

        for post, enriched in prev_posts:
            match_post_to_topics(post, enriched, topic_stats, prev_topic_counts)

        # Build topic list
        emotion_colors = {
            "joy": "hsl(85, 70%, 60%)", "anger": "hsl(25, 80%, 55%)",
            "fear": "hsl(295, 60%, 55%)", "trust": "hsl(230, 70%, 60%)",
            "surprise": "hsl(180, 60%, 60%)", "sadness": "hsl(210, 60%, 50%)",
            "disgust": "hsl(150, 50%, 45%)", "anticipation": "hsl(45, 80%, 60%)",
        }

        built_topics = []
        for topic_id_key, stats in topic_stats.items():
            if stats["count"] == 0:
                continue
            topic = stats["topic"]
            avg_valence = stats["total_valence"] / stats["count"]
            avg_sentiment = int(((avg_valence + 1) / 2) * 100)
            avg_sentiment = max(0, min(100, avg_sentiment))

            dom_emotion = "trust"
            if stats["emotions"]:
                dom_emotion = max(stats["emotions"], key=stats["emotions"].get)

            # Trend vs previous period
            prev_count = prev_topic_counts.get(topic_id_key, 0)
            if prev_count > 0:
                trend_pct = round(((stats["count"] - prev_count) / prev_count) * 100, 1)
            else:
                trend_pct = 100.0 if stats["count"] > 0 else 0.0

            built_topics.append({
                "id": topic.id,
                "name": topic.topic_name.replace("_", " ").title(),
                "slug": topic.topic_name.lower().replace(" ", "-"),
                "mentions": stats["count"],
                "sentiment": avg_sentiment,
                "sentimentScore": avg_sentiment,
                "trend": f"+{trend_pct}%" if trend_pct >= 0 else f"{trend_pct}%",
                "trendPercent": trend_pct,
                "dominantEmotion": dom_emotion,
                "emotionColor": emotion_colors.get(dom_emotion, "hsl(var(--muted))"),
                "description": topic.description or f"Discussions about {topic.topic_name.replace('_', ' ')}",
                "keywords": topic.keywords or [],
                "lastUpdated": datetime.utcnow().isoformat()
            })

        # Apply sorting
        sort_field_map = {
            "mentions": "mentions",
            "sentiment": "sentiment",
            "name": "name",
            "trend": "trendPercent"
        }
        actual_sort_field = sort_field_map.get(sort_by, "mentions")
        reverse = (order == "desc")
        built_topics.sort(
            key=lambda x: x[actual_sort_field] if isinstance(x[actual_sort_field], (int, float)) else 0,
            reverse=reverse
        )

        total_count = len(built_topics)
        paginated_topics = built_topics[offset:offset + limit]

        response = {
            "topics": paginated_topics,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total_count
            },
            "filters": {
                "sortBy": sort_by,
                "order": order,
                "search": search,
                "daysBack": days_back
            },
            "metadata": {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "daysBack": days_back
            }
        }

        logger.info(
            "topics_list_retrieved",
            total_topics=total_count,
            returned=len(paginated_topics),
            sort_by=sort_by,
            days_back=days_back
        )

        return response

    except Exception as e:
        logger.error(f"Error fetching topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch topics")


@router.get("/topics/{topic_id}")
async def get_topic_details(
    topic_id: int,
    days_back: int = Query(7, ge=1, le=90, description="Days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific topic.

    Returns topic info with sentiment trends, emotion distribution, top posts,
    and hourly distribution — all filtered to the requested time range.
    """
    start_date, end_date = get_time_range(days_back)

    # Fetch topic from database
    topic_query = select(Topic).where(Topic.id == topic_id)
    topic_result = await db.execute(topic_query)
    topic = topic_result.scalar_one_or_none()

    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")

    try:
        # Helper: check if a post matches this topic by keywords/hashtags
        topic_keywords = topic.keywords or []
        topic_hashtags = topic.hashtags or []

        def post_matches_topic(content: str, hashtags: list) -> bool:
            content_lower = (content or "").lower()
            hashtags_lower = [h.lower() for h in (hashtags or [])]
            for keyword in topic_keywords:
                for term in keyword.lower().split('_'):
                    if len(term) >= 3 and term in content_lower:
                        return True
            for tag in topic_hashtags:
                if tag.lower() in hashtags_lower or tag.lower().replace('#', '') in content_lower:
                    return True
            return False

        # --- Fetch all posts with enrichment in time range ---
        posts_query = (
            select(PostRaw, PostEnriched)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(and_(
                PostRaw.posted_at >= start_date,
                PostRaw.posted_at <= end_date
            ))
        )
        posts_result = await db.execute(posts_query)
        all_posts = posts_result.all()

        # Filter to only posts matching this topic
        matched_posts = [
            (post, enriched) for post, enriched in all_posts
            if post_matches_topic(post.content, post.hashtags)
        ]

        # --- Sentiment trends grouped by day ---
        from collections import defaultdict
        daily_stats = defaultdict(lambda: {"valence_sum": 0.0, "count": 0})
        for post, enriched in matched_posts:
            if post.posted_at:
                day = post.posted_at.date()
                daily_stats[day]["valence_sum"] += float(enriched.valence or 0)
                daily_stats[day]["count"] += 1

        sentiment_trends = []
        total_mentions = 0
        total_valence = 0.0
        for day in sorted(daily_stats.keys()):
            stats = daily_stats[day]
            avg_val = stats["valence_sum"] / stats["count"] if stats["count"] > 0 else 0
            sentiment_score = int(((avg_val + 1) / 2) * 100)
            sentiment_trends.append({
                "date": day.strftime("%b %d"),
                "fullDate": day.isoformat(),
                "sentiment": max(0, min(100, sentiment_score)),
                "mentions": stats["count"]
            })
            total_mentions += stats["count"]
            total_valence += stats["valence_sum"]

        avg_sentiment = int(((total_valence / total_mentions + 1) / 2) * 100) if total_mentions > 0 else 50
        avg_sentiment = max(0, min(100, avg_sentiment))

        # --- Emotion distribution ---
        emotion_sums = {e: 0.0 for e in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]}
        emotion_field_map = {
            "joy": "emotion_joy", "trust": "emotion_trust", "fear": "emotion_fear",
            "surprise": "emotion_surprise", "sadness": "emotion_sadness",
            "disgust": "emotion_disgust", "anger": "emotion_anger",
            "anticipation": "emotion_anticipation"
        }
        for _, enriched in matched_posts:
            for emotion_name, field_name in emotion_field_map.items():
                emotion_sums[emotion_name] += float(getattr(enriched, field_name, 0) or 0)

        if total_mentions > 0:
            raw_emotions = {k: v / total_mentions for k, v in emotion_sums.items()}
            total_emotion = sum(raw_emotions.values()) or 1
            emotions = {k: round((v / total_emotion) * 100, 1) for k, v in raw_emotions.items()}
            dom_emotion = max(raw_emotions, key=raw_emotions.get)
        else:
            emotions = {e: 0.0 for e in emotion_sums}
            dom_emotion = "neutral"

        emotion_colors = {
            "joy": "hsl(85, 70%, 60%)", "anger": "hsl(25, 80%, 55%)",
            "fear": "hsl(295, 60%, 55%)", "trust": "hsl(230, 70%, 60%)",
            "surprise": "hsl(180, 60%, 60%)", "sadness": "hsl(210, 60%, 50%)",
            "disgust": "hsl(150, 50%, 45%)", "anticipation": "hsl(45, 80%, 60%)",
        }

        # --- Top posts (sorted by engagement) ---
        matched_posts_sorted = sorted(
            matched_posts,
            key=lambda x: float(x[1].engagement_score or 0),
            reverse=True
        )[:5]
        top_posts = []
        for post, enriched in matched_posts_sorted:
            sentiment_score = int(((float(enriched.valence or 0) + 1) / 2) * 100)
            top_posts.append({
                "id": str(post.id),
                "text": (post.content or "")[:200],
                "author": post.user_id,
                "sentiment": max(0, min(100, sentiment_score)),
                "engagement": float(enriched.engagement_score or 0),
                "timestamp": post.posted_at.isoformat() if post.posted_at else None
            })

        # --- Hourly distribution ---
        hourly_counts = defaultdict(int)
        for post, _ in matched_posts:
            if post.posted_at:
                hourly_counts[post.posted_at.hour] += 1
        time_distribution = [
            {"hour": f"{h:02d}:00", "mentions": hourly_counts.get(h, 0)}
            for h in range(24)
            if hourly_counts.get(h, 0) > 0
        ]

        # --- Trend vs previous period ---
        prev_start = start_date - timedelta(days=days_back)
        prev_posts_query = (
            select(PostRaw)
            .where(and_(
                PostRaw.posted_at >= prev_start,
                PostRaw.posted_at <= start_date
            ))
        )
        prev_posts_result = await db.execute(prev_posts_query)
        prev_posts = prev_posts_result.scalars().all()
        prev_count = sum(
            1 for p in prev_posts
            if post_matches_topic(p.content, p.hashtags)
        )
        if prev_count > 0:
            trend_pct = round(((total_mentions - prev_count) / prev_count) * 100, 1)
        else:
            trend_pct = 100.0 if total_mentions > 0 else 0.0

        topic_details = {
            "id": topic.id,
            "name": topic.topic_name.replace("_", " ").title(),
            "slug": topic.topic_name.lower().replace(" ", "-"),
            "description": topic.description or f"Discussions about {topic.topic_name.replace('_', ' ')}",
            "mentions": total_mentions,
            "sentiment": avg_sentiment,
            "trend": f"+{trend_pct}%" if trend_pct >= 0 else f"{trend_pct}%",
            "trendPercent": trend_pct,
            "dominantEmotion": dom_emotion,
            "emotionColor": emotion_colors.get(dom_emotion, "hsl(var(--muted))"),
            "keywords": topic.keywords or [],
            "sentimentTrends": sentiment_trends,
            "emotions": emotions,
            "topPosts": top_posts,
            "timeDistribution": time_distribution,
            "lastUpdated": datetime.utcnow().isoformat(),
            "metadata": {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "daysBack": days_back
            }
        }

        logger.info(
            "topic_details_retrieved",
            topic_id=topic_id,
            mentions=total_mentions,
            sentiment=avg_sentiment,
            days_back=days_back
        )

        return topic_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching topic details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch topic details")


@router.get("/topics/{topic_id}/keywords")
async def get_topic_keywords(
    topic_id: int,
    days_back: int = Query(7, ge=1, le=90, description="Days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get keyword/sub-topic breakdown for a specific topic.

    Returns top 10 keywords/sub-topics with:
    - Keyword name
    - Mention count
    - Sentiment percentage
    - Trend
    - Dominant emotion
    """
    import random

    try:
        # First, get the topic to find its name
        topic_query = select(Topic).where(Topic.id == topic_id)
        topic_result = await db.execute(topic_query)
        topic = topic_result.scalar_one_or_none()

        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")

        raw_topic_name = topic.topic_name
        topic_name = topic.topic_name.lower().replace("_", " ")
        logger.info(f"Looking up keywords for topic: '{raw_topic_name}' -> normalized: '{topic_name}' (ID: {topic_id})")

        # Find keywords for this topic - try exact match first, then partial
        matched_key = None
        keywords_list = TOPIC_KEYWORDS.get(topic_name)
        if keywords_list:
            matched_key = topic_name
            logger.info(f"Exact match found for '{topic_name}'")

        if not keywords_list:
            # Try partial match
            for key in TOPIC_KEYWORDS:
                if key in topic_name or topic_name in key:
                    keywords_list = TOPIC_KEYWORDS[key]
                    matched_key = key
                    logger.info(f"Partial match: '{topic_name}' matched with '{key}'")
                    break

        if not keywords_list:
            # Return empty if no keywords defined for this topic
            logger.warning(f"No keywords found for topic: '{topic_name}'")
            return {
                "topicId": topic_id,
                "topicName": topic.topic_name.replace("_", " ").title(),
                "rawTopicName": raw_topic_name,
                "keywords": [],
                "debug": {
                    "normalizedName": topic_name,
                    "matchedKey": None
                }
            }

        # Get time range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        # Count keyword occurrences in posts
        posts_query = (
            select(PostRaw.content, PostEnriched.valence, PostEnriched.dominant_emotion)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(PostRaw.posted_at >= start_date)
            .where(PostRaw.posted_at <= end_date)
        )
        posts_result = await db.execute(posts_query)
        posts = posts_result.all()

        # Count keywords
        keyword_stats: Dict[str, Dict[str, Any]] = {}
        for keyword in keywords_list:
            keyword_stats[keyword] = {
                "mentions": 0,
                "total_valence": 0.0,
                "emotions": {}
            }

        for post_content, valence, dominant_emotion in posts:
            content_lower = (post_content or "").lower()
            for keyword in keywords_list:
                # Keywords may be underscore-separated (e.g., "elections_voting")
                # so we split and check if any term matches
                keyword_lower = keyword.lower()
                terms = keyword_lower.split('_')
                matched = False
                for term in terms:
                    if len(term) >= 3 and term in content_lower:  # Minimum 3 chars to avoid false positives
                        matched = True
                        break
                if matched:
                    stats = keyword_stats[keyword]
                    stats["mentions"] += 1
                    stats["total_valence"] += float(valence or 0)
                    emotion = dominant_emotion or "neutral"
                    stats["emotions"][emotion] = stats["emotions"].get(emotion, 0) + 1

        # Build response with keyword data
        emotions = ["trust", "joy", "fear", "anticipation", "anger", "sadness", "surprise", "disgust"]
        emotion_colors = {
            "trust": "bg-blue-500",
            "joy": "bg-yellow-500",
            "fear": "bg-purple-500",
            "surprise": "bg-teal-500",
            "sadness": "bg-indigo-500",
            "anger": "bg-orange-500",
            "anticipation": "bg-green-500",
            "disgust": "bg-red-500",
        }

        keywords_response = []
        for i, keyword in enumerate(keywords_list):
            stats = keyword_stats[keyword]
            mentions = stats["mentions"]

            if mentions == 0:
                sentiment = 50
                trend_value = 0.0
                dominant_emotion = emotions[i % len(emotions)]
            else:
                avg_valence = stats["total_valence"] / mentions if mentions > 0 else 0
                sentiment = int((avg_valence + 1) * 50)
                sentiment = max(0, min(100, sentiment))
                trend_value = round(random.uniform(-5, 15), 1)  # Mock trend for now
                dominant_emotion = max(stats["emotions"], key=stats["emotions"].get) if stats["emotions"] else emotions[i % len(emotions)]

            keywords_response.append({
                "keyword": keyword.capitalize(),
                "mentions": mentions,
                "sentiment": sentiment,
                "trend": f"+{trend_value}%" if trend_value >= 0 else f"{trend_value}%",
                "dominantEmotion": dominant_emotion,
                "emotionColor": emotion_colors.get(dominant_emotion, "bg-blue-500")
            })

        # Sort by mentions descending
        keywords_response.sort(key=lambda x: x["mentions"], reverse=True)

        logger.info(
            "topic_keywords_retrieved",
            topic_id=topic_id,
            topic_name=topic_name,
            matched_key=matched_key,
            keywords_count=len(keywords_response)
        )

        return {
            "topicId": topic_id,
            "topicName": topic.topic_name.replace("_", " ").title(),
            "rawTopicName": raw_topic_name,
            "keywords": keywords_response[:10],
            "debug": {
                "normalizedName": topic_name,
                "matchedKey": matched_key
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic keywords: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch topic keywords")


@router.get("/topics/{topic_id}/creators")
async def get_topic_creators(
    topic_id: int,
    days_back: int = Query(30, ge=1, le=90, description="Days to analyze"),
    limit: int = Query(10, ge=1, le=50, description="Number of creators to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get top creators for a topic ranked by follower count.

    Returns each creator's engagement rate, sentiment score, post frequency,
    and 8-emotion Plutchik profile aggregated from their enriched posts.
    """
    start_date, end_date = get_time_range(days_back)

    # Fetch topic
    topic_query = select(Topic).where(Topic.id == topic_id)
    topic_result = await db.execute(topic_query)
    topic = topic_result.scalar_one_or_none()

    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")

    try:
        topic_keywords = topic.keywords or []
        topic_hashtags = topic.hashtags or []

        def post_matches_topic(content: str, hashtags: list) -> bool:
            content_lower = (content or "").lower()
            hashtags_lower = [h.lower() for h in (hashtags or [])]
            for keyword in topic_keywords:
                for term in keyword.lower().split('_'):
                    if len(term) >= 3 and term in content_lower:
                        return True
            for tag in topic_hashtags:
                if tag.lower() in hashtags_lower or tag.lower().replace('#', '') in content_lower:
                    return True
            return False

        # Fetch all posts with enrichment and user data in time range
        posts_query = (
            select(PostRaw, PostEnriched, User)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .join(User, PostRaw.user_id == User.user_id)
            .where(and_(
                PostRaw.posted_at >= start_date,
                PostRaw.posted_at <= end_date
            ))
        )
        posts_result = await db.execute(posts_query)
        all_posts = posts_result.all()

        # Filter to posts matching this topic and group by user
        emotion_fields = [
            "emotion_joy", "emotion_trust", "emotion_fear", "emotion_surprise",
            "emotion_sadness", "emotion_disgust", "emotion_anger", "emotion_anticipation"
        ]

        user_data: Dict[str, Dict[str, Any]] = {}

        for post, enriched, user in all_posts:
            if not post_matches_topic(post.content, post.hashtags):
                continue

            uid = user.user_id
            if uid not in user_data:
                user_data[uid] = {
                    "user_id": uid,
                    "username": user.username,
                    "display_name": user.display_name or user.username,
                    "followers_count": user.followers_count or 0,
                    "post_count": 0,
                    "valence_sum": 0.0,
                    "engagement_sum": 0.0,
                    "emotion_sums": {f: 0.0 for f in emotion_fields},
                }

            d = user_data[uid]
            d["post_count"] += 1
            d["valence_sum"] += float(enriched.valence or 0)
            d["engagement_sum"] += float(post.engagement_score or 0)
            for f in emotion_fields:
                d["emotion_sums"][f] += float(getattr(enriched, f, 0) or 0)

        # Sort by followers_count desc, take top N
        sorted_creators = sorted(user_data.values(), key=lambda x: x["followers_count"], reverse=True)[:limit]

        creators = []
        for d in sorted_creators:
            n = d["post_count"]
            avg_valence = d["valence_sum"] / n if n > 0 else 0
            sentiment_score = round(((avg_valence + 1) / 2) * 100, 1)
            sentiment_score = max(0, min(100, sentiment_score))

            avg_engagement = d["engagement_sum"] / n if n > 0 else 0
            followers = d["followers_count"] or 1
            engagement_rate = round((avg_engagement / followers) * 100, 2)

            # Normalize emotions to percentages
            raw_emotions = {}
            for f in emotion_fields:
                emotion_name = f.replace("emotion_", "")
                raw_emotions[emotion_name] = d["emotion_sums"][f] / n if n > 0 else 0
            total_emotion = sum(raw_emotions.values()) or 1
            emotion_profile = {k: round((v / total_emotion) * 100, 1) for k, v in raw_emotions.items()}

            creators.append({
                "user_id": d["user_id"],
                "username": d["username"],
                "display_name": d["display_name"],
                "followers_count": d["followers_count"],
                "post_count": n,
                "post_frequency": round(n / days_back, 2),
                "sentiment_score": sentiment_score,
                "engagement_rate": engagement_rate,
                "emotion_profile": emotion_profile,
            })

        logger.info(
            "topic_creators_retrieved",
            topic_id=topic_id,
            creators_count=len(creators),
            days_back=days_back
        )

        return {
            "topicId": topic_id,
            "topicName": topic.topic_name.replace("_", " ").title(),
            "creators": creators,
            "metadata": {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "daysBack": days_back
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching topic creators: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch topic creators")


@router.get("/topics/{topic_id}/creators/{user_id}/timeseries")
async def get_creator_sentiment_timeseries(
    topic_id: int,
    user_id: str,
    days_back: int = Query(30, ge=1, le=90, description="Days to analyze"),
    bucket: str = Query("day", description="Aggregation bucket: day or week"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get sentiment timeseries for a specific creator within a topic.

    Returns daily (or weekly) sentiment scores and emotion profiles
    for tracking how a creator's EQ changes over time.
    """
    from collections import defaultdict

    start_date, end_date = get_time_range(days_back)

    # Fetch topic
    topic_query = select(Topic).where(Topic.id == topic_id)
    topic_result = await db.execute(topic_query)
    topic = topic_result.scalar_one_or_none()

    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")

    try:
        topic_keywords = topic.keywords or []
        topic_hashtags = topic.hashtags or []

        def post_matches_topic(content: str, hashtags: list) -> bool:
            content_lower = (content or "").lower()
            hashtags_lower = [h.lower() for h in (hashtags or [])]
            for keyword in topic_keywords:
                for term in keyword.lower().split('_'):
                    if len(term) >= 3 and term in content_lower:
                        return True
            for tag in topic_hashtags:
                if tag.lower() in hashtags_lower or tag.lower().replace('#', '') in content_lower:
                    return True
            return False

        # Fetch posts for this user in time range
        posts_query = (
            select(PostRaw, PostEnriched)
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(and_(
                PostRaw.user_id == user_id,
                PostRaw.posted_at >= start_date,
                PostRaw.posted_at <= end_date
            ))
        )
        posts_result = await db.execute(posts_query)
        all_posts = posts_result.all()

        # Filter to topic-matching posts and group by bucket
        emotion_fields = [
            "emotion_joy", "emotion_trust", "emotion_fear", "emotion_surprise",
            "emotion_sadness", "emotion_disgust", "emotion_anger", "emotion_anticipation"
        ]

        bucket_data = defaultdict(lambda: {
            "valence_sum": 0.0, "count": 0,
            "emotion_sums": {f: 0.0 for f in emotion_fields}
        })

        for post, enriched in all_posts:
            if not post_matches_topic(post.content, post.hashtags):
                continue
            if not post.posted_at:
                continue

            if bucket == "week":
                # ISO week start (Monday)
                day = post.posted_at.date()
                key = (day - timedelta(days=day.weekday())).isoformat()
            else:
                key = post.posted_at.date().isoformat()

            bd = bucket_data[key]
            bd["count"] += 1
            bd["valence_sum"] += float(enriched.valence or 0)
            for f in emotion_fields:
                bd["emotion_sums"][f] += float(getattr(enriched, f, 0) or 0)

        # Build timeseries
        timeseries = []
        for date_key in sorted(bucket_data.keys()):
            bd = bucket_data[date_key]
            n = bd["count"]
            avg_valence = bd["valence_sum"] / n if n > 0 else 0
            sentiment_score = round(((avg_valence + 1) / 2) * 100, 1)
            sentiment_score = max(0, min(100, sentiment_score))

            emotions = {}
            for f in emotion_fields:
                emotion_name = f.replace("emotion_", "")
                emotions[emotion_name] = round(bd["emotion_sums"][f] / n, 4) if n > 0 else 0

            timeseries.append({
                "date": date_key,
                "sentiment_score": sentiment_score,
                "post_count": n,
                "emotions": emotions,
            })

        logger.info(
            "creator_timeseries_retrieved",
            topic_id=topic_id,
            user_id=user_id,
            data_points=len(timeseries),
            days_back=days_back
        )

        return {
            "topicId": topic_id,
            "topicName": topic.topic_name.replace("_", " ").title(),
            "userId": user_id,
            "bucket": bucket,
            "timeseries": timeseries,
            "metadata": {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "daysBack": days_back
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching creator timeseries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch creator timeseries")
