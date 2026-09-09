"""
Brand Analysis API Endpoints

Brand monitoring powered by 27-emotion analysis.
A "brand" is a Topic with category='brand'. Posts are matched by
searching content against brand keywords.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text, case
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import structlog

from src.database.connection import get_db
from src.database.models import Topic, PostRaw, PostEnriched

logger = structlog.get_logger(__name__)

router = APIRouter()

BRAND_CATEGORY = "brand"

# Language → Region mapping for regional sentiment analysis
LANGUAGE_REGION_MAP = {
    "en": "North America",
    "es": "South America", "pt": "South America",
    "fr": "Europe", "de": "Europe", "it": "Europe", "nl": "Europe",
    "pl": "Europe", "sv": "Europe", "no": "Europe", "da": "Europe",
    "fi": "Europe", "ro": "Europe", "cs": "Europe", "hu": "Europe",
    "el": "Europe", "uk": "Europe", "bg": "Europe", "hr": "Europe",
    "ar": "Middle East & Africa", "he": "Middle East & Africa",
    "tr": "Middle East & Africa", "fa": "Middle East & Africa",
    "sw": "Middle East & Africa", "am": "Middle East & Africa",
    "hi": "Asia", "ta": "Asia", "te": "Asia", "bn": "Asia",
    "zh-cn": "Asia", "zh-tw": "Asia", "ja": "Asia", "ko": "Asia",
    "id": "Asia", "th": "Asia", "vi": "Asia", "ms": "Asia",
    "tl": "Asia", "mr": "Asia", "gu": "Asia",
}

REGIONS = ["North America", "Europe", "Asia", "South America", "Middle East & Africa"]
REGION_COLORS = {
    "North America": "oklch(0.65 0.18 230)",
    "Europe": "oklch(0.70 0.16 160)",
    "Asia": "oklch(0.72 0.20 60)",
    "South America": "oklch(0.68 0.18 120)",
    "Middle East & Africa": "oklch(0.66 0.14 300)",
}


def _brand_content_filter(brand: Topic):
    """Build a content filter matching the brand name OR specific keywords.

    Uses AND logic for generic terms (must co-occur with brand name) and
    OR logic for brand-specific terms (product names, handles).
    Generic terms like 'AI', 'tech', 'crypto' are too broad on their own.
    """
    brand_name = brand.topic_name
    keywords = brand.keywords or []

    # The brand name itself always matches
    name_match = PostRaw.content.ilike(f"%{brand_name}%")

    if not keywords:
        return name_match

    # Filter out very short/generic keywords that would match too broadly
    generic_terms = {"ai", "tech", "crypto", "blockchain", "ml", "llm", "app", "api", "bot"}
    specific_keywords = [kw for kw in keywords if kw.lower() not in generic_terms and len(kw) > 2]
    generic_keywords = [kw for kw in keywords if kw.lower() in generic_terms or len(kw) <= 2]

    # Specific keywords (product names, handles) match on their own via OR
    conditions = [name_match]
    for kw in specific_keywords:
        conditions.append(PostRaw.content.ilike(f"%{kw}%"))

    # Generic keywords only match when combined with the brand name
    for kw in generic_keywords:
        conditions.append(and_(
            PostRaw.content.ilike(f"%{brand_name}%"),
            PostRaw.content.ilike(f"%{kw}%"),
        ))

    return or_(*conditions)


class BrandCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    keywords: List[str] = Field(..., min_items=1, description="Search terms for this brand")
    hashtags: Optional[List[str]] = None


class BrandUpdateRequest(BaseModel):
    keywords: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    is_active: Optional[bool] = None


@router.post("")
async def create_brand(
    request: BrandCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new brand to monitor."""
    existing = await db.execute(
        select(Topic).where(and_(
            Topic.topic_name == request.name,
            Topic.category == BRAND_CATEGORY,
        ))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Brand '{request.name}' already exists")

    brand = Topic(
        topic_name=request.name,
        category=BRAND_CATEGORY,
        keywords=request.keywords,
        hashtags=request.hashtags,
        is_active=True,
    )
    db.add(brand)
    await db.commit()
    await db.refresh(brand)

    return {"id": brand.id, "name": brand.topic_name, "keywords": brand.keywords, "created": True}


@router.get("")
async def list_brands(db: AsyncSession = Depends(get_db)):
    """List all monitored brands with 7-day summary metrics."""
    brands_q = select(Topic).where(Topic.category == BRAND_CATEGORY).order_by(Topic.topic_name)
    result = await db.execute(brands_q)
    brands = result.scalars().all()

    brand_list = []
    for brand in brands:
        start = datetime.utcnow() - timedelta(days=7)
        content_filter = _brand_content_filter(brand)

        try:
            stats_q = (
                select(
                    func.count(PostRaw.id).label("mention_count"),
                    func.avg(PostEnriched.valence).label("avg_valence"),
                    func.avg(PostEnriched.toxicity_score).label("avg_toxicity"),
                    func.avg(PostEnriched.empathic_concern).label("avg_empathy"),
                )
                .join(PostEnriched, PostRaw.id == PostEnriched.post_id, isouter=True)
                .where(and_(content_filter, PostRaw.posted_at >= start))
            )
            stats = (await db.execute(stats_q)).one()

            # Dominant non-neutral emotion
            dom_q = (
                select(PostEnriched.dominant_emotion, func.count().label("cnt"))
                .join(PostRaw, PostEnriched.post_id == PostRaw.id)
                .where(and_(
                    content_filter,
                    PostRaw.posted_at >= start,
                    PostEnriched.dominant_emotion.isnot(None),
                    PostEnriched.dominant_emotion != "neutral",
                ))
                .group_by(PostEnriched.dominant_emotion)
                .order_by(desc("cnt"))
                .limit(1)
            )
            dom_row = (await db.execute(dom_q)).first()

            brand_list.append({
                "id": brand.id,
                "name": brand.topic_name,
                "keywords": brand.keywords,
                "hashtags": brand.hashtags,
                "is_active": brand.is_active,
                "created_at": brand.created_at,
                "mention_count_7d": stats.mention_count or 0,
                "avg_valence_7d": round(float(stats.avg_valence or 0), 4),
                "avg_toxicity_7d": round(float(stats.avg_toxicity or 0), 4),
                "avg_empathy_7d": round(float(stats.avg_empathy or 0), 4),
                "dominant_emotion": dom_row.dominant_emotion if dom_row else "neutral",
            })
        except Exception as e:
            logger.warning("brand_metrics_error", brand=brand.topic_name, error=str(e))
            brand_list.append({
                "id": brand.id, "name": brand.topic_name,
                "keywords": brand.keywords, "hashtags": brand.hashtags,
                "is_active": brand.is_active, "created_at": brand.created_at,
                "mention_count_7d": 0, "avg_valence_7d": 0, "avg_toxicity_7d": 0,
                "avg_empathy_7d": 0, "dominant_emotion": "neutral",
            })

    return {"brands": brand_list, "total": len(brand_list)}


@router.get("/{brand_id}")
async def get_brand_detail(
    brand_id: int,
    days_back: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Full brand detail with 27-emotion fingerprint."""
    brand = await db.get(Topic, brand_id)
    if not brand or brand.category != BRAND_CATEGORY:
        raise HTTPException(status_code=404, detail="Brand not found")

    start = datetime.utcnow() - timedelta(days=days_back)
    content_filter = _brand_content_filter(brand)

    metrics_q = (
        select(
            func.count(PostRaw.id).label("total_mentions"),
            func.count(func.distinct(PostRaw.user_id)).label("unique_authors"),
            func.avg(PostEnriched.valence).label("avg_valence"),
            func.avg(PostEnriched.arousal).label("avg_arousal"),
            func.avg(PostEnriched.intensity).label("avg_intensity"),
            func.avg(PostEnriched.toxicity_score).label("avg_toxicity"),
            func.avg(PostEnriched.safety_concern_score).label("avg_safety"),
            func.avg(PostEnriched.empathic_concern).label("avg_empathy"),
            func.avg(PostEnriched.wonder_index).label("avg_wonder"),
        )
        .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
        .where(and_(content_filter, PostRaw.posted_at >= start))
    )
    m = (await db.execute(metrics_q)).one()

    # 27-emotion fingerprint
    emo_q = (
        select(PostEnriched.raw_27_emotions)
        .join(PostRaw, PostEnriched.post_id == PostRaw.id)
        .where(and_(content_filter, PostRaw.posted_at >= start, PostEnriched.raw_27_emotions.isnot(None)))
        .limit(5000)
    )
    emo_rows = (await db.execute(emo_q)).all()

    emotion_totals: dict = {}
    emotion_counts: dict = {}
    for (raw_27,) in emo_rows:
        if not raw_27 or not isinstance(raw_27, dict):
            continue
        for emo, score in raw_27.items():
            if isinstance(score, (int, float)):
                emotion_totals[emo] = emotion_totals.get(emo, 0) + float(score)
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    emotion_fingerprint = {
        emo: round(emotion_totals[emo] / emotion_counts[emo], 4)
        for emo in sorted(emotion_totals, key=lambda e: emotion_totals[e] / max(emotion_counts[e], 1), reverse=True)
        if emotion_counts.get(emo, 0) > 0
    }

    return {
        "id": brand.id, "name": brand.topic_name,
        "keywords": brand.keywords, "hashtags": brand.hashtags,
        "days_analyzed": days_back,
        "metrics": {
            "total_mentions": m.total_mentions,
            "unique_authors": m.unique_authors,
            "avg_valence": round(float(m.avg_valence or 0), 4),
            "avg_arousal": round(float(m.avg_arousal or 0), 4),
            "avg_intensity": round(float(m.avg_intensity or 0), 4),
            "avg_toxicity": round(float(m.avg_toxicity or 0), 4),
            "avg_safety": round(float(m.avg_safety or 0), 4),
            "avg_empathy": round(float(m.avg_empathy or 0), 4),
            "avg_wonder": round(float(m.avg_wonder or 0), 4),
        },
        "emotion_fingerprint": emotion_fingerprint,
    }


@router.put("/{brand_id}")
async def update_brand(brand_id: int, request: BrandUpdateRequest, db: AsyncSession = Depends(get_db)):
    """Update brand keywords, hashtags, or active status."""
    brand = await db.get(Topic, brand_id)
    if not brand or brand.category != BRAND_CATEGORY:
        raise HTTPException(status_code=404, detail="Brand not found")
    if request.keywords is not None:
        brand.keywords = request.keywords
    if request.hashtags is not None:
        brand.hashtags = request.hashtags
    if request.is_active is not None:
        brand.is_active = request.is_active
    await db.commit()
    return {"id": brand.id, "name": brand.topic_name, "updated": True}


@router.delete("/{brand_id}")
async def delete_brand(brand_id: int, db: AsyncSession = Depends(get_db)):
    """Archive a brand."""
    brand = await db.get(Topic, brand_id)
    if not brand or brand.category != BRAND_CATEGORY:
        raise HTTPException(status_code=404, detail="Brand not found")
    brand.is_active = False
    brand.category = "brand_archived"
    await db.commit()
    return {"id": brand.id, "name": brand.topic_name, "archived": True}


@router.get("/{brand_id}/compare/{competitor_id}")
async def compare_brands(
    brand_id: int, competitor_id: int,
    days_back: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Side-by-side brand vs competitor comparison."""
    brand = await db.get(Topic, brand_id)
    competitor = await db.get(Topic, competitor_id)
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")

    start = datetime.utcnow() - timedelta(days=days_back)

    async def get_metrics(topic: Topic):
        cf = _brand_content_filter(topic)
        q = (
            select(
                func.count(PostRaw.id).label("mentions"),
                func.avg(PostEnriched.valence).label("valence"),
                func.avg(PostEnriched.arousal).label("arousal"),
                func.avg(PostEnriched.toxicity_score).label("toxicity"),
                func.avg(PostEnriched.empathic_concern).label("empathy"),
                func.avg(PostEnriched.safety_concern_score).label("safety"),
                func.avg(PostEnriched.wonder_index).label("wonder"),
            )
            .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
            .where(and_(cf, PostRaw.posted_at >= start))
        )
        r = (await db.execute(q)).one()
        return {
            "name": topic.topic_name, "mentions": r.mentions,
            "valence": round(float(r.valence or 0), 4),
            "arousal": round(float(r.arousal or 0), 4),
            "toxicity": round(float(r.toxicity or 0), 4),
            "empathy": round(float(r.empathy or 0), 4),
            "safety": round(float(r.safety or 0), 4),
            "wonder": round(float(r.wonder or 0), 4),
        }

    bm = await get_metrics(brand)
    cm = await get_metrics(competitor)

    diffs = []
    for m in ["valence", "toxicity", "empathy", "wonder"]:
        if abs(bm[m] - cm[m]) > 0.05:
            diffs.append({"metric": m, "winner": brand.topic_name if bm[m] > cm[m] else competitor.topic_name,
                          "brand_value": bm[m], "competitor_value": cm[m], "difference": round(abs(bm[m] - cm[m]), 4)})

    return {"days_analyzed": days_back, "brand": bm, "competitor": cm, "differentiators": diffs}


@router.get("/{brand_id}/regional-sentiment")
async def get_regional_sentiment(
    brand_id: int,
    days_back: int = Query(30, ge=1, le=180),
    db: AsyncSession = Depends(get_db),
):
    """
    Sentiment by region over time.

    Uses detected language as a proxy for geographic region.
    Returns time-bucketed sentiment metrics per region (North America, Europe,
    Asia, South America, Middle East & Africa).
    """
    brand = await db.get(Topic, brand_id)
    if not brand or brand.category != BRAND_CATEGORY:
        raise HTTPException(status_code=404, detail="Brand not found")

    start = datetime.utcnow() - timedelta(days=days_back)
    content_filter = _brand_content_filter(brand)

    # Build region CASE expression from language
    region_cases = []
    for lang, region in LANGUAGE_REGION_MAP.items():
        region_cases.append((PostRaw.language == lang, region))

    region_expr = case(*region_cases, else_="Other")

    q = (
        select(
            func.date_trunc("week", PostRaw.posted_at).label("period"),
            region_expr.label("region"),
            func.count(PostRaw.id).label("post_count"),
            func.avg(PostEnriched.valence).label("avg_valence"),
            func.avg(PostEnriched.arousal).label("avg_arousal"),
            func.avg(PostEnriched.toxicity_score).label("avg_toxicity"),
            func.avg(PostEnriched.empathic_concern).label("avg_empathy"),
        )
        .join(PostEnriched, PostRaw.id == PostEnriched.post_id)
        .where(and_(
            content_filter,
            PostRaw.posted_at >= start,
            PostRaw.language.isnot(None),
        ))
        .group_by("period", "region")
        .order_by("period")
    )

    result = await db.execute(q)
    rows = result.all()

    # Group by region
    from collections import defaultdict
    region_data: dict = defaultdict(list)

    for r in rows:
        region_name = r.region if r.region in REGIONS else "Other"
        region_data[region_name].append({
            "date": r.period.isoformat() if r.period else None,
            "post_count": r.post_count,
            "avg_valence": round(float(r.avg_valence or 0), 4),
            "avg_arousal": round(float(r.avg_arousal or 0), 4),
            "avg_toxicity": round(float(r.avg_toxicity or 0), 4),
            "avg_empathy": round(float(r.avg_empathy or 0), 4),
        })

    regions = [
        {
            "region": region,
            "color": REGION_COLORS.get(region, "#888"),
            "total_posts": sum(p["post_count"] for p in points),
            "timeseries": points,
        }
        for region, points in region_data.items()
        if points
    ]

    # Sort by total posts
    regions.sort(key=lambda r: r["total_posts"], reverse=True)

    return {
        "brand_id": brand_id,
        "brand_name": brand.topic_name,
        "days_analyzed": days_back,
        "regions": regions,
    }
