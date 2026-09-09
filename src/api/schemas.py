"""
Pydantic Schemas for API Request/Response Validation

Provides type-safe models for all API endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class SortOrder(str, Enum):
    """Sort order enum"""
    asc = "asc"
    desc = "desc"


class ArchetypeCategory(str, Enum):
    """Archetype category enum"""
    loyalist = "loyalist"
    agitator = "agitator"
    cynic = "cynic"
    idealist = "idealist"
    analyst = "analyst"
    opportunist = "opportunist"
    influencer = "influencer"
    casual = "casual"
    lurker = "lurker"
    contrarian = "contrarian"


# ============================================================
# USER SCHEMAS
# ============================================================

class UserBase(BaseModel):
    """Base user schema"""
    user_id: str
    username: Optional[str] = None
    platform: str


class UserResponse(UserBase):
    """User response schema"""
    display_name: Optional[str] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    verified: bool = False
    account_created_at: Optional[datetime] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserMetricsResponse(BaseModel):
    """User metrics response"""
    user_id: str
    topic_id: int
    topic_name: str

    # Emotion balance
    avg_valence: float
    avg_arousal: float

    # Emotion distribution
    emotion_distribution: Dict[str, float]
    primary_emotion: str
    secondary_emotion: str

    # Volatility
    valence_stddev: float
    arousal_stddev: float
    emotion_stability: float
    topic_volatility: float

    # Toxicity
    mean_toxicity: float
    mean_hostility: float

    # Engagement
    avg_engagement_score: float
    reaction_bias: float

    # Temporal
    circadian_pattern: List[int]
    peak_activity_hour: int

    # Social influence
    influence_score: float
    follower_log: float

    # Cognitive
    topic_diversity: float
    entropy_score: float
    avg_content_length: float

    # Trust
    trust_level: float
    agreement_rate: float

    # Stats
    post_count: int
    period_start: datetime
    period_end: datetime

    class Config:
        from_attributes = True


class UserPsychProfileResponse(BaseModel):
    """User psychographic profile response"""
    user_id: str
    topic_id: int
    topic_name: str
    cluster_id: str
    archetype_name: str
    archetype_description: Optional[str] = None
    confidence_score: float
    embedding: List[float] = Field(..., description="64-dimensional embedding vector")
    profile_date: datetime
    last_updated: datetime

    class Config:
        from_attributes = True


# ============================================================
# ARCHETYPE SCHEMAS
# ============================================================

class ArchetypeMetadataResponse(BaseModel):
    """Archetype metadata response"""
    id: int
    topic_id: int
    topic_name: str
    cluster_id: int
    archetype_name: str
    description: Optional[str] = None
    typical_emotions: Optional[Dict[str, float]] = None
    avg_toxicity: Optional[float] = None
    avg_influence: Optional[float] = None
    member_count: int
    centroid: List[float] = Field(..., description="64-dimensional centroid vector")
    created_at: datetime
    last_updated: datetime

    class Config:
        from_attributes = True


class ArchetypeListResponse(BaseModel):
    """List of archetypes with pagination"""
    archetypes: List[ArchetypeMetadataResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ArchetypeMembersResponse(BaseModel):
    """Archetype members response"""
    archetype_name: str
    topic_name: str
    members: List[UserPsychProfileResponse]
    total_members: int
    page: int
    page_size: int


# ============================================================
# ANALYTICS SCHEMAS
# ============================================================

class SentimentTrendPoint(BaseModel):
    """Single point in sentiment trend"""
    timestamp: datetime
    avg_valence: float
    avg_arousal: float
    primary_emotion: str
    emotion_distribution: Dict[str, float]
    post_count: int
    avg_toxicity: float


class SentimentTrendsResponse(BaseModel):
    """Sentiment trends over time"""
    topic_id: int
    topic_name: str
    period_start: datetime
    period_end: datetime
    data_points: List[SentimentTrendPoint]
    summary: Dict[str, Any]


class ArchetypeDistributionPoint(BaseModel):
    """Archetype distribution at a point in time"""
    archetype_name: str
    member_count: int
    percentage: float


class ArchetypeDistributionResponse(BaseModel):
    """Archetype distribution response"""
    topic_id: int
    topic_name: str
    timestamp: datetime
    distribution: List[ArchetypeDistributionPoint]
    total_users: int


class ClusterShiftResponse(BaseModel):
    """Cluster shift response"""
    user_id: str
    topic_id: int
    from_cluster_id: Optional[str] = None
    to_cluster_id: str
    shift_timestamp: datetime
    confidence_delta: Optional[float] = None
    trigger_event: Optional[str] = None

    class Config:
        from_attributes = True


class TopUsersResponse(BaseModel):
    """Top users by influence"""
    topic_id: int
    topic_name: str
    metric: str
    users: List[Dict[str, Any]]
    limit: int


# ============================================================
# SIMILARITY SEARCH SCHEMAS
# ============================================================

class SimilaritySearchRequest(BaseModel):
    """Request for similarity search"""
    user_id: str
    topic_id: int
    limit: int = Field(default=20, ge=1, le=100)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class SimilarUserResponse(BaseModel):
    """Similar user response"""
    user_id: str
    username: Optional[str] = None
    similarity_score: float
    archetype_name: str
    influence_score: float
    post_count: int


class SimilaritySearchResponse(BaseModel):
    """Similarity search results"""
    query_user_id: str
    topic_id: int
    topic_name: str
    similar_users: List[SimilarUserResponse]
    total_results: int


# ============================================================
# SAFETY & HOSTILITY SCHEMAS (P0)
# ============================================================

class SafetyDashboardResponse(BaseModel):
    """Aggregate safety metrics across topics"""
    period_start: datetime
    period_end: datetime
    total_posts_analyzed: int
    avg_safety_concern: float = Field(..., description="Average safety concern score 0-1")
    high_safety_concern_count: int = Field(..., description="Posts with safety_concern > 0.7")
    hostility_distribution: Dict[str, int] = Field(
        ..., description="Count by hostility level: {none, low, medium, high}"
    )
    total_escalation_events: int
    total_de_escalations: int
    top_hostile_topics: List[Dict[str, Any]] = Field(
        default_factory=list, description="Topics ranked by avg hostility"
    )


class HostilityAlertItem(BaseModel):
    """A single hostility escalation alert"""
    post_id: int
    external_id: Optional[str] = None
    content_preview: str = Field(..., description="First 200 chars of post")
    platform: str
    posted_at: datetime
    hostility_level: str
    hostility_score: float
    safety_concern_score: float
    escalation_count: int
    de_escalation_detected: bool
    dominant_emotion: Optional[str] = None
    topic: Optional[str] = None


class HostilityAlertsResponse(BaseModel):
    """Recent hostility escalation alerts"""
    alerts: List[HostilityAlertItem]
    total: int
    page: int
    page_size: int


class SafetyTimeseriesPoint(BaseModel):
    """Single point in safety time series"""
    timestamp: datetime
    avg_safety_concern: float
    avg_hostility_score: float
    high_hostility_count: int
    escalation_events: int
    de_escalation_events: int
    post_count: int


class SafetyTimeseriesResponse(BaseModel):
    """Safety metrics over time for a topic"""
    topic_id: int
    topic_name: str
    period_start: datetime
    period_end: datetime
    data_points: List[SafetyTimeseriesPoint]


# ============================================================
# EMPATHY & BREAKTHROUGH SCHEMAS (P1)
# ============================================================

class EmpathyOverviewResponse(BaseModel):
    """Community empathy health overview"""
    period_start: datetime
    period_end: datetime
    total_posts_analyzed: int
    avg_empathic_concern: float
    avg_personal_distress: float
    avg_meta_emotional_score: float
    empathy_health_index: float = Field(..., description="Composite: (empathic_concern - personal_distress + 1) / 2")
    quadrant_distribution: Dict[str, int] = Field(
        ..., description="Counts per quadrant: compassionate, overwhelmed, detached, distressed"
    )


class EmpathyTimeseriesPoint(BaseModel):
    """Single point in empathy time series"""
    timestamp: datetime
    avg_empathic_concern: float
    avg_personal_distress: float
    avg_meta_emotional_score: float
    post_count: int


class EmpathyTimeseriesResponse(BaseModel):
    """Empathy metrics over time for a topic"""
    topic_id: int
    topic_name: str
    period_start: datetime
    period_end: datetime
    data_points: List[EmpathyTimeseriesPoint]


class BreakthroughEvent(BaseModel):
    """A single breakthrough detection event"""
    post_id: int
    external_id: Optional[str] = None
    content_preview: str
    platform: str
    posted_at: datetime
    wonder_index: float
    discovery_level: str
    intensity: float
    dominant_emotion: Optional[str] = None


class BreakthroughsResponse(BaseModel):
    """Real breakthrough events from SDK"""
    breakthroughs: List[BreakthroughEvent]
    total: int
    page: int
    page_size: int
    breakthrough_rate: float = Field(..., description="Percentage of posts that are breakthroughs")


class EmotionSpectrumEntry(BaseModel):
    """Single emotion in the 27-GoEmotions spectrum"""
    emotion: str
    avg_score: float
    post_count: int
    percentage: float


class EmotionSpectrumResponse(BaseModel):
    """Full 27-GoEmotions distribution"""
    period_start: datetime
    period_end: datetime
    total_posts_analyzed: int
    spectrum: List[EmotionSpectrumEntry]


class EmotionSpectrumTimeseriesPoint(BaseModel):
    """GoEmotions distribution at a point in time"""
    timestamp: datetime
    distribution: Dict[str, float]
    post_count: int


class EmotionSpectrumTimeseriesResponse(BaseModel):
    """27-GoEmotions distribution over time"""
    topic_id: Optional[int] = None
    period_start: datetime
    period_end: datetime
    data_points: List[EmotionSpectrumTimeseriesPoint]


class PatternResponse(BaseModel):
    """Detected emotional pattern"""
    id: int
    user_id: str
    pattern_type: str
    name: str
    description: Optional[str] = None
    frequency: int
    confidence: float
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None

    class Config:
        from_attributes = True


class PatternsListResponse(BaseModel):
    """List of detected patterns"""
    patterns: List[PatternResponse]
    total: int
    page: int
    page_size: int


# ============================================================
# GROWTH & BELIEF SCHEMAS (P2)
# ============================================================

class GrowthDimensionResponse(BaseModel):
    """Single growth dimension for a user"""
    dimension: str
    percentage_change: Optional[float] = None
    recent_value: Optional[float] = None
    baseline_value: Optional[float] = None
    is_growing: bool = False
    measured_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserGrowthResponse(BaseModel):
    """User growth tracking across all 5 dimensions"""
    user_id: str
    dimensions: List[GrowthDimensionResponse]
    overall_growth_score: float = Field(..., description="Average growth across dimensions")


class BeliefResponse(BaseModel):
    """Detected belief"""
    id: int
    user_id: str
    belief_type: str
    content: str
    confidence: Optional[float] = None
    detected_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BeliefLandscapeResponse(BaseModel):
    """Aggregate belief detection landscape"""
    period_start: datetime
    period_end: datetime
    total_beliefs: int
    type_distribution: Dict[str, int]
    top_beliefs: List[BeliefResponse]


# ============================================================
# EXPORT SCHEMAS
# ============================================================

class ExportFormat(str, Enum):
    """Export format enum"""
    json = "json"
    csv = "csv"
    parquet = "parquet"


class ExportRequest(BaseModel):
    """Export request"""
    topic_id: Optional[int] = None
    archetype_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: ExportFormat = ExportFormat.json
    include_embeddings: bool = False


class ExportResponse(BaseModel):
    """Export response"""
    export_id: str
    status: str
    download_url: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None


# ============================================================
# PAGINATION SCHEMAS
# ============================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)

    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size"""
        return (self.page - 1) * self.page_size


# ============================================================
# FILTER SCHEMAS
# ============================================================

class DateRangeFilter(BaseModel):
    """Date range filter"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @validator('end_date')
    def end_after_start(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v


class UserFilter(BaseModel):
    """User filtering options"""
    topic_id: Optional[int] = None
    archetype_name: Optional[str] = None
    min_post_count: Optional[int] = Field(default=None, ge=0)
    min_influence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verified_only: bool = False


# ============================================================
# STATISTICS SCHEMAS
# ============================================================

class TopicStatistics(BaseModel):
    """Topic statistics"""
    topic_id: int
    topic_name: str
    total_posts: int
    total_users: int
    total_archetypes: int
    avg_sentiment_valence: float
    avg_sentiment_arousal: float
    most_common_emotion: str
    avg_toxicity: float
    period_start: datetime
    period_end: datetime


class SystemStatistics(BaseModel):
    """System-wide statistics"""
    total_posts: int
    total_users: int
    total_topics: int
    total_archetypes: int
    posts_processed_today: int
    users_profiled_today: int
    last_ingestion: Optional[datetime] = None
    last_clustering: Optional[datetime] = None


# ============================================================
# ERROR SCHEMAS
# ============================================================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "Validation error"
    detail: List[Dict[str, Any]]
    status_code: int = 422


# ============================================================
# HEALTH CHECK SCHEMAS
# ============================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    environment: str
    checks: Dict[str, str]
    warnings: Optional[List[str]] = None


# ============================================================
# INFLUENCER TRACKING SCHEMAS
# ============================================================

class InfluencerTier(str, Enum):
    """Influencer tier enum"""
    mega = "mega"      # 1M+ followers
    macro = "macro"    # 100K-1M followers
    micro = "micro"    # 10K-100K followers
    nano = "nano"      # <10K followers


class SentimentDirection(str, Enum):
    """Sentiment direction enum"""
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class ToxicityTrend(str, Enum):
    """Toxicity trend enum"""
    increasing = "increasing"
    decreasing = "decreasing"
    stable = "stable"


# Request Schemas
class InfluencerSentimentRequest(BaseModel):
    """Request for influencer sentiment time series"""
    topic_id: int = Field(..., description="Topic ID to analyze", gt=0)
    user_id: Optional[str] = Field(None, description="Specific influencer user ID (optional)")
    days_back: int = Field(7, ge=1, le=90, description="Days of history to retrieve")
    min_followers: int = Field(10000, ge=1000, description="Minimum follower count threshold")
    limit: int = Field(10, ge=1, le=50, description="Max number of influencers to return")


# Response Schemas
class InfluencerPostResponse(BaseModel):
    """Individual influencer post with metrics"""
    id: int
    post_id: int
    user_id: str
    username: str
    display_name: Optional[str] = None

    # Post details
    external_id: str
    post_url: str = Field(..., description="Clickable link to original post")
    platform: str
    content: Optional[str] = None
    posted_at: datetime

    # Influencer metrics
    follower_count: int
    influence_score: Optional[float] = None
    is_verified: bool = False

    # Current sentiment
    current_sentiment: Optional[float] = Field(None, description="Current sentiment score 0-100")
    sentiment_trend: Optional[str] = Field(None, description="rising, falling, stable")
    total_comments: int = 0

    class Config:
        from_attributes = True


class SentimentTimeSeriesPoint(BaseModel):
    """Single point in sentiment time series"""
    timestamp: datetime
    hours_since_post: float
    sentiment_score: Optional[float] = Field(None, description="Sentiment score 0-100")
    comment_count: int = 0
    dominant_emotion: Optional[str] = None
    toxicity: Optional[float] = None
    engagement: Optional[float] = None


class InfluencerSentimentTimeSeries(BaseModel):
    """Time series data for an influencer's post"""
    influencer_post_id: int
    post_url: str
    username: str
    posted_at: datetime

    # Time series data points
    data_points: List[SentimentTimeSeriesPoint]

    # Summary statistics
    avg_sentiment: float = Field(..., description="Average sentiment across all measurements")
    sentiment_change: float = Field(..., description="Change from initial to latest sentiment")
    peak_engagement_time: Optional[datetime] = None
    total_comments: int = 0


class InfluencerImpactSummary(BaseModel):
    """Summary of influencer's impact on community sentiment"""
    user_id: str
    username: str
    display_name: Optional[str] = None
    follower_count: int
    influence_score: Optional[float] = None
    tier: str = Field(..., description="mega, macro, micro, or nano")

    # Impact metrics
    total_tracked_posts: int = 0
    avg_post_sentiment: Optional[float] = Field(None, description="Average sentiment of their posts")
    avg_comment_sentiment: Optional[float] = Field(None, description="Average sentiment of comments on their posts")
    sentiment_spread: Optional[float] = Field(None, description="How much they move the needle")

    # Engagement
    avg_comments_per_post: Optional[int] = None
    avg_engagement_rate: Optional[float] = None

    # Recent activity
    last_post_at: Optional[datetime] = None
    posts_last_7_days: int = 0

    class Config:
        from_attributes = True


class InfluencerSentimentResponse(BaseModel):
    """Complete response for influencer sentiment analysis"""
    topic_id: int
    topic_name: str
    period_start: datetime
    period_end: datetime

    # Top influencers by impact
    top_influencers: List[InfluencerImpactSummary]

    # Time series data (for charting)
    time_series: List[InfluencerSentimentTimeSeries]

    # Overall metrics
    total_influencers_tracked: int
    total_posts_analyzed: int
    avg_community_response_sentiment: Optional[float] = None


class InfluencerRegistryResponse(BaseModel):
    """Response for influencer registry entry"""
    id: int
    user_id: str
    username: str
    display_name: Optional[str] = None
    topic_id: int
    topic_name: Optional[str] = None

    # Classification
    tier: str
    category: Optional[str] = None

    # Current metrics
    current_follower_count: int
    current_influence_score: Optional[float] = None
    avg_engagement_rate: Optional[float] = None

    # Historical performance
    total_tracked_posts: int
    avg_sentiment_impact: Optional[float] = None
    reach_score: Optional[float] = None

    # Status
    is_active: bool
    monitoring_enabled: bool

    # Tracking
    first_tracked_at: datetime
    last_post_at: Optional[datetime] = None
    last_updated: datetime

    class Config:
        from_attributes = True


class InfluencerListResponse(BaseModel):
    """List of influencers with pagination"""
    influencers: List[InfluencerRegistryResponse]
    total: int
    topic_id: int
    topic_name: str
    tier_filter: Optional[str] = None
