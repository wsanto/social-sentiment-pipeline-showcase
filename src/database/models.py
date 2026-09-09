"""
SQLAlchemy Database Models

Complete ORM models for India Social Media Sentiment Analysis system.
Includes pgvector support for embedding storage.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    Boolean,
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, REAL
from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# ============================================================
# RAW DATA LAYER
# ============================================================

class PostRaw(Base):
    """
    Raw social media posts fetched from LunarCrush API.
    Stores original post data before enrichment.
    """
    __tablename__ = "posts_raw"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String, nullable=False, index=True)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    hashtags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    mentions: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    engagement_score: Mapped[Optional[float]] = mapped_column(Float)
    language: Mapped[Optional[str]] = mapped_column(String, index=True)  # ISO 639-1 code (en, hi, ta, etc.)
    post_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="posts")
    enriched: Mapped[Optional["PostEnriched"]] = relationship(
        "PostEnriched",
        back_populates="post",
        uselist=False
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "platform IN ('twitter', 'reddit', 'telegram', 'instagram')",
            name="posts_raw_platform_check"
        ),
        Index("idx_posts_raw_posted_at_desc", posted_at.desc()),
        Index("idx_posts_raw_hashtags_gin", hashtags, postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<PostRaw(id={self.id}, external_id={self.external_id}, platform={self.platform})>"


class PostEnriched(Base):
    """
    Enriched posts with emotion analysis from Kaiko EQ+.
    One-to-one relationship with PostRaw.
    """
    __tablename__ = "posts_enriched"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("posts_raw.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )

    # Emotion metrics (from Kaiko EQ+)
    valence: Mapped[Optional[float]] = mapped_column(Float)  # -1 to +1
    arousal: Mapped[Optional[float]] = mapped_column(Float)  # 0 to 1

    # Discrete emotion distribution (Plutchik's 8 emotions)
    emotion_joy: Mapped[Optional[float]] = mapped_column(Float)
    emotion_trust: Mapped[Optional[float]] = mapped_column(Float)
    emotion_fear: Mapped[Optional[float]] = mapped_column(Float)
    emotion_surprise: Mapped[Optional[float]] = mapped_column(Float)
    emotion_sadness: Mapped[Optional[float]] = mapped_column(Float)
    emotion_disgust: Mapped[Optional[float]] = mapped_column(Float)
    emotion_anger: Mapped[Optional[float]] = mapped_column(Float)
    emotion_anticipation: Mapped[Optional[float]] = mapped_column(Float)

    # Kaiko V2 Metrics
    intensity: Mapped[Optional[float]] = mapped_column(Float)  # 0.0-1.0
    intensity_level: Mapped[Optional[str]] = mapped_column(String)  # minimal, subtle, moderate, high, critical
    emotional_complexity: Mapped[Optional[str]] = mapped_column(String)  # simple, layered, paradoxical, transcendent
    wonder_index: Mapped[Optional[float]] = mapped_column(Float)  # 0.0-1.0
    discovery_level: Mapped[Optional[str]] = mapped_column(String)  # routine, normal, significant, breakthrough, transcendent
    
    # Store full V2 emotion object for future use
    raw_v2_emotions: Mapped[Optional[dict]] = mapped_column(JSONB)

    # P0: Safety & Hostility (from SDK V2 hostilityState + safetyConcernScore)
    safety_concern_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-1, SDK safetyConcernScore
    hostility_level: Mapped[Optional[str]] = mapped_column(String)  # none/low/medium/high
    hostility_escalation_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    de_escalation_detected: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    is_breakthrough: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)  # SDK isBreakthrough

    # P1: Empathy & Meta-Emotional (from SDK V2)
    empathic_concern: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    personal_distress: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    meta_emotional_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    emotional_vector: Mapped[Optional[list]] = mapped_column(ARRAY(REAL), nullable=True)  # 9D SDK vector
    emotional_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    active_labels: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))  # Full 27 GoEmotions that passed threshold
    raw_27_emotions: Mapped[Optional[dict]] = mapped_column(JSONB)  # Structured 27-emotion scores

    # P2: Additional SDK fields (forward compat)
    emotional_signature: Mapped[Optional[str]] = mapped_column(String)  # Hash for dedup/similarity
    pattern_emotion_score: Mapped[Optional[float]] = mapped_column(Float)  # Pattern recognition quality
    dimensional_source: Mapped[Optional[str]] = mapped_column(String)  # ml_predicted vs rule_derived
    is_fallback: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)  # Circuit-breaker flag

    # Derived metrics
    dominant_emotion: Mapped[Optional[str]] = mapped_column(String, index=True)
    toxicity_score: Mapped[Optional[float]] = mapped_column(Float)
    hostility_score: Mapped[Optional[float]] = mapped_column(Float)  # SDK hostilityState.score
    salience_score: Mapped[Optional[float]] = mapped_column(Float)

    # SDK context-aware fields
    conversation_mode: Mapped[Optional[str]] = mapped_column(String)  # analytical_deep_dive, emotional_support, etc.
    personality_mode: Mapped[Optional[str]] = mapped_column(String)  # SDK personalityMode
    data_quality: Mapped[Optional[str]] = mapped_column(String)  # ml_predicted, rule_derived, fallback

    # Topic/entity extraction
    topics: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    entities: Mapped[Optional[dict]] = mapped_column(JSONB)  # {person: [], org: [], location: []}

    # Metadata
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    kaiko_version: Mapped[Optional[str]] = mapped_column(String)

    # Relationships
    post: Mapped["PostRaw"] = relationship("PostRaw", back_populates="enriched")

    # Indexes
    __table_args__ = (
        Index("idx_posts_enriched_topics_gin", topics, postgresql_using="gin"),
        Index("idx_posts_enriched_safety_concern", safety_concern_score.desc()),
        Index("idx_posts_enriched_hostility_level", hostility_level),
        Index("idx_posts_enriched_is_breakthrough", is_breakthrough),
    )

    def __repr__(self) -> str:
        return f"<PostEnriched(id={self.id}, post_id={self.post_id}, dominant_emotion={self.dominant_emotion})>"


# ============================================================
# USER DATA LAYER
# ============================================================

class User(Base):
    """
    User profiles from social media platforms.
    Stores demographic and account information.
    """
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    platform: Mapped[str] = mapped_column(String, nullable=False, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String)

    # Profile metrics
    followers_count: Mapped[int] = mapped_column(Integer, default=0)
    following_count: Mapped[int] = mapped_column(Integer, default=0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Location (India-specific)
    country: Mapped[str] = mapped_column(String, default="IN")
    state: Mapped[Optional[str]] = mapped_column(String)
    city: Mapped[Optional[str]] = mapped_column(String)

    # Metadata
    account_created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    profile_post_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Relationships
    posts: Mapped[List["PostRaw"]] = relationship("PostRaw", back_populates="user")
    metrics: Mapped[List["UserMetrics"]] = relationship("UserMetrics", back_populates="user")
    psych_profiles: Mapped[List["UserPsychProfile"]] = relationship("UserPsychProfile", back_populates="user")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "platform IN ('twitter', 'reddit', 'telegram', 'instagram')",
            name="users_platform_check"
        ),
        Index("idx_users_location", country, state, city),
        Index("idx_users_last_seen_desc", last_seen_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, username={self.username}, platform={self.platform})>"


# ============================================================
# TOPICS LAYER
# ============================================================

class Topic(Base):
    """
    Topics/categories for sentiment analysis.
    Examples: politics, crypto, sports, etc.
    """
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String, index=True)
    keywords: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    hashtags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user_metrics: Mapped[List["UserMetrics"]] = relationship("UserMetrics", back_populates="topic")
    psych_profiles: Mapped[List["UserPsychProfile"]] = relationship("UserPsychProfile", back_populates="topic")
    archetypes: Mapped[List["ArchetypeMetadata"]] = relationship("ArchetypeMetadata", back_populates="topic")

    # Indexes
    __table_args__ = (
        Index("idx_topics_keywords_gin", keywords, postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Topic(id={self.id}, topic_name={self.topic_name}, category={self.category})>"


# ============================================================
# USER METRICS LAYER (Aggregated)
# ============================================================

class UserMetrics(Base):
    """
    Aggregated user metrics per topic for a time window.
    Computed from enriched posts.
    """
    __tablename__ = "user_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Time window
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Emotion balance
    avg_valence: Mapped[Optional[float]] = mapped_column(Float)
    avg_arousal: Mapped[Optional[float]] = mapped_column(Float)
    emotion_distribution: Mapped[Optional[dict]] = mapped_column(JSONB)  # {joy: 0.41, trust: 0.23, ...}

    # Volatility
    valence_stddev: Mapped[Optional[float]] = mapped_column(Float)
    arousal_stddev: Mapped[Optional[float]] = mapped_column(Float)
    emotion_stability: Mapped[Optional[float]] = mapped_column(Float)
    topic_volatility: Mapped[Optional[float]] = mapped_column(Float)

    # Dominant patterns
    primary_emotion: Mapped[Optional[str]] = mapped_column(String)
    secondary_emotion: Mapped[Optional[str]] = mapped_column(String)

    # Toxicity/Hostility
    mean_toxicity: Mapped[Optional[float]] = mapped_column(Float)
    mean_hostility: Mapped[Optional[float]] = mapped_column(Float)

    # P0: Safety & Hostility Aggregates
    avg_safety_concern: Mapped[Optional[float]] = mapped_column(Float)
    hostility_incident_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    breakthrough_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # P1: Empathy & Meta-Emotional Aggregates
    avg_empathic_concern: Mapped[Optional[float]] = mapped_column(Float)
    avg_personal_distress: Mapped[Optional[float]] = mapped_column(Float)
    avg_meta_emotional_score: Mapped[Optional[float]] = mapped_column(Float)
    active_labels_distribution: Mapped[Optional[dict]] = mapped_column(JSONB)  # 27-emotion distribution

    # Engagement
    avg_engagement_score: Mapped[Optional[float]] = mapped_column(Float)
    reaction_bias: Mapped[Optional[float]] = mapped_column(Float)

    # Temporal patterns
    circadian_pattern: Mapped[Optional[dict]] = mapped_column(JSONB)  # {hour_0: 0.02, ...}
    peak_activity_hour: Mapped[Optional[int]] = mapped_column(Integer)

    # Social influence
    influence_score: Mapped[Optional[float]] = mapped_column(Float)
    follower_log: Mapped[Optional[float]] = mapped_column(Float)

    # Cognitive complexity
    topic_diversity: Mapped[Optional[int]] = mapped_column(Integer)
    entropy_score: Mapped[Optional[float]] = mapped_column(Float)
    avg_content_length: Mapped[Optional[float]] = mapped_column(Float)

    # Trust/Alignment
    trust_level: Mapped[Optional[float]] = mapped_column(Float)
    agreement_rate: Mapped[Optional[float]] = mapped_column(Float)

    # Post statistics
    post_count: Mapped[int] = mapped_column(Integer, default=0)

    # Metadata
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="metrics")
    topic: Mapped["Topic"] = relationship("Topic", back_populates="user_metrics")

    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "topic_id", "period_start", name="uq_user_topic_period"),
        Index("idx_user_metrics_user_topic", user_id, topic_id),
        Index("idx_user_metrics_period", period_start, period_end),
    )

    def __repr__(self) -> str:
        return f"<UserMetrics(id={self.id}, user_id={self.user_id}, topic_id={self.topic_id})>"


# ============================================================
# PSYCHOGRAPHIC PROFILE LAYER
# ============================================================

class UserPsychProfile(Base):
    """
    User psychographic profile with archetype assignment.
    Includes 64-dimensional embedding vector for similarity search.
    """
    __tablename__ = "user_psych_profile"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Cluster assignment
    cluster_id: Mapped[str] = mapped_column(String, nullable=False)  # e.g., 'P1', 'C3'
    archetype_name: Mapped[str] = mapped_column(String, nullable=False)
    cluster_confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0-1

    # Psychographic embedding (64-dim vector for pgvector)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(64))

    # UMAP 2D coordinates for scatter visualization
    umap_x: Mapped[Optional[float]] = mapped_column(Float)
    umap_y: Mapped[Optional[float]] = mapped_column(Float)

    # Aggregated metrics snapshot (denormalized for performance)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Temporal tracking
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    previous_cluster_id: Mapped[Optional[str]] = mapped_column(String)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="psych_profiles")
    topic: Mapped["Topic"] = relationship("Topic", back_populates="psych_profiles")

    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "topic_id", name="uq_user_topic_profile"),
        Index("idx_psych_profile_topic_cluster", topic_id, cluster_id),
        Index(
            "idx_psych_profile_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        Index("idx_psych_profile_assigned_at_desc", assigned_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<UserPsychProfile(user_id={self.user_id}, archetype={self.archetype_name})>"


# ============================================================
# ARCHETYPE METADATA LAYER
# ============================================================

class ArchetypeMetadata(Base):
    """
    Metadata for each archetype cluster per topic.
    Defines the 10 archetypes (e.g., The Loyalist, The Agitator).
    """
    __tablename__ = "archetype_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False
    )
    cluster_id: Mapped[str] = mapped_column(String, nullable=False)
    archetype_name: Mapped[str] = mapped_column(String, nullable=False)

    # Description
    description: Mapped[Optional[str]] = mapped_column(Text)
    characteristics: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))

    # Cluster statistics
    centroid: Mapped[Optional[Vector]] = mapped_column(Vector(64))
    member_count: Mapped[int] = mapped_column(Integer, default=0)

    # Typical emotional profile
    typical_emotions: Mapped[Optional[dict]] = mapped_column(JSONB)
    avg_toxicity: Mapped[Optional[float]] = mapped_column(Float)
    avg_influence: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="archetypes")

    # Constraints
    __table_args__ = (
        UniqueConstraint("topic_id", "cluster_id", name="uq_topic_cluster"),
        Index("idx_archetype_topic", topic_id),
    )

    def __repr__(self) -> str:
        return f"<ArchetypeMetadata(topic_id={self.topic_id}, archetype={self.archetype_name})>"


# ============================================================
# EMOTIONAL PATTERNS (P1)
# ============================================================

class EmotionalPattern(Base):
    """
    Detected emotional patterns per user from SDK V2 pattern detection.
    Tracks temporal, trigger-based, cyclical, sequential, and recovery patterns.
    """
    __tablename__ = "emotional_patterns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        index=True
    )

    # Pattern details
    pattern_type: Mapped[str] = mapped_column(String, nullable=False)  # temporal, trigger, cyclical, sequential, recovery
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Quality metrics
    frequency: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    pattern_data: Mapped[Optional[dict]] = mapped_column(JSONB)  # Pattern-specific data

    # Temporal tracking
    first_occurrence: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_occurrence: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User")

    __table_args__ = (
        Index("idx_emotional_patterns_user_type", user_id, pattern_type),
        Index("idx_emotional_patterns_confidence", confidence.desc()),
    )

    def __repr__(self) -> str:
        return f"<EmotionalPattern(user_id={self.user_id}, type={self.pattern_type}, name={self.name})>"


# ============================================================
# GROWTH TRACKING (P2)
# ============================================================

class UserGrowthTracking(Base):
    """
    Tracks EQ growth across 5 dimensions from SDK V2 growth section.
    Dimensions: regulation, awareness, vulnerability, resilience, complexity.
    """
    __tablename__ = "user_growth_tracking"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        index=True
    )

    # Growth dimension
    dimension: Mapped[str] = mapped_column(String, nullable=False)  # regulation, awareness, vulnerability, resilience, complexity
    percentage_change: Mapped[Optional[float]] = mapped_column(Float)
    recent_value: Mapped[Optional[float]] = mapped_column(Float)
    baseline_value: Mapped[Optional[float]] = mapped_column(Float)
    is_growing: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)

    # Tracking
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_growth_user_dimension", user_id, dimension),
        Index("idx_growth_measured_at", measured_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<UserGrowthTracking(user_id={self.user_id}, dimension={self.dimension}, growing={self.is_growing})>"


# ============================================================
# BELIEF DETECTION (P2)
# ============================================================

class BeliefDetection(Base):
    """
    Detected beliefs from SDK V2 beliefs section.
    Types: explicit, identity, value, purpose.
    """
    __tablename__ = "belief_detection"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    post_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("posts_raw.id", ondelete="CASCADE"),
        index=True
    )
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        index=True
    )

    # Belief details
    belief_type: Mapped[str] = mapped_column(String, nullable=False)  # explicit, identity, value, purpose
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    belief_data: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Tracking
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_belief_user_type", user_id, belief_type),
        Index("idx_belief_topic", topic_id),
        Index("idx_belief_confidence", confidence.desc()),
    )

    def __repr__(self) -> str:
        return f"<BeliefDetection(user_id={self.user_id}, type={self.belief_type})>"


# ============================================================
# EVENTS TABLE (P2 — Event Correlation)
# ============================================================

class ExternalEvent(Base):
    """
    External events for sentiment correlation analysis.
    Elections, market crashes, policy changes, etc.
    """
    __tablename__ = "external_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_name: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)  # political, economic, social, crisis, tech
    description: Mapped[Optional[str]] = mapped_column(Text)
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[Optional[str]] = mapped_column(String)
    impact_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-1, manual or computed
    event_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Correlation results (computed)
    valence_impact: Mapped[Optional[float]] = mapped_column(Float)
    arousal_impact: Mapped[Optional[float]] = mapped_column(Float)
    toxicity_impact: Mapped[Optional[float]] = mapped_column(Float)
    correlated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_events_date", event_date.desc()),
        Index("idx_events_type", event_type),
    )

    def __repr__(self) -> str:
        return f"<ExternalEvent(name={self.event_name}, date={self.event_date})>"


# ============================================================
# CLUSTER EVOLUTION TRACKING
# ============================================================

class ClusterShift(Base):
    """
    Tracks when users move between archetype clusters.
    Used for temporal analysis of sentiment shifts.
    """
    __tablename__ = "cluster_shifts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    from_cluster_id: Mapped[Optional[str]] = mapped_column(String)
    to_cluster_id: Mapped[str] = mapped_column(String, nullable=False)

    shift_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    confidence_delta: Mapped[Optional[float]] = mapped_column(Float)

    trigger_event: Mapped[Optional[str]] = mapped_column(String)

    # Indexes
    __table_args__ = (
        Index("idx_cluster_shifts_timestamp_desc", shift_timestamp.desc()),
    )

    def __repr__(self) -> str:
        return f"<ClusterShift(user_id={self.user_id}, from={self.from_cluster_id}, to={self.to_cluster_id})>"


# ============================================================
# PROCESSING JOBS TRACKING
# ============================================================

class ProcessingJob(Base):
    """
    Tracks data processing jobs (ingestion, enrichment, aggregation, clustering).
    Used for monitoring and debugging the pipeline.
    """
    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)

    # Job metadata
    topic_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("topics.id"))
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)

    error_message: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="processing_jobs_status_check"
        ),
        Index("idx_processing_jobs_created_desc", created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, type={self.job_type}, status={self.status})>"


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry(Base):
    """
    Registry for ML model versions and performance tracking.
    Supports model persistence, versioning, and A/B testing.
    """
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)

    # Optional topic-specific model
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        index=True
    )

    # Performance metrics
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Additional metadata
    model_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Model status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    # Constraints
    __table_args__ = (
        UniqueConstraint("model_type", "version", "topic_id", name="uq_model_version"),
        Index("idx_model_registry_active", model_type, is_active),
        Index("idx_model_registry_created_desc", created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, type={self.model_type}, version={self.version}, active={self.is_active})>"


# ============================================================
# INFLUENCER TRACKING LAYER
# ============================================================

class InfluencerRegistry(Base):
    """
    Registry of identified influencers per topic.
    Maintains a list of high-follower users to track for sentiment impact analysis.
    """
    __tablename__ = "influencer_registry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Influencer classification
    tier: Mapped[str] = mapped_column(String, nullable=False)  # mega, macro, micro, nano
    category: Mapped[Optional[str]] = mapped_column(String)  # politics, entertainment, tech, etc.

    # Current metrics
    current_follower_count: Mapped[int] = mapped_column(Integer, nullable=False)
    current_influence_score: Mapped[Optional[float]] = mapped_column(Float)
    avg_engagement_rate: Mapped[Optional[float]] = mapped_column(Float)

    # Historical performance
    total_tracked_posts: Mapped[int] = mapped_column(Integer, default=0)
    avg_sentiment_impact: Mapped[Optional[float]] = mapped_column(Float)  # How much they move sentiment
    reach_score: Mapped[Optional[float]] = mapped_column(Float)  # Estimated unique viewers

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Tracking
    first_tracked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_post_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User")
    topic: Mapped["Topic"] = relationship("Topic")
    influencer_posts: Mapped[List["InfluencerPost"]] = relationship("InfluencerPost", back_populates="influencer_registry")

    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "topic_id", name="uq_influencer_user_topic"),
        CheckConstraint(
            "tier IN ('mega', 'macro', 'micro', 'nano')",
            name="influencer_registry_tier_check"
        ),
        Index("idx_influencer_registry_user", user_id),
        Index("idx_influencer_registry_topic", topic_id),
        Index("idx_influencer_registry_tier", tier),
        Index("idx_influencer_registry_followers", current_follower_count.desc()),
        Index("idx_influencer_registry_last_updated_desc", last_updated.desc()),
    )

    def __repr__(self) -> str:
        return f"<InfluencerRegistry(id={self.id}, user_id={self.user_id}, topic_id={self.topic_id}, tier={self.tier})>"


class InfluencerPost(Base):
    """
    Tracks posts from identified influencers for sentiment response analysis.
    Stores post links and metadata for tracking community sentiment over time.
    """
    __tablename__ = "influencer_posts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("posts_raw.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    influencer_registry_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("influencer_registry.id", ondelete="SET NULL"),
        index=True
    )

    # Post metadata
    external_id: Mapped[str] = mapped_column(String, nullable=False)  # Original platform post ID
    post_url: Mapped[str] = mapped_column(String, nullable=False)  # Direct link to social media post
    platform: Mapped[str] = mapped_column(String, nullable=False)  # twitter, reddit, telegram, instagram

    # Influencer metrics at time of post
    follower_count_snapshot: Mapped[int] = mapped_column(Integer, nullable=False)
    influence_score: Mapped[Optional[float]] = mapped_column(Float)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Post performance
    initial_engagement: Mapped[Optional[float]] = mapped_column(Float)  # Engagement at time of capture
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    # Tracking
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    post: Mapped["PostRaw"] = relationship("PostRaw")
    user: Mapped["User"] = relationship("User")
    topic: Mapped["Topic"] = relationship("Topic")
    influencer_registry: Mapped[Optional["InfluencerRegistry"]] = relationship("InfluencerRegistry", back_populates="influencer_posts")
    metrics: Mapped[List["InfluencerPostMetrics"]] = relationship("InfluencerPostMetrics", back_populates="influencer_post", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        UniqueConstraint("external_id", "platform", name="uq_influencer_post_external"),
        CheckConstraint(
            "platform IN ('twitter', 'reddit', 'telegram', 'instagram')",
            name="influencer_posts_platform_check"
        ),
        Index("idx_influencer_posts_user", user_id),
        Index("idx_influencer_posts_topic", topic_id),
        Index("idx_influencer_posts_posted_at_desc", posted_at.desc()),
        Index("idx_influencer_posts_follower_count", follower_count_snapshot.desc()),
        Index("idx_influencer_posts_registry", influencer_registry_id),
    )

    def __repr__(self) -> str:
        return f"<InfluencerPost(id={self.id}, user_id={self.user_id}, platform={self.platform}, posted_at={self.posted_at})>"


class InfluencerPostMetrics(Base):
    """
    Time-series sentiment metrics for influencer posts.
    Tracks how community sentiment evolves in response to influencer posts over time.
    """
    __tablename__ = "influencer_post_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    influencer_post_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("influencer_posts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Time window
    measurement_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    hours_since_post: Mapped[float] = mapped_column(Float, nullable=False)  # For time-decay analysis

    # Sentiment metrics (from comments/replies)
    avg_sentiment_valence: Mapped[Optional[float]] = mapped_column(Float)  # -1 to +1
    avg_sentiment_arousal: Mapped[Optional[float]] = mapped_column(Float)  # 0 to 1
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)  # Normalized 0-100

    # Emotion distribution
    emotion_distribution: Mapped[Optional[dict]] = mapped_column(JSONB)  # {joy: 0.30, anger: 0.15, ...}
    dominant_emotion: Mapped[Optional[str]] = mapped_column(String)

    # Engagement metrics
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    total_engagement: Mapped[Optional[float]] = mapped_column(Float)

    # Toxicity tracking
    avg_toxicity: Mapped[Optional[float]] = mapped_column(Float)
    toxicity_trend: Mapped[Optional[str]] = mapped_column(String)  # increasing, decreasing, stable

    # Sentiment momentum
    sentiment_velocity: Mapped[Optional[float]] = mapped_column(Float)  # Rate of sentiment change
    sentiment_direction: Mapped[Optional[str]] = mapped_column(String)  # positive, negative, neutral

    # Metadata
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    influencer_post: Mapped["InfluencerPost"] = relationship("InfluencerPost", back_populates="metrics")

    # Constraints
    __table_args__ = (
        UniqueConstraint("influencer_post_id", "measurement_timestamp", name="uq_influencer_metrics_timestamp"),
        CheckConstraint(
            "toxicity_trend IN ('increasing', 'decreasing', 'stable') OR toxicity_trend IS NULL",
            name="influencer_metrics_toxicity_trend_check"
        ),
        CheckConstraint(
            "sentiment_direction IN ('positive', 'negative', 'neutral') OR sentiment_direction IS NULL",
            name="influencer_metrics_sentiment_direction_check"
        ),
        Index("idx_influencer_metrics_post", influencer_post_id),
        Index("idx_influencer_metrics_timestamp_desc", measurement_timestamp.desc()),
        Index("idx_influencer_metrics_hours_since", hours_since_post),
        Index("idx_influencer_metrics_sentiment", sentiment_score.desc()),
    )

    def __repr__(self) -> str:
        return f"<InfluencerPostMetrics(id={self.id}, influencer_post_id={self.influencer_post_id}, hours_since_post={self.hours_since_post})>"
