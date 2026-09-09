"""
Database Utility Functions

Helper functions for common database operations.
"""

from typing import Optional, List, Dict, Any, Type, TypeVar
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    Base,
    PostRaw,
    PostEnriched,
    User,
    Topic,
    UserMetrics,
    UserPsychProfile,
    ArchetypeMetadata,
    ClusterShift,
    ProcessingJob,
)
from src.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


# ============================================================
# GENERIC CRUD OPERATIONS
# ============================================================

async def create_record(
    db: AsyncSession,
    model: Type[T],
    **kwargs
) -> T:
    """
    Create a new database record.

    Args:
        db: Database session
        model: SQLAlchemy model class
        **kwargs: Field values

    Returns:
        Created record instance
    """
    record = model(**kwargs)
    db.add(record)
    await db.flush()
    await db.refresh(record)
    logger.debug(f"Created {model.__name__} record", id=getattr(record, "id", None))
    return record


async def get_by_id(
    db: AsyncSession,
    model: Type[T],
    record_id: Any
) -> Optional[T]:
    """
    Get a record by its primary key.

    Args:
        db: Database session
        model: SQLAlchemy model class
        record_id: Primary key value

    Returns:
        Record instance or None
    """
    result = await db.execute(select(model).where(model.id == record_id))
    return result.scalar_one_or_none()


async def get_all(
    db: AsyncSession,
    model: Type[T],
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[T]:
    """
    Get all records of a model.

    Args:
        db: Database session
        model: SQLAlchemy model class
        limit: Maximum number of records
        offset: Number of records to skip

    Returns:
        List of records
    """
    query = select(model)
    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)

    result = await db.execute(query)
    return list(result.scalars().all())


async def update_record(
    db: AsyncSession,
    model: Type[T],
    record_id: Any,
    **kwargs
) -> Optional[T]:
    """
    Update a record by ID.

    Args:
        db: Database session
        model: SQLAlchemy model class
        record_id: Primary key value
        **kwargs: Fields to update

    Returns:
        Updated record or None
    """
    stmt = (
        update(model)
        .where(model.id == record_id)
        .values(**kwargs)
        .returning(model)
    )
    result = await db.execute(stmt)
    await db.flush()
    record = result.scalar_one_or_none()
    if record:
        await db.refresh(record)
    return record


async def delete_record(
    db: AsyncSession,
    model: Type[T],
    record_id: Any
) -> bool:
    """
    Delete a record by ID.

    Args:
        db: Database session
        model: SQLAlchemy model class
        record_id: Primary key value

    Returns:
        True if deleted, False if not found
    """
    stmt = delete(model).where(model.id == record_id)
    result = await db.execute(stmt)
    await db.flush()
    return result.rowcount > 0


async def count_records(
    db: AsyncSession,
    model: Type[T],
    **filters
) -> int:
    """
    Count records matching filters.

    Args:
        db: Database session
        model: SQLAlchemy model class
        **filters: Filter conditions

    Returns:
        Count of matching records
    """
    query = select(func.count()).select_from(model)
    for field, value in filters.items():
        query = query.where(getattr(model, field) == value)

    result = await db.execute(query)
    return result.scalar()


# ============================================================
# USER OPERATIONS
# ============================================================

async def get_or_create_user(
    db: AsyncSession,
    user_id: str,
    username: str,
    platform: str,
    **kwargs
) -> tuple[User, bool]:
    """
    Get existing user or create new one.

    Args:
        db: Database session
        user_id: User ID
        username: Username
        platform: Platform name
        **kwargs: Additional user fields

    Returns:
        Tuple of (user, was_created)
    """
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()

    if user:
        # Update last_seen_at
        from datetime import datetime
        user.last_seen_at = datetime.utcnow()
        return user, False

    # Create new user
    user = await create_record(
        db,
        User,
        user_id=user_id,
        username=username,
        platform=platform,
        **kwargs
    )
    return user, True


async def get_user_by_username(
    db: AsyncSession,
    username: str,
    platform: str
) -> Optional[User]:
    """Get user by username and platform."""
    result = await db.execute(
        select(User).where(
            User.username == username,
            User.platform == platform
        )
    )
    return result.scalar_one_or_none()


# ============================================================
# POST OPERATIONS
# ============================================================

async def create_post(
    db: AsyncSession,
    external_id: str,
    user_id: str,
    content: str,
    platform: str,
    posted_at,
    **kwargs
) -> Optional[PostRaw]:
    """
    Create a new post (if not exists).

    Args:
        db: Database session
        external_id: External post ID
        user_id: User ID
        content: Post content
        platform: Platform name
        posted_at: Post timestamp
        **kwargs: Additional post fields

    Returns:
        Created post or None if already exists
    """
    # Check if post already exists
    result = await db.execute(
        select(PostRaw).where(PostRaw.external_id == external_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        logger.debug(f"Post {external_id} already exists")
        return None

    post = await create_record(
        db,
        PostRaw,
        external_id=external_id,
        user_id=user_id,
        content=content,
        platform=platform,
        posted_at=posted_at,
        **kwargs
    )
    return post


async def get_unenriched_posts(
    db: AsyncSession,
    limit: int = 100
) -> List[PostRaw]:
    """
    Get posts that haven't been enriched yet.

    Args:
        db: Database session
        limit: Maximum number of posts

    Returns:
        List of posts without enrichment
    """
    query = (
        select(PostRaw)
        .outerjoin(PostEnriched)
        .where(PostEnriched.id.is_(None))
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())


# ============================================================
# TOPIC OPERATIONS
# ============================================================

async def get_topic_by_name(
    db: AsyncSession,
    topic_name: str
) -> Optional[Topic]:
    """Get topic by name."""
    result = await db.execute(
        select(Topic).where(Topic.topic_name == topic_name)
    )
    return result.scalar_one_or_none()


async def get_active_topics(
    db: AsyncSession
) -> List[Topic]:
    """Get all active topics."""
    result = await db.execute(
        select(Topic).where(Topic.is_active == True)
    )
    return list(result.scalars().all())


# ============================================================
# PSYCHOGRAPHIC PROFILE OPERATIONS
# ============================================================

async def get_user_profile(
    db: AsyncSession,
    user_id: str,
    topic_id: int
) -> Optional[UserPsychProfile]:
    """
    Get user's psychographic profile for a topic.

    Args:
        db: Database session
        user_id: User ID
        topic_id: Topic ID

    Returns:
        User profile or None
    """
    result = await db.execute(
        select(UserPsychProfile).where(
            UserPsychProfile.user_id == user_id,
            UserPsychProfile.topic_id == topic_id
        )
    )
    return result.scalar_one_or_none()


async def upsert_user_profile(
    db: AsyncSession,
    user_id: str,
    topic_id: int,
    cluster_id: str,
    archetype_name: str,
    **kwargs
) -> UserPsychProfile:
    """
    Create or update user psychographic profile.

    Args:
        db: Database session
        user_id: User ID
        topic_id: Topic ID
        cluster_id: Cluster ID
        archetype_name: Archetype name
        **kwargs: Additional profile fields

    Returns:
        User profile
    """
    from datetime import datetime

    # Check if profile exists
    existing = await get_user_profile(db, user_id, topic_id)

    if existing:
        # Track cluster shift if changed
        if existing.cluster_id != cluster_id:
            shift = ClusterShift(
                user_id=user_id,
                topic_id=topic_id,
                from_cluster_id=existing.cluster_id,
                to_cluster_id=cluster_id,
                confidence_delta=kwargs.get("cluster_confidence", 0) - (existing.cluster_confidence or 0)
            )
            db.add(shift)

        # Update existing profile
        existing.cluster_id = cluster_id
        existing.archetype_name = archetype_name
        existing.previous_cluster_id = existing.cluster_id
        existing.last_updated = datetime.utcnow()

        for key, value in kwargs.items():
            setattr(existing, key, value)

        await db.flush()
        await db.refresh(existing)
        return existing

    # Create new profile
    profile = await create_record(
        db,
        UserPsychProfile,
        user_id=user_id,
        topic_id=topic_id,
        cluster_id=cluster_id,
        archetype_name=archetype_name,
        **kwargs
    )
    return profile


async def get_archetype_members(
    db: AsyncSession,
    topic_id: int,
    cluster_id: str,
    limit: int = 100,
    min_confidence: float = 0.5
) -> List[UserPsychProfile]:
    """
    Get all users in a specific archetype cluster.

    Args:
        db: Database session
        topic_id: Topic ID
        cluster_id: Cluster ID
        limit: Maximum number of users
        min_confidence: Minimum cluster confidence

    Returns:
        List of user profiles
    """
    query = (
        select(UserPsychProfile)
        .where(
            UserPsychProfile.topic_id == topic_id,
            UserPsychProfile.cluster_id == cluster_id,
            UserPsychProfile.cluster_confidence >= min_confidence
        )
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())


# ============================================================
# ARCHETYPE OPERATIONS
# ============================================================

async def get_topic_archetypes(
    db: AsyncSession,
    topic_id: int
) -> List[ArchetypeMetadata]:
    """Get all archetypes for a topic."""
    result = await db.execute(
        select(ArchetypeMetadata).where(ArchetypeMetadata.topic_id == topic_id)
    )
    return list(result.scalars().all())


async def update_archetype_stats(
    db: AsyncSession,
    topic_id: int,
    cluster_id: str,
    member_count: int,
    **kwargs
) -> Optional[ArchetypeMetadata]:
    """
    Update archetype statistics.

    Args:
        db: Database session
        topic_id: Topic ID
        cluster_id: Cluster ID
        member_count: Number of members
        **kwargs: Additional stats to update

    Returns:
        Updated archetype metadata
    """
    from datetime import datetime

    result = await db.execute(
        select(ArchetypeMetadata).where(
            ArchetypeMetadata.topic_id == topic_id,
            ArchetypeMetadata.cluster_id == cluster_id
        )
    )
    archetype = result.scalar_one_or_none()

    if not archetype:
        return None

    archetype.member_count = member_count
    archetype.last_updated = datetime.utcnow()

    for key, value in kwargs.items():
        setattr(archetype, key, value)

    await db.flush()
    await db.refresh(archetype)
    return archetype


# ============================================================
# PROCESSING JOB OPERATIONS
# ============================================================

async def create_processing_job(
    db: AsyncSession,
    job_type: str,
    **kwargs
) -> ProcessingJob:
    """Create a new processing job."""
    from datetime import datetime

    job = await create_record(
        db,
        ProcessingJob,
        job_type=job_type,
        status="pending",
        started_at=datetime.utcnow(),
        **kwargs
    )
    return job


async def update_job_status(
    db: AsyncSession,
    job_id: int,
    status: str,
    **kwargs
) -> Optional[ProcessingJob]:
    """
    Update processing job status.

    Args:
        db: Database session
        job_id: Job ID
        status: New status
        **kwargs: Additional fields to update

    Returns:
        Updated job
    """
    from datetime import datetime

    if status == "completed":
        kwargs["completed_at"] = datetime.utcnow()

    return await update_record(db, ProcessingJob, job_id, status=status, **kwargs)


async def get_pending_jobs(
    db: AsyncSession,
    job_type: Optional[str] = None,
    limit: int = 10
) -> List[ProcessingJob]:
    """Get pending processing jobs."""
    query = select(ProcessingJob).where(ProcessingJob.status == "pending")

    if job_type:
        query = query.where(ProcessingJob.job_type == job_type)

    query = query.order_by(ProcessingJob.created_at).limit(limit)

    result = await db.execute(query)
    return list(result.scalars().all())


# ============================================================
# ANALYTICS UTILITIES
# ============================================================

async def get_topic_statistics(
    db: AsyncSession,
    topic_id: int
) -> Dict[str, Any]:
    """
    Get statistics for a topic.

    Args:
        db: Database session
        topic_id: Topic ID

    Returns:
        Dictionary of statistics
    """
    # Count users with profiles
    profile_count = await count_records(db, UserPsychProfile, topic_id=topic_id)

    # Count archetypes
    archetype_count = await count_records(db, ArchetypeMetadata, topic_id=topic_id)

    # Count posts (via enriched posts with topic tag)
    # This is a simplified version - would need more complex query in practice

    return {
        "topic_id": topic_id,
        "user_profiles": profile_count,
        "archetypes": archetype_count,
        "total_users": profile_count,  # Simplified
    }


async def get_cluster_distribution(
    db: AsyncSession,
    topic_id: int
) -> Dict[str, int]:
    """
    Get distribution of users across clusters.

    Args:
        db: Database session
        topic_id: Topic ID

    Returns:
        Dictionary mapping cluster_id to count
    """
    query = (
        select(
            UserPsychProfile.cluster_id,
            func.count(UserPsychProfile.id).label("count")
        )
        .where(UserPsychProfile.topic_id == topic_id)
        .group_by(UserPsychProfile.cluster_id)
    )

    result = await db.execute(query)
    return {row.cluster_id: row.count for row in result}
