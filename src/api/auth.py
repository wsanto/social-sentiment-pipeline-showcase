"""
API Authentication

Provides API key authentication and rate limiting.
"""

from typing import Optional
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from src.database.connection import get_db
from src.database.models import Base
from sqlalchemy import Column, String, Boolean, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

logger = structlog.get_logger(__name__)

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey(Base):
    """
    API Key model for authentication
    """
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Rate limiting
    rate_limit_per_minute: Mapped[Optional[int]] = mapped_column(Integer, default=60)
    rate_limit_per_hour: Mapped[Optional[int]] = mapped_column(Integer, default=1000)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Permissions (JSONB would be better, but keeping simple)
    can_read: Mapped[bool] = mapped_column(Boolean, default=True)
    can_write: Mapped[bool] = mapped_column(Boolean, default=False)
    can_export: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<APIKey(name={self.name}, active={self.is_active})>"


async def get_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db)
) -> APIKey:
    """
    Validate API key from header

    Args:
        api_key_header: API key from X-API-Key header
        db: Database session

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key_header:
        logger.warning("api_key_missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing. Provide X-API-Key header."
        )

    # Query database for API key
    query = select(APIKey).where(APIKey.key == api_key_header)
    result = await db.execute(query)
    api_key = result.scalar_one_or_none()

    if not api_key:
        logger.warning("api_key_invalid", key_prefix=api_key_header[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if not api_key.is_active:
        logger.warning("api_key_inactive", key_name=api_key.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive"
        )

    # Update last used timestamp
    api_key.last_used_at = datetime.utcnow()
    await db.commit()

    logger.info("api_key_validated", key_name=api_key.name)

    return api_key


async def require_read_permission(
    api_key: APIKey = Depends(get_api_key)
) -> APIKey:
    """Require read permission"""
    if not api_key.can_read:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have read permission"
        )
    return api_key


async def require_write_permission(
    api_key: APIKey = Depends(get_api_key)
) -> APIKey:
    """Require write permission"""
    if not api_key.can_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have write permission"
        )
    return api_key


async def require_export_permission(
    api_key: APIKey = Depends(get_api_key)
) -> APIKey:
    """Require export permission"""
    if not api_key.can_export:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have export permission"
        )
    return api_key


# Utility functions for API key management

async def create_api_key(
    db: AsyncSession,
    name: str,
    can_read: bool = True,
    can_write: bool = False,
    can_export: bool = True,
    rate_limit_per_minute: int = 60,
    rate_limit_per_hour: int = 1000
) -> APIKey:
    """
    Create a new API key

    Args:
        db: Database session
        name: Descriptive name for the key
        can_read: Read permission
        can_write: Write permission
        can_export: Export permission
        rate_limit_per_minute: Rate limit per minute
        rate_limit_per_hour: Rate limit per hour

    Returns:
        Created APIKey object
    """
    import secrets

    # Generate secure API key
    key = secrets.token_urlsafe(32)

    api_key = APIKey(
        key=key,
        name=name,
        is_active=True,
        can_read=can_read,
        can_write=can_write,
        can_export=can_export,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_per_hour=rate_limit_per_hour
    )

    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    logger.info("api_key_created", name=name, key_prefix=key[:8])

    return api_key


async def revoke_api_key(db: AsyncSession, key: str) -> bool:
    """
    Revoke (deactivate) an API key

    Args:
        db: Database session
        key: API key to revoke

    Returns:
        True if revoked, False if not found
    """
    query = select(APIKey).where(APIKey.key == key)
    result = await db.execute(query)
    api_key = result.scalar_one_or_none()

    if not api_key:
        return False

    api_key.is_active = False
    await db.commit()

    logger.info("api_key_revoked", name=api_key.name)

    return True
