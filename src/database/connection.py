"""
Database Connection Module

Manages PostgreSQL database connections with async support and connection pooling.
Also provides synchronous sessions for Celery tasks.
"""

from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool, QueuePool
from contextlib import asynccontextmanager, contextmanager

from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# Global async engine instance
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

# Global sync engine instance (for Celery tasks)
_sync_engine: Engine | None = None
_sync_session_factory: sessionmaker | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the global database engine.

    Returns:
        AsyncEngine: SQLAlchemy async engine
    """
    global _engine

    if _engine is None:
        logger.info(
            "Creating database engine",
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )

        # Choose pool based on environment
        if settings.testing:
            pool_class = NullPool  # No pooling for tests
        else:
            pool_class = AsyncAdaptedQueuePool  # Connection pooling for production

        _engine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_pre_ping=True,  # Verify connections before use
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            poolclass=pool_class,
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the global session factory.

    Returns:
        async_sessionmaker: Factory for creating async sessions
    """
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.

    Usage:
        async with get_db_session() as db:
            result = await db.execute(query)

    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI routes:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            result = await db.execute(query)

    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database (create all tables).

    WARNING: This should only be used in development/testing.
    In production, use Alembic migrations.
    """
    from src.database.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        logger.info("Creating database tables")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete ALL data! Only use in testing.
    """
    from src.database.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        logger.warning("Dropping all database tables")
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")


async def close_db() -> None:
    """
    Close database connections and dispose of the engine.
    Call this on application shutdown.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.info("Closing database connections")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


def reset_async_engine() -> None:
    """
    Reset the global async engine and session factory.

    MUST be called before asyncio.run() in Celery tasks to prevent
    'Future attached to a different loop' errors. Each asyncio.run()
    creates a new event loop, but the old engine holds connections
    bound to the previous (closed) loop.
    """
    global _engine, _session_factory
    _engine = None
    _session_factory = None


async def check_db_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        bool: True if connection is successful
    """
    try:
        async with get_db_session() as db:
            await db.execute("SELECT 1")
        logger.info("Database connection check: OK")
        return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


# =============================================================================
# Synchronous Database Sessions (for Celery tasks)
# =============================================================================

def get_sync_engine() -> Engine:
    """
    Get or create the global synchronous database engine.
    Used by Celery tasks that need synchronous database access.

    Returns:
        Engine: SQLAlchemy synchronous engine
    """
    global _sync_engine

    if _sync_engine is None:
        # Convert async URL to sync (remove +asyncpg)
        sync_url = settings.database_url.replace('+asyncpg', '')

        logger.info(
            "Creating synchronous database engine for Celery tasks",
            pool_size=5,
            max_overflow=10,
        )

        _sync_engine = create_engine(
            sync_url,
            echo=settings.database_echo,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            poolclass=QueuePool,
        )

    return _sync_engine


def get_sync_session_factory() -> sessionmaker:
    """
    Get or create the global synchronous session factory.

    Returns:
        sessionmaker: Factory for creating sync sessions
    """
    global _sync_session_factory

    if _sync_session_factory is None:
        engine = get_sync_engine()
        _sync_session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    return _sync_session_factory


def get_sync_session() -> Session:
    """
    Get a synchronous database session for Celery tasks.

    IMPORTANT: Caller is responsible for closing the session!

    Usage:
        session = get_sync_session()
        try:
            result = session.execute(query)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    Returns:
        Session: Synchronous database session
    """
    factory = get_sync_session_factory()
    return factory()


@contextmanager
def sync_db_session() -> Generator[Session, None, None]:
    """
    Context manager for synchronous database sessions.
    Automatically handles commit/rollback/close.

    Usage:
        with sync_db_session() as db:
            result = db.execute(query)

    Yields:
        Session: Synchronous database session
    """
    session = get_sync_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Sync database session error", error=str(e))
        raise
    finally:
        session.close()


def close_sync_db() -> None:
    """
    Close synchronous database connections and dispose of the engine.
    Call this on application shutdown.
    """
    global _sync_engine, _sync_session_factory

    if _sync_engine is not None:
        logger.info("Closing synchronous database connections")
        _sync_engine.dispose()
        _sync_engine = None
        _sync_session_factory = None
        logger.info("Synchronous database connections closed")
