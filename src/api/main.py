"""
India Social Media Sentiment Analysis - FastAPI Application

Main application entry point.
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import settings
from src.config.logging import setup_logging, get_logger
from src.api.middleware.mock_detection import MockDataDetectionMiddleware
from src.api.middleware.prometheus import PrometheusMiddleware, metrics_endpoint

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="India Social Media Sentiment Analysis",
    description="Psychology-based social media sentiment analysis system for India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
cors_origins = settings.cors_origins_list.copy()
# Ensure all known frontends are allowed
for origin in [
    "http://localhost:3000",
    "http://localhost:3002",
    "https://synapse-reporting.kaikostudios.xyz",
    "https://v0-india-social-psychographic.vercel.app",
]:
    if origin not in cors_origins:
        cors_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# Add mock data detection middleware (monitors responses for potential mock data)
# Set enable_body_scanning=True for development/staging to scan response bodies
# Keep it False in production for performance
import os
app.add_middleware(
    MockDataDetectionMiddleware,
    enable_body_scanning=os.getenv("ENVIRONMENT", "development").lower() != "production"
)

# Prometheus metrics endpoint
from starlette.routing import Route
app.routes.append(Route("/metrics", metrics_endpoint))


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info(
        "Starting India Social Media Sentiment Analysis API",
        environment=settings.sentry_environment,
        debug=settings.debug,
    )

    # Auto-migration disabled -- migrations are applied manually to avoid
    # conflicts with branched alembic history. Run: alembic upgrade head
    logger.info("Skipping auto-migration (apply manually via alembic)")

    logger.info(
        "API Keys configured",
        kaiko_configured=bool(settings.kaiko_api_key and settings.kaiko_api_key != "your_kaiko_api_key_here"),
        lunarcrush_configured=bool(settings.lunarcrush_api_key and settings.lunarcrush_api_key != "your_lunarcrush_api_key_here"),
    )

    # Warn loudly if alerting is not configured
    if not settings.alert_email_to or not settings.alert_email_to.strip():
        logger.critical(
            "ALERTING NOT CONFIGURED - pipeline failures will go unnoticed! "
            "Set ALERT_EMAIL_TO, ALERT_EMAIL_API_KEY, ALERT_EMAIL_PROVIDER env vars."
        )
    elif not settings.alert_email_api_key or not settings.alert_email_api_key.strip():
        logger.critical(
            "ALERT_EMAIL_API_KEY not set - email alerts will fail silently! "
            "Set ALERT_EMAIL_API_KEY env var for your email provider."
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down India Social Media Sentiment Analysis API")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "India Social Media Sentiment Analysis",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"], status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        JSONResponse: Health status of the application
    """
    health_status = {
        "status": "healthy",
        "service": "india-sentiment-analysis",
        "version": "1.0.0",
        "environment": settings.sentry_environment,
        "checks": {
            "api": "ok",
            "configuration": "ok",
        }
    }

    # Check API keys configuration
    kaiko_configured = bool(
        settings.kaiko_api_key and
        settings.kaiko_api_key != "your_kaiko_api_key_here"
    )
    lunarcrush_configured = bool(
        settings.lunarcrush_api_key and
        settings.lunarcrush_api_key != "your_lunarcrush_api_key_here"
    )

    health_status["checks"]["kaiko_api_key"] = "configured" if kaiko_configured else "missing"
    health_status["checks"]["lunarcrush_api_key"] = "configured" if lunarcrush_configured else "missing"

    # Determine overall status
    if not kaiko_configured or not lunarcrush_configured:
        health_status["status"] = "degraded"
        health_status["warnings"] = []
        if not kaiko_configured:
            health_status["warnings"].append("Kaiko API key not configured")
        if not lunarcrush_configured:
            health_status["warnings"].append("LunarCrush API key not configured")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=health_status
    )


@app.get("/config", tags=["Configuration"])
async def get_config():
    """
    Get non-sensitive configuration information.

    Returns:
        dict: Public configuration settings
    """
    return {
        "topics": settings.topics_list,
        "target_country": settings.target_country,
        "target_languages": settings.target_languages_list,
        "clustering": {
            "n_clusters": settings.clustering_n_clusters,
            "embedding_dimension": settings.embedding_dimension,
            "min_posts_for_profiling": settings.min_posts_for_profiling,
        },
        "processing": {
            "fetch_interval_minutes": settings.fetch_interval_minutes,
            "enrich_interval_minutes": settings.enrich_interval_minutes,
            "aggregate_interval_minutes": settings.aggregate_interval_minutes,
            "cluster_interval_hours": settings.cluster_interval_hours,
        },
        "features": {
            "archetype_labeling": settings.enable_archetype_labeling,
            "network_analysis": settings.enable_network_analysis,
            "predictive_analytics": settings.enable_predictive_analytics,
            "multilingual_processing": settings.enable_multilingual_processing,
        }
    }


# Import and include routers
from src.api.routes import users, archetypes, analytics, dashboard, topics, reports, influencers, profiling, monitoring, validation, safety, empathy, growth, advanced_analytics, events, brands

app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(archetypes.router, prefix="/api/v1", tags=["Archetypes"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(dashboard.router, prefix="/api/v1/analytics", tags=["Dashboard"])
app.include_router(topics.router, prefix="/api/v1", tags=["Topics"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(influencers.router, prefix="/api/v1/influencers", tags=["Influencers"])
app.include_router(profiling.router, prefix="/api/v1/profiling", tags=["Psychographics"])
app.include_router(monitoring.router)  # Monitoring endpoints (already has /api/v1/monitoring prefix)
app.include_router(validation.router, prefix="/api/v1/posts", tags=["Validation"])
app.include_router(safety.router, prefix="/api/v1/analytics", tags=["Safety & Hostility"])
app.include_router(empathy.router, prefix="/api/v1/analytics", tags=["Empathy & Spectrum"])
app.include_router(growth.router, prefix="/api/v1/analytics", tags=["Growth & Beliefs"])
app.include_router(advanced_analytics.router)  # Has its own /api/v1/analytics/advanced prefix
app.include_router(events.router, prefix="/api/v1/analytics", tags=["Events & Beliefs"])
app.include_router(brands.router, prefix="/api/v1/brands", tags=["SentimentScope Brands"])
# Note: collection router temporarily disabled - requires src.services module


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
