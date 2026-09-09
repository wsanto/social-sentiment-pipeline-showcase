"""
Application Settings

Centralized configuration management using pydantic-settings.
Loads configuration from environment variables.
"""

from typing import List
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ============================================================
    # Database Configuration
    # ============================================================
    database_url: str = Field(
        default="postgresql+asyncpg://india_sentiment:dev_password@localhost:5432/india_sentiment",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=10, description="Max overflow connections")
    database_echo: bool = Field(default=False, description="Echo SQL queries")

    @model_validator(mode='after')
    def validate_database_url(self):
        """Convert sync PostgreSQL URL to async format for asyncpg"""
        if self.database_url.startswith("postgresql://") and not self.database_url.startswith("postgresql+"):
            # Railway provides postgresql:// but we need postgresql+asyncpg://
            self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self

    # ============================================================
    # Redis Configuration
    # ============================================================
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_max_connections: int = Field(default=50, description="Redis max connections")

    # ============================================================
    # API Keys
    # ============================================================
    kaiko_api_key: str = Field(default="", description="Kaiko SDK API key")
    lunarcrush_api_key: str = Field(default="", description="LunarCrush API key")

    # ============================================================
    # API Configuration
    # ============================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(
        default=8000,
        alias="PORT",  # Railway uses PORT, fallback to API_PORT
        description="API port (Railway uses PORT env var)"
    )
    api_workers: int = Field(default=4, description="Number of API workers")
    api_reload: bool = Field(default=True, description="Auto-reload on code changes")

    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: str = Field(default="", description="Comma-separated valid API keys")

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8088",
        description="Comma-separated CORS origins"
    )

    @field_validator("api_keys")
    def validate_api_keys(cls, v: str) -> str:
        if not v:
            return v
        keys = [k.strip() for k in v.split(",")]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate API keys found")
        return v

    @property
    def api_keys_list(self) -> List[str]:
        """Get API keys as a list"""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # ============================================================
    # Celery Configuration
    # ============================================================
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/1")
    celery_task_serializer: str = Field(default="json")
    celery_result_serializer: str = Field(default="json")
    celery_accept_content: str = Field(default="json")
    celery_timezone: str = Field(default="Asia/Kolkata")
    celery_enable_utc: bool = Field(default=True)

    # ============================================================
    # Machine Learning Configuration
    # ============================================================
    clustering_n_clusters: int = Field(default=10, description="Number of archetypes per topic")
    embedding_dimension: int = Field(default=64, description="Dimension of embeddings")
    umap_n_neighbors: int = Field(default=15, description="UMAP n_neighbors parameter")
    umap_min_dist: float = Field(default=0.1, description="UMAP min_dist parameter")
    min_posts_for_profiling: int = Field(default=10, description="Min posts for user profile")

    # ============================================================
    # Data Processing Configuration
    # ============================================================
    ingestion_batch_size: int = Field(default=1000)
    enrichment_batch_size: int = Field(default=500)
    aggregation_batch_size: int = Field(default=1000)

    fetch_interval_minutes: int = Field(default=15)
    enrich_interval_minutes: int = Field(default=5)
    aggregate_interval_minutes: int = Field(default=60)
    cluster_interval_hours: int = Field(default=6)

    data_retention_days: int = Field(default=365)

    # ============================================================
    # Continuous Data Collection
    # ============================================================
    data_collection_interval_minutes: int = Field(
        default=15,
        description="How often to poll LunarCrush for new data (minutes)"
    )
    auto_start_data_collection: bool = Field(
        default=False,
        description="Automatically start data collection on startup"
    )

    # ============================================================
    # Monitoring & Logging
    # ============================================================
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json or console

    prometheus_port: int = Field(default=9090)
    enable_metrics: bool = Field(default=True)

    sentry_dsn: str = Field(default="")
    sentry_environment: str = Field(default="development")
    sentry_traces_sample_rate: float = Field(default=0.1)

    # ============================================================
    # Location & Language Configuration
    # ============================================================
    target_country: str = Field(default="IN")
    target_timezones: str = Field(default="Asia/Kolkata,Asia/Calcutta")
    target_languages: str = Field(
        default="en,hi,ta,te,bn,mr,gu,kn,ml,pa,or,as",
        description="Comma-separated language codes"
    )

    @property
    def target_timezones_list(self) -> List[str]:
        return [tz.strip() for tz in self.target_timezones.split(",")]

    @property
    def target_languages_list(self) -> List[str]:
        return [lang.strip() for lang in self.target_languages.split(",")]

    # ============================================================
    # Topics Configuration
    # ============================================================
    topics: str = Field(
        default="politics,crypto,sports,entertainment,technology,health,education,economy,environment,social_issues"
    )

    @property
    def topics_list(self) -> List[str]:
        return [topic.strip() for topic in self.topics.split(",")]

    # ============================================================
    # Security Configuration
    # ============================================================
    secret_key: str = Field(default="change_me_in_production")
    bcrypt_rounds: int = Field(default=12)
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_per_hour: int = Field(default=1000)

    # ============================================================
    # Development Flags
    # ============================================================
    debug: bool = Field(default=False)
    testing: bool = Field(default=False)

    # ============================================================
    # Feature Flags
    # ============================================================
    enable_archetype_labeling: bool = Field(default=True)
    enable_network_analysis: bool = Field(default=False)
    enable_predictive_analytics: bool = Field(default=False)
    enable_multilingual_processing: bool = Field(default=False)

    # ============================================================
    # Alerting Configuration
    # ============================================================
    alert_email_to: str = Field(
        default="",
        description="Email address to send alerts to"
    )
    alert_email_from: str = Field(
        default="alerts@kaikostudios.xyz",
        description="Email address alerts are sent from"
    )
    alert_email_provider: str = Field(
        default="resend",
        description="Email provider: resend, sendgrid, or smtp"
    )
    alert_email_api_key: str = Field(
        default="",
        description="API key for email provider (Resend or SendGrid)"
    )
    alert_smtp_host: str = Field(default="smtp.gmail.com")
    alert_smtp_port: int = Field(default=587)
    alert_smtp_user: str = Field(default="")
    alert_smtp_password: str = Field(default="")
    alert_data_stale_minutes: int = Field(
        default=30,
        description="Minutes without new data before alerting"
    )
    alert_enrichment_stale_minutes: int = Field(
        default=60,
        description="Minutes without enrichment before alerting"
    )
    alert_cooldown_minutes: int = Field(
        default=240,
        description="Minimum minutes between same alert type"
    )
    alert_enabled: bool = Field(
        default=True,
        description="Enable/disable alerting system"
    )


# Create global settings instance
settings = Settings()
