"""
Celery Application Configuration

Simplified single-queue architecture for reliable task processing.

Manages asynchronous task processing for:
- Data ingestion from LunarCrush (every 30 minutes - rate limited to 2000/day)
- Emotion enrichment with Kaiko Synapse V2 (every 5 minutes)
- Metrics aggregation (hourly)
- ML clustering (every 6 hours)
- Psychographic profiling (every 6 hours)
- Database maintenance (daily/weekly)

Architecture Notes:
- Uses single 'celery' queue for simplicity and reliability
- Can scale to multiple queues when needed (>1000 tasks/hour)
- Tasks are ordered to respect data dependencies
"""

from celery import Celery
from celery.schedules import crontab
from src.config.settings import settings

HEARTBEAT_REDIS_KEY = "celery:worker:heartbeat"

# Create Celery app
app = Celery('india_sentiment')

# Configure from settings
app.conf.update(
    # ===========================================
    # Broker and Backend
    # ===========================================
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,

    # ===========================================
    # Serialization
    # ===========================================
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,

    # ===========================================
    # Task Execution
    # ===========================================
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit

    # ===========================================
    # Results
    # ===========================================
    result_expires=86400,  # 24 hours

    # ===========================================
    # Worker Configuration
    # ===========================================
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (reclaim memory more frequently)
    worker_max_memory_per_child=400000,  # 400MB hard limit (KB) - auto-restart before OOM kill

    # ===========================================
    # Beat Schedule (Periodic Tasks)
    # ===========================================
    # NOTE: All tasks use the default 'celery' queue for simplicity.
    # Schedule is ordered to respect data flow:
    #   1. Ingestion fetches raw posts
    #   2. Enrichment processes pending posts
    #   3. Aggregation calculates metrics
    #   4. Views refresh with new data
    #   5. Clustering/Profiling uses aggregated data
    beat_schedule={
        # -----------------------------------------
        # DATA PIPELINE
        # -----------------------------------------
        'fetch-posts': {
            'task': 'src.tasks.ingestion_tasks.fetch_posts_task',
            'schedule': crontab(minute='*/30'),  # Every 30 min (LunarCrush: 2000 req/day limit)
            'options': {'soft_time_limit': 600, 'time_limit': 900},  # 10/15 min
        },
        'enrich-posts': {
            'task': 'src.tasks.enrichment_tasks.enrich_pending_posts_task',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
            'options': {'soft_time_limit': 240, 'time_limit': 300},  # 4/5 min
        },

        # -----------------------------------------
        # HOURLY AGGREGATION (staggered to avoid conflicts)
        # -----------------------------------------
        'aggregate-metrics': {
            'task': 'src.tasks.aggregation_tasks.aggregate_user_metrics_task',
            'schedule': crontab(minute=5),  # At :05 past every hour
        },
        'refresh-views': {
            'task': 'refresh_materialized_views',
            'schedule': crontab(minute=10),  # At :10 past every hour
        },

        # -----------------------------------------
        # ML TASKS (every 6 hours, less frequent)
        # -----------------------------------------
        'cluster-users': {
            'task': 'src.tasks.clustering_tasks.cluster_all_topics_task',
            'schedule': crontab(minute=15, hour='*/6'),  # Every 6 hours at :15
            'options': {'soft_time_limit': 2700, 'time_limit': 3300},  # 45/55 min
        },
        'build-profiles': {
            'task': 'src.tasks.profiling.build_psych_profiles',
            'schedule': crontab(minute=30, hour='*/6'),  # Every 6 hours at :30
            'args': ['politics'],  # Default topic for demo
            'options': {'soft_time_limit': 2700, 'time_limit': 3300},  # 45/55 min
        },

        # -----------------------------------------
        # MONITORING & ALERTS
        # -----------------------------------------
        'detect-anomalies': {
            'task': 'src.tasks.analytics_tasks.detect_anomalies_task',
            'schedule': crontab(minute=45, hour='*/6'),  # Every 6 hours at :45
            'options': {'soft_time_limit': 600, 'time_limit': 900},
        },
        'check-alerts': {
            'task': 'src.tasks.analytics_tasks.check_alerts_task',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
            'options': {'soft_time_limit': 60, 'time_limit': 120},
        },

        # -----------------------------------------
        # WORKER HEALTH
        # -----------------------------------------
        'worker-heartbeat': {
            'task': 'src.tasks.maintenance_tasks.worker_heartbeat_task',
            'schedule': crontab(minute='*/1'),  # Every minute
        },

        # -----------------------------------------
        # MAINTENANCE (low frequency)
        # -----------------------------------------
        'cleanup-jobs': {
            'task': 'src.tasks.maintenance_tasks.cleanup_old_jobs_task',
            'schedule': crontab(minute=0, hour=3),  # Daily at 3:00 AM
        },
        'prune-old-data': {
            'task': 'src.tasks.maintenance_tasks.prune_raw_data_task',
            'schedule': crontab(minute=0, hour=4, day_of_week=0),  # Sunday at 4:00 AM
        },
    },

    # NOTE: No task_routes configured - all tasks use default 'celery' queue
    # This simplifies deployment and debugging. Add queue routing later when:
    # - Task volume exceeds 1000/hour
    # - Need priority separation between task types
    # - ML tasks start blocking real-time ingestion
)

# Auto-discover tasks from modules
app.autodiscover_tasks([
    'src.tasks.ingestion_tasks',
    'src.tasks.enrichment_tasks',
    'src.tasks.aggregation_tasks',
    'src.tasks.clustering_tasks',
    'src.tasks.maintenance_tasks',
    'src.tasks.profiling',
    'src.tasks.analytics_tasks',
])


if __name__ == '__main__':
    app.start()
