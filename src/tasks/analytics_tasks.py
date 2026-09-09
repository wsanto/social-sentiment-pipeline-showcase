"""
Celery Tasks for Analytics: Anomaly Detection & Alert Checking

Scheduled tasks for automated monitoring:
- Anomaly detection: z-score based detection every 6 hours
- Alert checking: data freshness + enrichment status every 5 minutes
"""

from datetime import datetime, timedelta
import numpy as np

from src.tasks.celery_app import app
from src.database.connection import get_db_session, reset_async_engine
from src.database.models import PostRaw, PostEnriched, ProcessingJob
from src.services.alerting import AlertingService, AlertConfig
from src.config.settings import settings
from src.config.logging import get_logger

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


def _get_alerting_service() -> AlertingService:
    """Create alerting service from settings."""
    config = AlertConfig(
        email_to=getattr(settings, 'alert_email_to', ''),
        email_from=getattr(settings, 'alert_email_from', 'alerts@kaikostudios.xyz'),
        provider=getattr(settings, 'alert_email_provider', 'resend'),
        api_key=getattr(settings, 'alert_email_api_key', ''),
    )
    return AlertingService(config)


# ============================================================
# ANOMALY DETECTION TASK (P0 #2)
# ============================================================

@app.task(bind=True, max_retries=2)
def detect_anomalies_task(self, hours_back: int = 6, sensitivity: float = 3.0):
    """
    Detect sentiment anomalies and send alerts.

    Runs every 6 hours via beat schedule. Checks for z-score anomalies
    in valence, arousal, toxicity, and post volume.
    """
    import asyncio
    reset_async_engine()

    logger.info(f"Starting anomaly detection (hours_back={hours_back}, sensitivity={sensitivity})")

    async def _detect():
        async with get_db_session() as db:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)

            # Get hourly statistics
            query = text("""
                SELECT
                    DATE_TRUNC('hour', pr.posted_at) as hour,
                    AVG(pe.valence) as avg_valence,
                    AVG(pe.arousal) as avg_arousal,
                    AVG(pe.toxicity_score) as avg_toxicity,
                    COUNT(*) as post_count
                FROM posts_raw pr
                JOIN posts_enriched pe ON pr.id = pe.post_id
                WHERE pr.posted_at BETWEEN :start_time AND :end_time
                GROUP BY hour
                ORDER BY hour
            """)

            result = await db.execute(query, {"start_time": start_time, "end_time": end_time})
            data = []
            for row in result:
                data.append({
                    "timestamp": row.hour,
                    "valence": float(row.avg_valence or 0),
                    "arousal": float(row.avg_arousal or 0),
                    "toxicity": float(row.avg_toxicity or 0),
                    "volume": row.post_count,
                })

            if len(data) < 5:
                logger.info("anomaly_detection_insufficient_data", data_points=len(data))
                return {"anomalies": 0, "message": "Insufficient data"}

            # Calculate z-scores
            valences = np.array([d["valence"] for d in data])
            toxicities = np.array([d["toxicity"] for d in data])
            volumes = np.array([d["volume"] for d in data])

            mean_v, std_v = np.mean(valences), np.std(valences)
            mean_t, std_t = np.mean(toxicities), np.std(toxicities)
            mean_vol, std_vol = np.mean(volumes), np.std(volumes)

            anomalies = []
            for d in data:
                z_valence = abs((d["valence"] - mean_v) / std_v) if std_v > 0 else 0
                z_toxicity = abs((d["toxicity"] - mean_t) / std_t) if std_t > 0 else 0
                z_volume = abs((d["volume"] - mean_vol) / std_vol) if std_vol > 0 else 0

                types = []
                if z_valence > sensitivity:
                    types.append("valence_spike" if d["valence"] > mean_v else "valence_drop")
                if z_toxicity > sensitivity:
                    types.append("toxicity_spike")
                if z_volume > sensitivity:
                    types.append("volume_spike" if d["volume"] > mean_vol else "volume_drop")

                if types:
                    anomalies.append({"timestamp": d["timestamp"], "types": types})

            # Filter to only actionable anomaly types
            actionable_types = {"toxicity_spike", "volume_spike", "volume_drop"}
            anomalies = [
                a for a in anomalies
                if actionable_types & set(a["types"])
            ]

            if anomalies:
                logger.warning("anomalies_detected", count=len(anomalies),
                              types=[a["types"] for a in anomalies[:5]])

                # Send alert
                alerter = _get_alerting_service()
                await alerter.send_email(
                    subject=f"Synapse Alert: {len(anomalies)} sentiment anomalies detected",
                    body=f"Detected {len(anomalies)} anomalies in the last {hours_back} hours.\n\n"
                         + "\n".join(f"  {a['timestamp']}: {', '.join(a['types'])}" for a in anomalies[:10]),
                    alert_type="anomaly",
                )
            else:
                logger.info("anomaly_detection_clean", hours=hours_back)

            return {"anomalies": len(anomalies)}

    return asyncio.run(_detect())


# ============================================================
# ALERT CHECK TASK (P0 #3)
# ============================================================

@app.task(bind=True)
def check_alerts_task(self):
    """
    Check data freshness and enrichment health.

    Runs every 5 minutes via beat schedule. Alerts on:
    - Data stale: no new posts in last 2 hours
    - Enrichment stale: no new enriched posts in last 30 minutes
    - High fallback rate: >10% of recent enrichments are fallback
    """
    import asyncio
    reset_async_engine()

    async def _check():
        async with get_db_session() as db:
            now = datetime.utcnow()
            issues = []

            # Check data freshness (last post ingested)
            r = await db.execute(select(func.max(PostRaw.fetched_at)))
            last_fetch = r.scalar()
            if last_fetch and (now - last_fetch) > timedelta(hours=2):
                hours_stale = (now - last_fetch).total_seconds() / 3600
                issues.append(f"Data stale: last post ingested {hours_stale:.1f} hours ago")

            # Check enrichment freshness
            r = await db.execute(select(func.max(PostEnriched.processed_at)))
            last_enriched = r.scalar()
            if last_enriched and (now - last_enriched) > timedelta(minutes=30):
                mins_stale = (now - last_enriched).total_seconds() / 60
                issues.append(f"Enrichment stale: last enriched {mins_stale:.0f} minutes ago")

            # Check fallback rate (last 100 enrichments)
            r = await db.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN is_fallback = true THEN 1 END) as fallback_count
                FROM (
                    SELECT is_fallback FROM posts_enriched
                    ORDER BY processed_at DESC LIMIT 100
                ) recent
            """))
            row = r.fetchone()
            if row and row.total > 0:
                fallback_rate = row.fallback_count / row.total
                if fallback_rate > 0.1:
                    issues.append(f"High fallback rate: {fallback_rate*100:.0f}% of recent enrichments are fallback (ML classifier may be down)")

            # Check pending enrichment backlog
            r = await db.execute(text("""
                SELECT COUNT(*) FROM posts_raw pr
                LEFT JOIN posts_enriched pe ON pr.id = pe.post_id
                WHERE pe.id IS NULL
            """))
            pending = r.scalar() or 0
            if pending > 5000:
                issues.append(f"Large enrichment backlog: {pending:,} posts pending")

            if issues:
                logger.warning("alert_check_issues", issues=issues)
                alerter = _get_alerting_service()
                await alerter.send_email(
                    subject=f"Synapse Alert: {len(issues)} issue(s) detected",
                    body="Pipeline health check found issues:\n\n" + "\n".join(f"  - {i}" for i in issues),
                    alert_type="health_check",
                )
            else:
                logger.info("alert_check_healthy")

            return {"issues": len(issues), "details": issues}

    return asyncio.run(_check())
