"""
Data processing pipelines

This package contains:
- Data ingestion pipeline (fetch from LunarCrush)
- Emotion enrichment pipeline (analyze with Kaiko)
- User metrics aggregation pipeline (27 psychological features)
- Psychographic clustering pipeline
"""

from src.processing.ingestion_pipeline import IngestionPipeline
from src.processing.enrichment_pipeline import EnrichmentPipeline
from src.processing.metrics_pipeline import MetricsAggregationPipeline

__all__ = ["IngestionPipeline", "EnrichmentPipeline", "MetricsAggregationPipeline"]
