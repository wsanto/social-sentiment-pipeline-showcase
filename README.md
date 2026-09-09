# Social Sentiment Data Pipeline (showcase excerpt)

An excerpt from a production data pipeline that ingests social media content via a third-party
data provider, enriches it with sentiment/emotion analysis, and tracks trends over time. This repo
shows the **pipeline architecture** — task orchestration, ingestion/enrichment/metrics stages, ML
clustering, and the API layer — as a demonstration of building a real-time data pipeline, not any
of the actual data it processes.

**No real data included.** This pipeline runs against a live third-party API and tracks real
social media activity in production. Nothing it has collected — no posts, no accounts, no brand or
topic configuration beyond the two generic, code-defined examples below — is included here. Every
file was checked for hardcoded real-world references before being added to this excerpt.

This is a curated excerpt, not the full pipeline: several modules import internals that live in the
private codebase, so this repo is for reading, not running.

## What's included here

- **`src/tasks/`** — the Celery task layer: ingestion, enrichment, aggregation, analytics,
  clustering, and maintenance tasks, plus the app setup.
- **`src/processing/`** — the three pipeline stages (ingestion, enrichment, metrics) the tasks
  drive.
- **`src/ml/`** — clustering pipeline, feature engineering, and model persistence.
- **`src/api/`** — the FastAPI app, auth, schemas, and two representative routes: `brands.py`
  (brands are database-configured monitoring targets, not hardcoded) and `topics.py` (a static
  topic → sub-topic keyword taxonomy, e.g. "politics" → elections/legislation/etc.).
- **`src/database/`** — connection handling, ORM models, and utilities.
- **`src/integrations/`** — typed clients for the third-party data provider and an internal
  service, both reading credentials from environment variables.
- **`src/config/`** — settings and logging setup.

## What was built but isn't shown here

- **Influencer tracking and profiling services** — identifying influencers and measuring sentiment
  response to their posts.
- **Most of the API surface** — dashboards, monitoring/alerting, archetype and event endpoints.
- **Superset dashboard configuration**, deployment infrastructure (Railway, Celery/Redis ops docs),
  and all operational runbooks.
- Any actual collected data, database backups, or logs — none of that left the private repo.

I'm happy to walk through the design of any of these in conversation — they're just not published
as code.

## Stack

Python, FastAPI, Celery + Redis for async task processing, SQLAlchemy/PostgreSQL, scikit-learn-style
ML clustering, a third-party social data API integration.
