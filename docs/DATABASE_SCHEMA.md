# DATABASE_SCHEMA.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the logical and physical database design for the
platform. The schema is normalized to support reproducible evaluations,
versioned assets, experiment tracking, analytics, and future
scalability.

------------------------------------------------------------------------

# Database Choice

**Primary (MVP):** SQLite

**Future Options:** - PostgreSQL - MySQL (supported with SQLAlchemy)

ORM: **SQLAlchemy**

Migration Tool: **Alembic**

------------------------------------------------------------------------

# Entity Relationship Overview

``` text
Providers ─────┐
               │
Prompts ──< PromptVersions ──┐
                             │
Benchmarks ──< DatasetVersions│
                             │
                             ▼
                      EvaluationRuns
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
         EvaluationResults         RunMetrics
                │
                ▼
        FailureAnalysis
```

------------------------------------------------------------------------

# Core Tables

## providers

  Column        Type        Constraints
  ------------- ----------- -------------
  provider_id   UUID        PK
  name          TEXT        UNIQUE
  type          TEXT        local/api
  sdk_version   TEXT        
  created_at    TIMESTAMP   

------------------------------------------------------------------------

## prompts

  Column        Type
  ------------- -----------
  prompt_id     UUID
  name          TEXT
  description   TEXT
  author        TEXT
  status        TEXT
  created_at    TIMESTAMP

------------------------------------------------------------------------

## prompt_versions

  Column         Type
  -------------- -----------
  version_id     UUID
  prompt_id      FK
  version        INTEGER
  content        TEXT
  content_hash   TEXT
  tags           JSON
  created_at     TIMESTAMP

One prompt can have many immutable versions.

------------------------------------------------------------------------

## benchmarks

  Column         Type
  -------------- ------
  benchmark_id   UUID
  name           TEXT
  domain         TEXT
  description    TEXT

------------------------------------------------------------------------

## dataset_versions

  Column               Type
  -------------------- -----------
  dataset_version_id   UUID
  benchmark_id         FK
  version              TEXT
  question_count       INTEGER
  checksum             TEXT
  created_at           TIMESTAMP

------------------------------------------------------------------------

## evaluation_runs

  Column               Type
  -------------------- -----------
  run_id               UUID
  provider_id          FK
  model_name           TEXT
  prompt_version_id    FK
  dataset_version_id   FK
  temperature          REAL
  max_tokens           INTEGER
  composite_score      REAL
  started_at           TIMESTAMP
  completed_at         TIMESTAMP
  status               TEXT

Indexes: - model_name - status - started_at

------------------------------------------------------------------------

## evaluation_results

  Column         Type
  -------------- ---------
  result_id      UUID
  run_id         FK
  question_id    TEXT
  question       TEXT
  ground_truth   TEXT
  response       TEXT
  latency_ms     INTEGER
  token_usage    INTEGER

One row per evaluated question.

------------------------------------------------------------------------

## run_metrics

  Column          Type
  --------------- ------
  metric_id       UUID
  run_id          FK
  accuracy        REAL
  hallucination   REAL
  instruction     REAL
  safety          REAL
  latency         REAL
  cost            REAL
  consistency     REAL

------------------------------------------------------------------------

## failure_analysis

  Column        Type
  ------------- ------
  failure_id    UUID
  result_id     FK
  category      TEXT
  explanation   TEXT
  severity      TEXT

Categories: - Hallucination - Factual Error - Reasoning Error -
Formatting Error - Refusal - Safety Issue

------------------------------------------------------------------------

## mlflow_runs

  Column            Type
  ----------------- ------
  mlflow_run_id     TEXT
  run_id            FK
  artifact_path     TEXT
  experiment_name   TEXT

------------------------------------------------------------------------

# Relationships

-   Provider → EvaluationRuns (1:N)
-   Prompt → PromptVersions (1:N)
-   Benchmark → DatasetVersions (1:N)
-   PromptVersion → EvaluationRuns (1:N)
-   DatasetVersion → EvaluationRuns (1:N)
-   EvaluationRun → EvaluationResults (1:N)
-   EvaluationRun → RunMetrics (1:1)
-   EvaluationResult → FailureAnalysis (1:N)

------------------------------------------------------------------------

# Indexing Strategy

Indexes recommended on: - model_name - provider_id - prompt_version_id -
dataset_version_id - benchmark_id - status - created_at - category

------------------------------------------------------------------------

# Constraints

-   Prompt versions are immutable.
-   Dataset versions are immutable.
-   Foreign keys enforced.
-   Composite score derived, not manually edited.
-   API keys are never stored.

------------------------------------------------------------------------

# Example DDL

``` sql
CREATE TABLE evaluation_runs (
  run_id TEXT PRIMARY KEY,
  provider_id TEXT NOT NULL,
  prompt_version_id TEXT NOT NULL,
  dataset_version_id TEXT NOT NULL,
  model_name TEXT,
  temperature REAL,
  composite_score REAL,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  status TEXT
);
```

------------------------------------------------------------------------

# Migration Strategy

-   Alembic manages schema versions.
-   Backward-compatible changes preferred.
-   Breaking schema changes require migration scripts.

------------------------------------------------------------------------

# Query Examples

Top models:

``` sql
SELECT model_name, AVG(composite_score)
FROM evaluation_runs
GROUP BY model_name
ORDER BY AVG(composite_score) DESC;
```

Recent failures:

``` sql
SELECT category, explanation
FROM failure_analysis
ORDER BY failure_id DESC
LIMIT 20;
```

------------------------------------------------------------------------

# Future Extensions

-   PostgreSQL partitioning
-   Materialized views
-   Time-series analytics
-   Multi-tenant schemas
-   Vector embeddings table
-   Human review annotations
-   Audit log tables
