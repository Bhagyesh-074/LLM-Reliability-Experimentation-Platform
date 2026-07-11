# API_SPEC.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the internal REST API contracts used by the
platform. The API provides a stable interface between the UI, evaluation
engine, registries, analytics services, and future external
integrations.

------------------------------------------------------------------------

# API Principles

-   RESTful design
-   JSON request/response bodies
-   Versioned endpoints (`/api/v1`)
-   Consistent error model
-   Idempotent read operations
-   OpenAPI-compatible schemas

------------------------------------------------------------------------

# Authentication

**MVP** - Local-only deployment - Authentication disabled by default

**Future** - API Keys - OAuth2 - JWT - Role-Based Access Control (RBAC)

------------------------------------------------------------------------

# Standard Response Format

## Success

``` json
{
  "success": true,
  "data": {},
  "message": "Operation completed."
}
```

## Error

``` json
{
  "success": false,
  "error": {
    "code": "INVALID_DATASET",
    "message": "Dataset schema validation failed.",
    "details": {}
  }
}
```

------------------------------------------------------------------------

# Health

## GET /api/v1/health

Checks application status.

Response

``` json
{
  "status":"healthy",
  "database":"connected",
  "mlflow":"running",
  "providers":["ollama","openai"]
}
```

------------------------------------------------------------------------

# Providers

## GET /api/v1/providers

Returns supported providers.

## GET /api/v1/providers/models

Returns available models.

Query Parameters

-   provider

------------------------------------------------------------------------

# Prompt Registry

## GET /api/v1/prompts

List prompts.

Supports filtering by:

-   tag
-   author
-   status

------------------------------------------------------------------------

## POST /api/v1/prompts

Create a prompt.

Body

``` json
{
  "name":"Medical QA",
  "content":"...",
  "tags":["medical"]
}
```

------------------------------------------------------------------------

## GET /api/v1/prompts/{id}

Fetch prompt details.

------------------------------------------------------------------------

## POST /api/v1/prompts/{id}/versions

Create a new immutable prompt version.

------------------------------------------------------------------------

# Benchmark Registry

## GET /api/v1/benchmarks

List benchmarks.

## POST /api/v1/benchmarks

Upload CSV/JSON benchmark.

Returns validation report.

## GET /api/v1/benchmarks/{id}

Retrieve benchmark metadata.

------------------------------------------------------------------------

# Evaluations

## POST /api/v1/evaluations

Start an evaluation.

Body

``` json
{
  "provider":"openai",
  "model":"gpt-4o",
  "benchmark":"medical_v1",
  "prompt_version":"v3",
  "temperature":0.3
}
```

Response

``` json
{
  "run_id":"run_001",
  "status":"queued"
}
```

------------------------------------------------------------------------

## GET /api/v1/evaluations/{run_id}

Returns evaluation status.

Possible values:

-   queued
-   running
-   completed
-   failed

------------------------------------------------------------------------

## GET /api/v1/evaluations/{run_id}/results

Returns detailed results, metrics, and artifacts.

------------------------------------------------------------------------

# Metrics

## GET /api/v1/metrics

Returns supported metrics.

## GET /api/v1/runs/{run_id}/metrics

Returns metric breakdown for a completed run.

------------------------------------------------------------------------

# Leaderboard

## GET /api/v1/leaderboard

Query Parameters

-   benchmark
-   provider
-   model
-   date_range

Returns ranked evaluation results.

------------------------------------------------------------------------

# Failure Analysis

## GET /api/v1/failures

Supports filters:

-   provider
-   benchmark
-   metric
-   domain
-   difficulty

Returns failed responses with explanations.

------------------------------------------------------------------------

# Analytics

## GET /api/v1/analytics/trends

Returns historical trends.

## GET /api/v1/analytics/regressions

Returns detected regressions.

## GET /api/v1/analytics/cost

Returns token usage, latency, and estimated cost.

------------------------------------------------------------------------

# Plugins

## GET /api/v1/plugins

Lists installed plugins.

## POST /api/v1/plugins/reload

Reload plugin registry.

------------------------------------------------------------------------

# Error Codes

  Code                Meaning
  ------------------- -----------------------------
  INVALID_DATASET     Dataset schema invalid
  INVALID_PROMPT      Prompt validation failed
  PROVIDER_ERROR      Provider unavailable
  MODEL_NOT_FOUND     Requested model unavailable
  RATE_LIMITED        Upstream rate limited
  EVALUATION_FAILED   Evaluation terminated
  INTERNAL_ERROR      Unexpected server error

------------------------------------------------------------------------

# Status Codes

  HTTP   Meaning
  ------ ------------------
  200    Success
  201    Created
  202    Accepted
  400    Bad Request
  401    Unauthorized
  403    Forbidden
  404    Not Found
  409    Conflict
  422    Validation Error
  429    Rate Limited
  500    Internal Error

------------------------------------------------------------------------

# Versioning

All endpoints are versioned under:

    /api/v1/

Breaking changes require a new major API version.

------------------------------------------------------------------------

# Future Endpoints

-   `/api/v2/evaluations/stream`
-   `/api/v2/reviews`
-   `/api/v2/workspaces`
-   `/api/v2/auth`
-   `/api/v2/webhooks`
-   `/api/v2/agents`
