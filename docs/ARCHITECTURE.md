# ARCHITECTURE.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the overall software architecture of the platform.
The architecture follows a modular, layered design that separates user
interaction, orchestration, model providers, evaluation logic,
analytics, and persistence. Every module has a single responsibility and
can evolve independently.

------------------------------------------------------------------------

# Architectural Principles

-   Local-first by default
-   Provider agnostic
-   Modular and extensible
-   Reproducible experiments
-   Configuration over hardcoding
-   Testability and maintainability
-   Clear separation of concerns

------------------------------------------------------------------------

# High-Level Architecture

``` text
                        User
                          │
                    Streamlit UI
                          │
                 Application Layer
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
 Registry Services   Evaluation Engine   Dashboard
        │                 │                  │
        └──────────────┬──┴──────────────────┘
                       │
               Provider Router
                       │
      ┌────────┬────────┬────────┬────────┐
      │        │        │        │
   Ollama   OpenAI  Anthropic  Gemini
                       │
                  Response Cache
                       │
               Metrics & Scorers
                       │
      ┌────────┬────────┬────────────┐
      │        │        │            │
 Accuracy  Hallucination  Safety  Instruction
                       │
             Statistics & Regression
                       │
          SQLite + MLflow + Reports
```

------------------------------------------------------------------------

# Layered Architecture

## Presentation Layer

Responsibilities: - Streamlit pages - User interaction - Charts -
Tables - Forms - Progress tracking

No business logic should exist in this layer.

------------------------------------------------------------------------

## Application Layer

Coordinates all workflows.

Responsibilities:

-   Pipeline orchestration
-   Request validation
-   Dependency injection
-   Configuration loading
-   Error handling

------------------------------------------------------------------------

## Registry Layer

Contains reusable project assets.

Modules:

-   Prompt Registry
-   Benchmark Registry
-   Provider Registry
-   Configuration Registry

Every asset is versioned.

------------------------------------------------------------------------

## Evaluation Layer

Core of the platform.

Pipeline:

Dataset → Prompt → Provider → Response → Metrics → Statistics → Storage

Supports:

-   synchronous execution
-   asynchronous execution
-   batching
-   caching
-   retries

------------------------------------------------------------------------

## Provider Layer

Every provider implements the same interface.

``` python
class BaseLLMProvider:

    def generate(self, request):
        pass
```

Implementations:

-   OllamaProvider
-   OpenAIProvider
-   AnthropicProvider
-   GeminiProvider

The rest of the application never depends on provider-specific SDKs.

------------------------------------------------------------------------

## Metrics Layer

Independent metric modules.

Examples:

-   AccuracyScorer
-   HallucinationScorer
-   InstructionScorer
-   SafetyScorer
-   CostScorer
-   LatencyScorer

Each scorer receives:

Response

Ground Truth

Metadata

Returns

``` python
MetricResult
```

------------------------------------------------------------------------

## Statistics Layer

Computes:

-   averages
-   confidence intervals
-   regression detection
-   pairwise comparison
-   score normalization

Produces dashboard-ready outputs.

------------------------------------------------------------------------

## Persistence Layer

SQLite

Stores:

-   evaluations
-   prompts
-   datasets
-   runs
-   baselines

MLflow

Stores:

-   metrics
-   parameters
-   artifacts
-   experiment metadata

------------------------------------------------------------------------

# Request Flow

``` text
User
 │
 ▼
Select Benchmark
 │
 ▼
Select Prompt
 │
 ▼
Select Model
 │
 ▼
Evaluation Engine
 │
 ▼
Provider Router
 │
 ▼
LLM Provider
 │
 ▼
Response
 │
 ▼
Metric Scorers
 │
 ▼
Statistics
 │
 ▼
SQLite
 │
 ▼
MLflow
 │
 ▼
Dashboard
```

------------------------------------------------------------------------

# Folder Structure

``` text
app/
core/
providers/
registry/
benchmarks/
datasets/
metrics/
statistics/
analytics/
dashboard/
database/
cache/
reports/
plugins/
tests/
configs/
docker/
docs/
```

------------------------------------------------------------------------

# Design Patterns

-   Strategy Pattern (provider abstraction)
-   Factory Pattern (provider creation)
-   Repository Pattern (database access)
-   Dependency Injection
-   Observer Pattern (progress updates)
-   Plugin Architecture (custom metrics)

------------------------------------------------------------------------

# Scalability Strategy

Current: - Local execution - SQLite - Streamlit

Future: - PostgreSQL - Redis cache - Celery workers - Kubernetes -
Distributed evaluation

------------------------------------------------------------------------

# Security Architecture

-   API keys loaded from environment variables
-   No persistent API-key storage
-   Input validation
-   Dataset schema validation
-   Sanitized file uploads
-   Audit logging for evaluations

------------------------------------------------------------------------

# Error Handling

Recoverable: - API timeout - Network error - Rate limit

Fatal: - Invalid dataset - Invalid prompt - Corrupted configuration

Retry strategy: - exponential backoff - configurable retry count

------------------------------------------------------------------------

# Future Architecture

Potential additions:

-   FastAPI REST API
-   Multi-user authentication
-   RBAC
-   Distributed worker queues
-   Vector database integration
-   Human review workflow
-   CI/CD evaluation pipeline

------------------------------------------------------------------------

# Architecture Goals

The architecture is designed to be:

-   Modular
-   Extensible
-   Testable
-   Reproducible
-   Provider agnostic
-   Production inspired
-   Easy to understand
-   Suitable for future enterprise-scale evolution
