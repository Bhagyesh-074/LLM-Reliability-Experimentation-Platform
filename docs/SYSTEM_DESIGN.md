# SYSTEM_DESIGN.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the low-level system design (LLD) of the platform.
It describes components, interfaces, data flow, internal services,
database interactions, class responsibilities, and execution sequence.
It complements `ARCHITECTURE.md` by explaining **how** the system is
implemented.

------------------------------------------------------------------------

# Design Goals

-   High cohesion, low coupling
-   Provider independence
-   Testable components
-   Reproducible evaluations
-   Extensible metric framework
-   Configuration-driven behavior

------------------------------------------------------------------------

# Core Components

## 1. UI Layer

Responsibilities: - Collect user input - Display progress - Render
charts and reports - Trigger evaluation workflows

Never contains evaluation logic.

------------------------------------------------------------------------

## 2. Evaluation Orchestrator

Coordinates the complete evaluation lifecycle.

Responsibilities: - Validate configuration - Load benchmark - Resolve
prompt version - Create evaluation jobs - Dispatch provider requests -
Collect responses - Invoke scorers - Persist results - Trigger analytics

Input: - EvaluationConfig

Output: - EvaluationRun

------------------------------------------------------------------------

## 3. Provider Router

Chooses the correct provider implementation.

``` text
Evaluation Request
        │
        ▼
 Provider Router
 ├── OllamaProvider
 ├── OpenAIProvider
 ├── AnthropicProvider
 └── GeminiProvider
```

Selection is driven entirely by configuration.

------------------------------------------------------------------------

## 4. Provider Interface

``` python
class BaseLLMProvider:
    def generate(request: LLMRequest) -> LLMResponse:
        ...
```

Every provider must: - accept identical requests - return identical
response objects - normalize provider-specific metadata

------------------------------------------------------------------------

## 5. Registry Services

### Prompt Registry

Stores immutable prompt versions.

### Benchmark Registry

Stores datasets and metadata.

### Provider Registry

Lists available providers and capabilities.

### Configuration Registry

Loads YAML configuration and validates schemas.

------------------------------------------------------------------------

## 6. Evaluation Pipeline

``` text
Dataset
   │
Prompt Resolution
   │
Request Builder
   │
Provider
   │
Response
   │
Metric Engine
   │
Statistics
   │
Persistence
   │
Dashboard
```

------------------------------------------------------------------------

## 7. Metric Engine

Each scorer implements:

``` python
class Metric:
    def evaluate(response, reference, metadata) -> MetricResult:
        ...
```

Independent scorers: - Accuracy - Hallucination - Instruction - Safety -
Latency - Cost - Consistency

Composite score is computed after all metrics finish.

------------------------------------------------------------------------

## 8. Statistics Engine

Responsibilities: - Aggregate metrics - Normalize scores - Pairwise
comparison - Confidence intervals - Regression detection - Domain-wise
summaries

Produces analytics-ready objects.

------------------------------------------------------------------------

## 9. Persistence Service

### SQLite

Tables: - prompts - prompt_versions - benchmarks - runs - results -
metrics - baselines

### MLflow

Logs: - parameters - metrics - artifacts - environment metadata

------------------------------------------------------------------------

# Sequence Diagram

``` text
User
 │
 ▼
Streamlit UI
 │
 ▼
Evaluation Orchestrator
 │
 ▼
Provider Router
 │
 ▼
Selected Provider
 │
 ▼
LLM Response
 │
 ▼
Metric Engine
 │
 ▼
Statistics Engine
 │
 ▼
SQLite + MLflow
 │
 ▼
Dashboard Update
```

------------------------------------------------------------------------

# Data Objects

## EvaluationConfig

-   provider
-   model
-   prompt_version
-   benchmark
-   temperature
-   max_tokens

## LLMRequest

-   system_prompt
-   user_prompt
-   parameters

## LLMResponse

-   text
-   latency
-   token_usage
-   finish_reason
-   raw_metadata

## MetricResult

-   metric_name
-   score
-   explanation
-   confidence

## EvaluationRun

-   run_id
-   timestamps
-   provider
-   metrics
-   artifacts

------------------------------------------------------------------------

# Internal Services

-   CacheService
-   RetryService
-   ValidationService
-   ReportService
-   AnalyticsService
-   MLflowService
-   DatabaseService

Each service exposes a small, well-defined interface.

------------------------------------------------------------------------

# Error Flow

Recoverable: - timeout - temporary provider failure - rate limiting

Handling: 1. retry 2. fallback 3. mark failed item 4. continue remaining
evaluation

Fatal: - invalid benchmark - invalid prompt - corrupted configuration

------------------------------------------------------------------------

# Configuration Model

Configuration is YAML driven.

``` yaml
provider: openai
model: gpt-4o
temperature: 0.3
benchmark: medical_v1
prompt: medical_v3
metrics:
  - accuracy
  - hallucination
  - instruction
```

------------------------------------------------------------------------

# Extensibility

Adding a provider: 1. Implement BaseLLMProvider 2. Register provider 3.
Add configuration

Adding a metric: 1. Implement Metric interface 2. Register scorer 3.
Dashboard automatically discovers metric

------------------------------------------------------------------------

# Testing Strategy

Unit Tests - Providers - Metrics - Registries - Statistics

Integration Tests - End-to-end evaluation - Database persistence -
MLflow logging

Regression Tests - Metric stability - Dataset compatibility

------------------------------------------------------------------------

# Design Decisions

  Decision               Rationale
  ---------------------- -------------------------------------
  Provider abstraction   Avoid vendor lock-in
  Registry pattern       Version everything
  YAML configuration     Reproducibility
  SQLite + MLflow        Lightweight but production-inspired
  Modular metrics        Easy extensibility
  Async-ready pipeline   Faster evaluation

------------------------------------------------------------------------

# Future Enhancements

-   Distributed execution workers
-   PostgreSQL backend
-   REST API
-   Human review queues
-   Kubernetes deployment
-   Multi-user workspaces
-   Continuous evaluation in CI/CD
