# TESTING.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the testing strategy for the platform. It ensures
every component is verified independently and as part of the complete
evaluation workflow. The goal is to guarantee correctness,
reproducibility, reliability, and maintainability.

------------------------------------------------------------------------

# Testing Objectives

-   Verify functional correctness
-   Detect regressions early
-   Ensure reproducibility
-   Validate provider integrations
-   Maintain high code quality
-   Support safe refactoring

------------------------------------------------------------------------

# Testing Pyramid

``` text
                End-to-End Tests
             ----------------------
            Integration Tests
        ----------------------------
              Unit Tests
```

Target Distribution: - Unit Tests: \~70% - Integration Tests: \~20% -
End-to-End Tests: \~10%

------------------------------------------------------------------------

# Unit Testing

Framework: **pytest**

Each module must have dedicated tests.

## Components

### Providers

-   Request construction
-   Response normalization
-   Error handling
-   Retry logic

### Metrics

-   Accuracy scorer
-   Hallucination scorer
-   Instruction scorer
-   Safety scorer
-   Cost scorer
-   Latency scorer

### Registries

-   Prompt CRUD
-   Prompt versioning
-   Dataset validation
-   Configuration loading

### Statistics

-   Aggregation
-   Confidence intervals
-   Pairwise comparison
-   Regression detection

### Persistence

-   Repository CRUD
-   Transactions
-   Constraints

Target Coverage: **90%+**

------------------------------------------------------------------------

# Integration Testing

Verify interactions between modules.

## Scenarios

-   UI → Evaluation Engine
-   Evaluation Engine → Provider
-   Provider → Metrics
-   Metrics → Database
-   Database → Dashboard
-   Evaluation → MLflow

Use temporary SQLite databases and mocked providers where appropriate.

------------------------------------------------------------------------

# End-to-End Testing

Execute complete workflows.

## Example

1.  Load benchmark
2.  Select provider
3.  Run evaluation
4.  Persist results
5.  Log to MLflow
6.  Display dashboard
7.  Export report

Expected outcome: - No errors - Results persisted - Metrics displayed
correctly

------------------------------------------------------------------------

# Performance Testing

Measure:

-   Evaluation throughput
-   Average latency
-   Dashboard responsiveness
-   Database query time
-   Memory consumption

Target: - 100 benchmark questions in under 5 minutes (recommended
hardware).

------------------------------------------------------------------------

# Regression Testing

Maintain fixed benchmark datasets.

Compare: - Composite score - Individual metrics - Runtime - Cost

Trigger regression alert if configured thresholds are exceeded.

------------------------------------------------------------------------

# Security Testing

Validate:

-   API key handling
-   File upload validation
-   Prompt injection resistance
-   Input sanitization
-   SQL injection protection

------------------------------------------------------------------------

# Compatibility Testing

Supported:

-   Windows
-   macOS
-   Linux

Python: - 3.11+ - 3.12+

Providers: - Ollama - OpenAI - Anthropic - Gemini

------------------------------------------------------------------------

# Mocking Strategy

Mock: - External APIs - Network failures - Rate limits - MLflow -
Time-dependent operations

Avoid mocking internal business logic.

------------------------------------------------------------------------

# Test Data

Maintain versioned fixtures:

-   Medical benchmark
-   Legal benchmark
-   Finance benchmark
-   Adversarial prompts
-   Invalid datasets

Keep fixtures deterministic.

------------------------------------------------------------------------

# Automation

Run on every pull request:

-   Ruff
-   Black
-   MyPy
-   Unit tests
-   Integration tests

Nightly: - End-to-end tests - Performance benchmarks

------------------------------------------------------------------------

# Code Coverage

Minimum thresholds:

-   Overall: 85%
-   Metrics: 95%
-   Providers: 90%
-   Registries: 90%
-   Evaluation Engine: 90%

Coverage failures block merges.

------------------------------------------------------------------------

# Failure Reporting

Every failed test should report: - Test name - Component - Expected
result - Actual result - Stack trace - Related run ID (if applicable)

------------------------------------------------------------------------

# Test Directory Structure

``` text
tests/
├── unit/
├── integration/
├── e2e/
├── performance/
├── security/
├── fixtures/
├── mocks/
└── conftest.py
```

------------------------------------------------------------------------

# Continuous Integration

Pipeline stages:

1.  Lint
2.  Type Check
3.  Unit Tests
4.  Integration Tests
5.  Build
6.  Artifact Generation

Main branch additionally runs: - E2E suite - Performance suite

------------------------------------------------------------------------

# Definition of Test Success

A release is test-ready when:

-   All automated tests pass
-   Coverage thresholds are met
-   No critical security findings
-   Regression checks pass
-   Dashboard workflows complete successfully
-   MLflow logs are generated correctly

------------------------------------------------------------------------

# Future Enhancements

-   Browser automation (Playwright)
-   Load testing (Locust)
-   Chaos testing
-   Mutation testing
-   Continuous benchmarking
-   Cross-provider comparison suite
