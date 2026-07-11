# RULES.md

# LLM Reliability & Experimentation Platform v2.0

## Purpose

This document defines the engineering, coding, architectural, and
collaboration rules that every contributor and AI coding assistant must
follow. These rules ensure the project remains consistent, maintainable,
reproducible, and aligned with the design documents.

------------------------------------------------------------------------

# Rule 1 --- Documentation is the Source of Truth

Before implementing any feature, consult:

1.  PRD.md
2.  REQUIREMENTS.md
3.  ARCHITECTURE.md
4.  SYSTEM_DESIGN.md
5.  DATABASE_SCHEMA.md
6.  API_SPEC.md

Implementation must conform to these documents. Do not invent new
architecture or APIs without updating the documentation first.

------------------------------------------------------------------------

# Rule 2 --- One Task at a Time

-   Implement only the assigned task.
-   Do not implement future tasks preemptively.
-   Keep pull requests and commits focused.

------------------------------------------------------------------------

# Rule 3 --- Respect the Architecture

-   UI must not contain business logic.
-   Business logic must not directly manipulate UI.
-   Providers must only communicate through the Provider interface.
-   Database access must go through repositories/services.

------------------------------------------------------------------------

# Rule 4 --- Provider Independence

Never write provider-specific logic outside the Provider layer.

All providers must implement the common `BaseLLMProvider` interface.

------------------------------------------------------------------------

# Rule 5 --- Version Everything

The following must be versioned:

-   Prompts
-   Benchmarks
-   Configuration
-   Database schema
-   APIs
-   Documentation

Evaluation runs must always reference explicit versions.

------------------------------------------------------------------------

# Rule 6 --- Configuration Over Hardcoding

Do not hardcode: - API keys - Model names - Prompt text - Dataset
paths - Thresholds - URLs

Use configuration files or environment variables.

------------------------------------------------------------------------

# Rule 7 --- Code Quality

Every contribution must:

-   Use Python type hints
-   Include meaningful docstrings
-   Follow Ruff and Black formatting
-   Pass MyPy type checking
-   Avoid duplicated logic

------------------------------------------------------------------------

# Rule 8 --- Testing is Mandatory

Every new feature requires appropriate tests:

-   Unit tests
-   Integration tests (when applicable)
-   Regression tests for evaluation logic

No feature is complete without automated tests.

------------------------------------------------------------------------

# Rule 9 --- Error Handling

-   Fail gracefully.
-   Provide actionable error messages.
-   Never silently ignore exceptions.
-   Retry transient failures using exponential backoff where
    appropriate.

------------------------------------------------------------------------

# Rule 10 --- Logging

Important workflows must log:

-   Run ID
-   Provider
-   Model
-   Prompt version
-   Dataset version
-   Execution time
-   Errors

Never log secrets or API keys.

------------------------------------------------------------------------

# Rule 11 --- Security

-   Never commit secrets.
-   Never persist API keys.
-   Validate uploaded datasets.
-   Sanitize user input.
-   Keep dependencies updated.

------------------------------------------------------------------------

# Rule 12 --- Performance

-   Prefer asynchronous execution where beneficial.
-   Cache repeat evaluations.
-   Avoid unnecessary API calls.
-   Profile before optimizing.

------------------------------------------------------------------------

# Rule 13 --- Git Workflow

Branches:

-   main
-   develop
-   feature/`<task-id>`{=html}
-   hotfix/`<issue>`{=html}

Commits should be small and descriptive.

Example:

    feat(provider): implement OpenAIProvider

------------------------------------------------------------------------

# Rule 14 --- AI Assistant Usage

## Claude

Use for: - Backend - Services - APIs - Database - Refactoring

## Google AI Studio

Use for: - Algorithms - Metrics - Optimization - Code reviews -
Debugging

## Antigravity

Use for: - UI - Components - Layouts - Design system

Do not ask any AI to build unrelated features outside the current task.

------------------------------------------------------------------------

# Rule 15 --- Definition of Done

A task is complete only if:

-   Requirements satisfied
-   Tests pass
-   Documentation updated
-   Linting passes
-   Type checking passes
-   Architecture remains consistent

------------------------------------------------------------------------

# Rule 16 --- Continuous Improvement

When improving the project:

1.  Identify the problem.
2.  Update documentation if architecture changes.
3.  Implement the change.
4.  Add tests.
5.  Review impact on existing modules.

------------------------------------------------------------------------

# Rule 17 --- Review Checklist

Before merging:

-   Architecture respected
-   No duplicated code
-   No TODO placeholders
-   Tests green
-   Documentation updated
-   Security considerations reviewed

------------------------------------------------------------------------

# Guiding Principles

-   Build for maintainability, not shortcuts.
-   Optimize only after measuring.
-   Prefer simple, modular solutions.
-   Preserve reproducibility.
-   Keep interfaces stable.
-   Let documentation drive implementation.
