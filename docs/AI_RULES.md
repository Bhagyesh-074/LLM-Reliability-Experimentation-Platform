# AI_RULES.md

# AI Coding Agent Rules

## LLM Reliability & Experimentation Platform v2.0

> This document is written specifically for AI coding assistants
> (Claude, Google AI Studio, Codex, Gemini CLI, Cursor, etc.). It
> defines strict implementation constraints to keep the codebase
> consistent with the project's engineering documents.

------------------------------------------------------------------------

# Primary Directive

The project documentation is the source of truth.

Before writing any code, read and follow:

1.  PRD.md
2.  REQUIREMENTS.md
3.  ARCHITECTURE.md
4.  SYSTEM_DESIGN.md
5.  DATABASE_SCHEMA.md
6.  API_SPEC.md
7.  DEVELOPMENT_PLAN.md
8.  RULES.md

Never contradict these documents.

------------------------------------------------------------------------

# Scope Rules

-   Implement only the requested task.
-   Do not modify unrelated modules.
-   Do not implement future features.
-   Do not remove existing functionality unless explicitly requested.
-   Stop when the requested task is complete.

------------------------------------------------------------------------

# Architecture Rules

-   Do not change the folder structure.
-   Do not rename public classes, APIs, or database tables without
    explicit instruction.
-   Keep UI, business logic, providers, metrics, analytics, and
    persistence separated.
-   Follow dependency direction defined in ARCHITECTURE.md.

------------------------------------------------------------------------

# Code Generation Rules

Always produce:

-   Production-ready code
-   Fully typed Python
-   Clear docstrings
-   Small, reusable functions
-   Meaningful variable names
-   No placeholder implementations
-   No commented-out dead code

Never produce:

-   TODO-only implementations
-   Mock business logic in production code
-   Hidden side effects
-   Global mutable state without justification

------------------------------------------------------------------------

# Dependency Rules

-   Do not introduce new libraries unless required.
-   Prefer the project's existing stack.
-   Explain why a new dependency is needed before using it.

------------------------------------------------------------------------

# Database Rules

-   Use SQLAlchemy models and repositories.
-   Do not write raw SQL unless performance or migrations require it.
-   Respect foreign keys and constraints.
-   Never bypass repository abstractions.

------------------------------------------------------------------------

# Provider Rules

-   All providers implement BaseLLMProvider.
-   Never place provider-specific logic outside the provider layer.
-   Normalize responses into a common LLMResponse model.

------------------------------------------------------------------------

# Prompt & Benchmark Rules

-   Prompts are immutable once versioned.
-   Benchmarks are versioned.
-   Preserve reproducibility by storing explicit versions.

------------------------------------------------------------------------

# Testing Rules

Every functional change must include:

-   Unit tests
-   Integration tests (if interfaces changed)
-   Updated fixtures where needed

Never reduce test coverage.

------------------------------------------------------------------------

# Documentation Rules

If behavior changes:

-   Update the relevant Markdown documents.
-   Keep examples synchronized with the implementation.
-   Do not leave documentation outdated.

------------------------------------------------------------------------

# Performance Rules

-   Prefer async execution for independent I/O.
-   Cache repeated requests when safe.
-   Avoid unnecessary provider calls.
-   Measure before optimizing.

------------------------------------------------------------------------

# Security Rules

-   Never expose API keys.
-   Never commit secrets.
-   Validate uploaded files.
-   Sanitize inputs.
-   Avoid logging sensitive data.

------------------------------------------------------------------------

# Git Rules

Generate changes suitable for a single focused commit.

Suggested commit format:

    type(scope): concise description

Examples:

-   feat(provider): add Gemini provider
-   fix(metrics): correct hallucination scorer
-   refactor(api): simplify evaluation routes

------------------------------------------------------------------------

# AI-Specific Workflow

1.  Read the relevant design documents.
2.  Understand the current task.
3.  Identify affected files only.
4.  Implement the smallest complete solution.
5.  Write tests.
6.  Verify type safety and formatting.
7.  Stop.

------------------------------------------------------------------------

# Self-Review Checklist

Before returning code, verify:

-   Follows project architecture
-   No duplicated logic
-   No unnecessary abstractions
-   Uses existing utilities where appropriate
-   Includes tests
-   Does not break public interfaces
-   Matches acceptance criteria
-   Documentation updated if required

------------------------------------------------------------------------

# Preferred Tool Responsibilities

Claude: - Services - APIs - Database - Refactoring

Google AI Studio: - Algorithms - Metrics - Evaluation logic -
Optimization - Code review

Antigravity: - React UI - Layouts - Components - Design system only

------------------------------------------------------------------------

# Completion Criteria

An AI task is complete only when:

-   Requested functionality works
-   Tests pass
-   Linting passes
-   Type checking passes
-   Architecture remains intact
-   No unrelated files were modified

When in doubt, choose the simpler solution that best matches the
documented architecture.
