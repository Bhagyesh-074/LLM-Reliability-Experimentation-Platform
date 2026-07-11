# AI_WORKFLOW_GUIDE.md

# How to Build the Project Using Claude, Google AI Studio, and Antigravity

## Purpose

This guide explains the exact workflow for building the project with AI
coding assistants. Follow this process throughout the project to keep
the implementation consistent with the design documents.

------------------------------------------------------------------------

# Phase 1 -- Prepare the Repository

Before asking any AI to write code:

1.  Create the repository.
2.  Add every documentation file:
    -   PRD.md
    -   REQUIREMENTS.md
    -   ARCHITECTURE.md
    -   SYSTEM_DESIGN.md
    -   DATABASE_SCHEMA.md
    -   API_SPEC.md
    -   DESIGN.md
    -   DEVELOPMENT_PLAN.md
    -   TESTING.md
    -   RULES.md
    -   AI_RULES.md
    -   AI_BUILD_PLAN.md
3.  Commit these documents before implementation begins.

Never start coding before the documentation is complete.

------------------------------------------------------------------------

# Phase 2 -- Work Epic by Epic

Do **not** ask an AI to build the whole project.

Instead:

AI_BUILD_PLAN → Epic → Task → Code → Test → Review → Commit

Complete one task before starting the next.

------------------------------------------------------------------------

# Phase 3 -- Select the Correct AI

## Claude

Use for: - Backend - FastAPI - SQLAlchemy - Services - Repositories -
Refactoring

## Google AI Studio

Use for: - Evaluation algorithms - Metrics - Statistics - Optimization -
Code review - Debugging

## Antigravity

Use for: - UI - Layouts - Components - Design system

Never use Antigravity for backend logic.

------------------------------------------------------------------------

# Phase 4 -- Feed Context

Do not paste every document every time.

Use only the documents required for the current task.

Typical mapping:

  Task        Documents
  ----------- ------------------------------------------
  Provider    SYSTEM_DESIGN, API_SPEC, DATABASE_SCHEMA
  Database    DATABASE_SCHEMA, ARCHITECTURE
  UI          PRD, DESIGN, ARCHITECTURE
  Metrics     REQUIREMENTS, SYSTEM_DESIGN
  API         API_SPEC, DATABASE_SCHEMA
  Dashboard   PRD, DESIGN

Keep context focused.

------------------------------------------------------------------------

# Phase 5 -- Prompt Structure

Always use the same structure.

## Step 1

State the task.

Example:

"I am implementing Task T04-03 from AI_BUILD_PLAN."

## Step 2

List the documents.

Example:

Read: - SYSTEM_DESIGN.md - API_SPEC.md - AI_RULES.md

## Step 3

Describe scope.

Example:

Implement only BaseLLMProvider.

Do not implement any providers.

## Step 4

State quality expectations.

Example:

-   Production quality
-   Typed Python
-   Ruff compliant
-   MyPy compliant
-   Unit tests included
-   No placeholders

------------------------------------------------------------------------

# Prompt Template

    You are an experienced senior software engineer.

    Task:
    <task id>

    Read:
    <documents>

    Implement ONLY the requested task.

    Requirements:
    - Follow AI_RULES.md
    - Follow ARCHITECTURE.md
    - Follow SYSTEM_DESIGN.md

    Do not modify unrelated files.

    Return complete production-ready code.

    Include tests.

    Stop after the task is complete.

------------------------------------------------------------------------

# Phase 6 -- Review

Never merge AI-generated code immediately.

Review:

-   Architecture
-   Naming
-   SOLID
-   Type hints
-   Security
-   Tests
-   Error handling

Ask Google AI Studio:

"Review this implementation for architectural issues, bugs, performance
problems, and edge cases."

------------------------------------------------------------------------

# Phase 7 -- Test

Run:

-   Ruff
-   Black
-   MyPy
-   Pytest

Fix all failures before continuing.

------------------------------------------------------------------------

# Phase 8 -- Commit

One task = one commit.

Example:

feat(provider): implement BaseLLMProvider

Avoid mixing unrelated work.

------------------------------------------------------------------------

# Phase 9 -- UI Workflow

1.  Ask Antigravity for UI only.
2.  Export components.
3.  Integrate with backend yourself or with Claude.
4.  Remove any fake/mock business logic.

------------------------------------------------------------------------

# Phase 10 -- When Stuck

Do not ask:

"Fix my project."

Instead ask:

-   Explain this error.
-   Review this module.
-   Improve this algorithm.
-   Refactor this class.
-   Suggest a better design.

Focused prompts produce better results.

------------------------------------------------------------------------

# Golden Rules

-   One task at a time.
-   One AI per responsibility.
-   Keep documentation authoritative.
-   Review all generated code.
-   Test every change.
-   Commit frequently.
-   Never let AI redesign t he architecture without updating
    documentation first.

------------------------------------------------------------------------

# Complete Development Flow

1.  Read AI_BUILD_PLAN.md
2.  Select the next task.
3.  Identify required design documents.
4.  Prompt the correct AI.
5.  Receive code.
6.  Review with Google AI Studio.
7.  Run linting, typing, and tests.
8.  Fix issues.
9.  Commit.
10. Repeat until the epic is complete.
11. After each epic, perform an integration test before starting the
    next one.
