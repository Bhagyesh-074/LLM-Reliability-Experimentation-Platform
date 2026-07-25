# LLM Reliability & Experimentation Platform

A production-quality platform for evaluating, benchmarking, and comparing Large Language Models across multiple providers using semantic accuracy scoring, NLI-based hallucination detection, statistical regression analysis, and MLflow experiment tracking.

## Features

| Module | Description |
|---|---|
| Provider System | Unified interface across Ollama, OpenAI, Anthropic, Google Gemini |
| Prompt Registry | Versioned, immutable prompts with `{variable}` templating and syntax validation |
| Benchmark Registry | CSV/JSON dataset upload with schema validation and versioning |
| Evaluation Engine | Async pipeline: dataset → prompt → provider → response → scoring → persistence |
| Metrics Engine | Accuracy, Hallucination, Instruction Following, Safety, Latency, Cost + Composite Score |
| Failure Analysis | Automatic categorization (Hallucination, Factual Error, Formatting, Safety, Refusal) with severity |
| Statistics Engine | Confidence intervals, pairwise model comparison, regression detection |
| MLflow Integration | Full experiment tracking — parameters, metrics, artifacts |
| Dashboard | 9-page Streamlit dashboard with leaderboard, radar charts, trends, cost/latency analysis |

## Project Structure

```
LLM-Reliability-Experimentation-Platform/
├── core/
│   ├── config.py              # Configuration loader
│   ├── logger.py              # Logging setup
│   └── evaluation/
│       ├── config.py          # EvaluationConfig
│       ├── orchestrator.py    # EvaluationOrchestrator — core pipeline
│       ├── persistence.py     # Result/metric persistence
│       ├── request_builder.py # Prompt to LLMRequest builder
│       └── mlflow_tracker.py  # MLflow experiment logging
├── providers/
│   ├── base.py                 # BaseLLMProvider interface
│   ├── ollama_provider.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   ├── factory.py               # ProviderFactory
│   └── registry.py              # ProviderRegistry
├── registry/
│   ├── prompt_service.py         # Prompt CRUD and versioning
│   ├── benchmark_service.py      # Benchmark CRUD and validation
│   └── dataset_validator.py      # CSV schema validation
├── metrics/
│   ├── base.py                    # Metric interface
│   ├── accuracy.py                # Semantic similarity scorer
│   ├── hallucination.py           # NLI-based scorer
│   ├── instruction.py             # Rule-based instruction scorer
│   ├── safety.py                  # Blocklist-based safety scorer
│   ├── latency.py
│   ├── cost.py
│   └── composite.py                # Weighted composite scorer
├── statistics/
│   ├── aggregation.py
│   ├── confidence.py               # Confidence interval calculator
│   ├── pairwise.py                 # Model comparison (t-test)
│   └── regression.py                # Regression detection
├── database/
│   ├── models.py                    # SQLAlchemy models
│   ├── session.py                   # Engine and session factory
│   └── repositories/                # Repository pattern (CRUD)
├── dashboard/
│   ├── app.py                        # Streamlit entry point
│   ├── pages/                        # 9 pages (dashboard, evaluation, results, etc.)
│   └── components/                    # Reusable charts, sidebar
├── tests/
│   └── unit/                          # 300+ unit tests
├── configs/
│   ├── default.yaml
│   └── safety_blocklist.yaml
├── docker/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Add at least one provider key (Ollama requires none):

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

### 4. Run Tests

```bash
pytest tests/
```

### 5. Run with Docker

```bash
docker compose build
docker compose up
```

## Dashboard Pages

| Page | Description |
|---|---|
| Dashboard | Leaderboard, radar chart comparison, regression alerts, recent runs |
| Providers | Configure and test connections for all 4 providers |
| Prompts | Create, version, and manage prompt templates |
| Benchmarks | Upload and validate CSV datasets |
| Evaluation | Configure and run a full evaluation with live progress |
| Results | Per-question breakdown with metric scores and status |
| Failures | Categorized failure browser with filters and case detail |
| Analytics | Trend lines, cost comparison, latency comparison, temperature vs accuracy |
| Settings | MLflow tracking URI, default temperature, regression threshold |

## Metrics Explained

| Metric | Method |
|---|---|
| Accuracy | Sentence embeddings and cosine similarity against ground truth |
| Hallucination | NLI model checks entailment between response and reference |
| Instruction Following | Deterministic rule checks (format, length, keywords) |
| Safety | Configurable keyword blocklist (violence, self-harm, hate speech, sexual content) |
| Latency | Tiered scoring based on response time |
| Cost | Tiered scoring based on token usage |
| Composite | Weighted average of all six metrics |

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Enables OpenAI provider |
| `ANTHROPIC_API_KEY` | Enables Anthropic provider |
| `GEMINI_API_KEY` | Enables Google Gemini provider |
| `DATABASE_URL` | Defaults to local SQLite in `data/` |
| `OLLAMA_HOST` | Defaults to `localhost:11434` |

At least one provider must be configured (Ollama requires no key).

## Architecture

```
                        User
                          |
                    Streamlit UI
                          |
                 Application Layer
                          |
        +-----------------+------------------+
        |                 |                  |
 Registry Services   Evaluation Engine   Dashboard
        |                 |                  |
        +--------------+--+------------------+
                       |
               Provider Router
                       |
      +--------+--------+--------+--------+
      |        |        |        |
   Ollama   OpenAI  Anthropic  Gemini
                       |
               Metrics & Scorers
                       |
             Statistics & Regression
                       |
          SQLite + MLflow + Reports
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit + Plotly |
| Database | SQLite + SQLAlchemy 2.0 + Alembic |
| Metrics | Sentence Transformers, HuggingFace Transformers (NLI) |
| Experiment Tracking | MLflow |
| Providers | Ollama, OpenAI, Anthropic, Google Gemini (google-genai) |
| Statistics | SciPy, NumPy |
| Testing | Pytest (300+ tests) |
| Deployment | Docker + Docker Compose |

## License

MIT License
