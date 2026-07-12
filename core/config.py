"""
core/config.py
--------------
YAML-backed configuration loader using Pydantic BaseSettings.

Loads from `configs/default.yaml` and allows environment variable overrides
with the prefix `LLM_PLATFORM__` (double-underscore for nested keys).

Usage:
    from core.config import settings

    print(settings.app.name)
    print(settings.database.url)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Top-level application metadata."""

    name: str = "LLM Reliability Platform"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"


class DatabaseConfig(BaseModel):
    """SQLAlchemy / SQLite connection settings."""

    url: str = "sqlite:///./data/platform.db"
    echo: bool = False
    pool_pre_ping: bool = True


class MLflowConfig(BaseModel):
    """MLflow tracking and artifact settings."""

    tracking_uri: str = "./mlruns"
    experiment_name: str = "llm-reliability"
    artifact_location: str = "./mlartifacts"


class LoggingConfig(BaseModel):
    """Loguru logging configuration."""

    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
    rotation: str = "100 MB"
    retention: str = "30 days"
    log_dir: str = "./logs"
    colorize: bool = True


class ProvidersConfig(BaseModel):
    """Shared defaults for all LLM provider clients."""

    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


class BenchmarksConfig(BaseModel):
    """Benchmark run parameters."""

    default_runs: int = 5
    concurrency: int = 4
    warmup_runs: int = 1


class EvaluationConfig(BaseModel):
    """Evaluation metric selection and thresholds."""

    default_metrics: list[str] = Field(
        default=[
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "tokens_per_second",
            "cost_per_1k_tokens",
            "error_rate",
        ]
    )
    latency_threshold_ms: int = 5000
    error_rate_threshold: float = 0.05


class DashboardConfig(BaseModel):
    """Streamlit dashboard display settings."""

    page_title: str = "LLM Reliability Platform"
    page_icon: str = "⚡"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a plain dict.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML contents.  Returns an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    """Root settings object.

    Values are populated in this priority order (highest wins):
    1. Environment variables prefixed with ``LLM_PLATFORM__``
       (double-underscore separates nested keys, e.g.
       ``LLM_PLATFORM__DATABASE__URL``).
    2. The YAML file at ``configs/default.yaml``.
    3. Field defaults defined on the nested models above.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_PLATFORM__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    @classmethod
    def from_yaml(cls, path: Path = _CONFIG_PATH) -> "Settings":
        """Construct a Settings instance by merging YAML data with env vars.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully-populated Settings instance.
        """
        yaml_data = _load_yaml(path)
        return cls(**yaml_data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call).

    The cache is cleared automatically when the process restarts. During
    testing, call ``get_settings.cache_clear()`` before patching env vars.

    Returns:
        The application-wide Settings object.
    """
    return Settings.from_yaml()


# Convenience alias used throughout the codebase.
settings: Settings = get_settings()