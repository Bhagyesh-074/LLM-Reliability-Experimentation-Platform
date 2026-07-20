"""Core interfaces for the LLM Reliability & Experimentation Platform's
metric engine.

This module defines the abstract :class:`Metric` contract that every
scorer (Accuracy, Hallucination, Instruction Following, Safety, Latency,
Cost, Consistency, ...) must implement, along with the
:class:`MetricResult` Pydantic model used to report a single metric's
outcome in a uniform, machine-readable shape. A composite/aggregate score
is expected to be computed downstream from a collection of
``MetricResult`` objects, once all individual metrics have finished.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Uniform result payload returned by every :class:`Metric` implementation.

    Attributes:
        metric_name: Stable identifier for the metric that produced this
            result (e.g. ``"semantic_accuracy"``). Used for aggregation,
            logging, and building the composite score.
        score: Normalized metric score in the inclusive range ``[0.0,
            1.0]``, where ``1.0`` represents the best possible outcome and
            ``0.0`` the worst.
        explanation: Human-readable explanation of how the score was
            derived, suitable for display in reports and dashboards.
        confidence: Confidence in the reported score, in the inclusive
            range ``[0.0, 1.0]``. Deterministic metrics (e.g. exact string
            match) may report ``1.0``; metrics based on noisier signals
            should report a lower value when appropriate.
    """

    metric_name: str = Field(..., min_length=1, description="Stable identifier for the metric.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score in [0.0, 1.0].")
    explanation: str = Field(..., min_length=1, description="Human-readable rationale for the score.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the reported score, in [0.0, 1.0]."
    )


class Metric(ABC):
    """Abstract base class for all metric scorers in the platform.

    Concrete subclasses (e.g. ``AccuracyScorer``, ``HallucinationScorer``,
    ``InstructionFollowingScorer``, ``SafetyScorer``, ``LatencyScorer``,
    ``CostScorer``, ``ConsistencyScorer``) implement :meth:`evaluate` to
    compare a model's ``response`` against a ``reference`` (ground truth,
    rubric, or other target) and return a :class:`MetricResult`.

    Implementations should be side-effect free with respect to their
    inputs and must avoid re-initializing expensive resources (e.g. ML
    models, API clients) on every call to :meth:`evaluate`. Such resources
    should be loaded once, typically in ``__init__``.
    """

    @abstractmethod
    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Evaluate a model ``response`` against a ``reference``.

        Args:
            response: The text produced by the system under evaluation.
            reference: The ground-truth answer, rubric, or other target
                that ``response`` is being compared against.
            metadata: Additional context for the evaluation (e.g. prompt,
                model name, latency, token counts, run configuration).
                Implementations that do not need extra context may ignore
                this argument, but must still accept it.

        Returns:
            A :class:`MetricResult` describing the outcome of the
            evaluation.
        """
        raise NotImplementedError