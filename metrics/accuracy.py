"""Semantic accuracy scorer based on sentence-embedding cosine similarity."""

from __future__ import annotations

from typing import Any, ClassVar

from sentence_transformers import SentenceTransformer, util

from metrics.base import Metric, MetricResult

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.85


class AccuracyScorer(Metric):
    """Scores semantic accuracy of a response against a reference answer.

    Uses a sentence-transformers bi-encoder to embed both the ``response``
    and the ``reference`` into a shared vector space, then scores them by
    cosine similarity. This captures paraphrases and reworded answers that
    exact-match or lexical-overlap metrics would incorrectly penalize.

    The embedding model is loaded once per ``AccuracyScorer`` instance (in
    ``__init__``) and reused across all subsequent calls to
    :meth:`evaluate`, since model loading is the dominant cost and must
    not be repeated per evaluation.

    Attributes:
        threshold: Minimum cosine similarity, in ``[0.0, 1.0]``, for a
            response to be considered a semantic match ("pass") with the
            reference. Defaults to ``0.85``.
        model_name: Name of the sentence-transformers model used to embed
            text. Defaults to ``"all-MiniLM-L6-v2"``.
    """

    METRIC_NAME: ClassVar[str] = "semantic_accuracy"

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        model_name: str = _DEFAULT_MODEL_NAME,
        model: SentenceTransformer | None = None,
    ) -> None:
        """Initialize the scorer and load the embedding model.

        Args:
            threshold: Minimum cosine similarity required for a "pass".
                Must be within ``[0.0, 1.0]``.
            model_name: sentence-transformers model identifier to load
                when ``model`` is not supplied.
            model: An already-instantiated ``SentenceTransformer``.
                Primarily intended for dependency injection (e.g. in
                tests), so the real model does not need to be
                downloaded/loaded. When omitted, a model is loaded from
                ``model_name``.

        Raises:
            ValueError: If ``threshold`` is not within ``[0.0, 1.0]``.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold}")

        self.threshold = threshold
        self.model_name = model_name
        self._model: SentenceTransformer = model if model is not None else SentenceTransformer(model_name)

    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Score ``response`` against ``reference`` using cosine similarity.

        Both texts are embedded in a single batched call to the underlying
        model, then compared with cosine similarity.

        Args:
            response: The text produced by the system under evaluation.
            reference: The ground-truth answer to compare against.
            metadata: Additional evaluation context. Not used by this
                scorer but accepted to satisfy the ``Metric`` interface.

        Returns:
            A :class:`MetricResult` where ``score`` is the cosine
            similarity between the embedded ``response`` and ``reference``
            (clamped to ``[0.0, 1.0]``), and ``explanation`` reports the
            similarity value alongside a pass/fail verdict relative to
            ``self.threshold``.

        Raises:
            ValueError: If ``response`` or ``reference`` is empty or
                whitespace-only.
        """
        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")
        if not reference or not reference.strip():
            raise ValueError("reference must be a non-empty string")

        embeddings = self._model.encode([response, reference], convert_to_tensor=True)
        similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())

        # Cosine similarity is mathematically in [-1.0, 1.0]; clamp into the
        # [0.0, 1.0] score range expected by MetricResult.
        score = max(0.0, min(1.0, similarity))
        passed = score >= self.threshold
        verdict = "PASS" if passed else "FAIL"

        explanation = (
            f"Cosine similarity between response and reference embeddings "
            f"({self.model_name}) is {score:.4f}; threshold is "
            f"{self.threshold:.4f} -> {verdict}."
        )

        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=score,
            explanation=explanation,
            confidence=1.0,
        )