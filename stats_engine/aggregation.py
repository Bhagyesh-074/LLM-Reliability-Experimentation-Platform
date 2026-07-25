"""Aggregation utilities for summarizing raw evaluation scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class AggregationService:
    """Computes summary statistics over raw evaluation scores."""

    def compute_mean_std(self, scores: List[float]) -> Tuple[float, float]:
        """Compute the mean and standard deviation of a list of scores.

        The standard deviation is the population standard deviation
        (``ddof=0``), which is appropriate for describing the spread of a
        given batch of scores rather than estimating a population
        parameter from a sample.

        Args:
            scores: Raw numeric scores.

        Returns:
            A ``(mean, std)`` tuple. Returns ``(0.0, 0.0)`` if ``scores``
            is empty, and ``(score, 0.0)`` for a single-element list.
        """
        if not scores:
            return 0.0, 0.0
        if len(scores) == 1:
            return float(scores[0]), 0.0

        arr = np.asarray(scores, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    def aggregate_by_model(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Group evaluation results by model and compute per-model statistics.

        Args:
            results: A list of result dicts, each expected to contain a
                ``model_name`` key and a numeric ``score`` key. Rows
                missing either key are skipped.

        Returns:
            A mapping from ``model_name`` to a dict with keys ``mean``,
            ``std``, ``count``, and ``scores`` (the raw scores for that
            model, in encounter order). Returns an empty dict if
            ``results`` is empty or contains no usable rows.
        """
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in results:
            model_name = row.get("model_name")
            score = row.get("score")
            if model_name is None or score is None:
                continue
            grouped[model_name].append(float(score))

        aggregated: Dict[str, Dict[str, Any]] = {}
        for model_name, scores in grouped.items():
            mean_val, std_val = self.compute_mean_std(scores)
            aggregated[model_name] = {
                "mean": mean_val,
                "std": std_val,
                "count": len(scores),
                "scores": scores,
            }
        return aggregated