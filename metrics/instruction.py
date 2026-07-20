"""Deterministic, rule-based instruction-following scorer."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from metrics.base import Metric, MetricResult

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class InstructionScorer(Metric):
    """Scores instruction-following via deterministic, metadata-driven rules.

    Three independent rules are checked, each only when its triggering key
    is present in ``metadata``:

    - ``expected_format == "json"``: ``response`` must parse as valid
      JSON.
    - ``max_sentences: N``: ``response`` must contain at most ``N``
      sentences.
    - ``required_keywords: [list[str]]``: every keyword must appear
      (case-insensitively) somewhere in ``response``.

    Rules whose triggering key is absent from ``metadata`` are skipped
    entirely and do not count toward the denominator of the score. This is
    an MVP deterministic rule-checker; no LLM-judge fallback is used.

    An instance is stateless and holds no loaded resources, so it may be
    freely reused (and is thread-safe) across calls to :meth:`evaluate`.
    """

    METRIC_NAME: ClassVar[str] = "instruction_following"

    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Score ``response`` against whichever rules are present in ``metadata``.

        Args:
            response: The text produced by the system under evaluation.
            reference: The ground-truth answer. Not used by this scorer
                (rules are checked against ``response`` and ``metadata``
                alone) but accepted to satisfy the ``Metric`` interface.
            metadata: Evaluation context that may contain any of
                ``"expected_format"``, ``"max_sentences"``, and
                ``"required_keywords"`` to activate the corresponding
                rule. Keys not present simply skip that rule.

        Returns:
            A :class:`MetricResult` where ``score`` is the fraction of
            *applicable* rules that passed (``1.0``, with reduced
            ``confidence``, if no rules are applicable), and
            ``explanation`` lists each applicable rule's pass/fail
            outcome and reasoning.

        Raises:
            ValueError: If ``response`` is empty or whitespace-only, or if
                a rule's metadata value is malformed (e.g.
                ``max_sentences`` is not an integer).
        """
        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")

        results: list[tuple[str, bool, str]] = []

        if "expected_format" in metadata:
            results.append(self._check_expected_format(response, metadata["expected_format"]))

        if "max_sentences" in metadata:
            results.append(self._check_max_sentences(response, metadata["max_sentences"]))

        if "required_keywords" in metadata:
            results.append(self._check_required_keywords(response, metadata["required_keywords"]))

        if not results:
            return MetricResult(
                metric_name=self.METRIC_NAME,
                score=1.0,
                explanation="No applicable instruction rules were present in metadata; trivial pass.",
                confidence=0.5,
            )

        passed_count = sum(1 for _, passed, _ in results if passed)
        score = passed_count / len(results)

        rule_lines = "; ".join(
            f"{name}={'PASS' if passed else 'FAIL'} ({detail})" for name, passed, detail in results
        )
        explanation = (
            f"{passed_count}/{len(results)} applicable rule(s) passed -> score={score:.4f}. "
            f"Details: {rule_lines}."
        )

        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=score,
            explanation=explanation,
            confidence=1.0,
        )

    @staticmethod
    def _check_expected_format(response: str, expected_format: Any) -> tuple[str, bool, str]:
        """Check the ``expected_format`` rule.

        Currently only ``"json"`` is a recognized format; any other value
        is treated as an unsupported (auto-failing) format so that typos
        in metadata are surfaced rather than silently ignored.

        Args:
            response: The response text to validate.
            expected_format: The value of ``metadata["expected_format"]``.

        Returns:
            A ``(rule_name, passed, detail)`` tuple.
        """
        rule_name = "expected_format"
        if expected_format != "json":
            return rule_name, False, f"unsupported expected_format {expected_format!r}"

        try:
            json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            return rule_name, False, f"response is not valid JSON ({exc})"

        return rule_name, True, "response parsed as valid JSON"

    @staticmethod
    def _check_max_sentences(response: str, max_sentences: Any) -> tuple[str, bool, str]:
        """Check the ``max_sentences`` rule.

        Args:
            response: The response text to validate.
            max_sentences: The value of ``metadata["max_sentences"]``,
                expected to be an ``int``-like value.

        Returns:
            A ``(rule_name, passed, detail)`` tuple.

        Raises:
            ValueError: If ``max_sentences`` cannot be interpreted as an
                integer.
        """
        rule_name = "max_sentences"
        try:
            limit = int(max_sentences)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata['max_sentences'] must be an int, got {max_sentences!r}") from exc

        sentence_count = InstructionScorer._count_sentences(response)
        passed = sentence_count <= limit
        return rule_name, passed, f"{sentence_count} sentence(s) found, limit is {limit}"

    @staticmethod
    def _check_required_keywords(response: str, required_keywords: Any) -> tuple[str, bool, str]:
        """Check the ``required_keywords`` rule.

        Args:
            response: The response text to validate.
            required_keywords: The value of
                ``metadata["required_keywords"]``, expected to be an
                iterable of strings.

        Returns:
            A ``(rule_name, passed, detail)`` tuple.
        """
        rule_name = "required_keywords"
        keywords = list(required_keywords)
        response_lower = response.lower()
        missing = [kw for kw in keywords if kw.lower() not in response_lower]
        passed = not missing
        detail = "all keywords present" if passed else f"missing keywords: {missing}"
        return rule_name, passed, detail

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in ``text`` via simple terminal-punctuation splitting.

        Args:
            text: The text to split into sentences.

        Returns:
            The number of non-empty sentence fragments found (``0`` for
            blank text).
        """
        stripped = text.strip()
        if not stripped:
            return 0
        fragments = [f for f in _SENTENCE_SPLIT_RE.split(stripped) if f.strip()]
        return len(fragments)