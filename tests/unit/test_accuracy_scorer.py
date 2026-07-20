"""Unit tests for :class:`metrics.accuracy.AccuracyScorer`.

The real sentence-transformers model is never loaded in these tests --
either a lightweight mock embedding model is injected via
``AccuracyScorer(model=...)``, or ``SentenceTransformer`` itself is
patched -- so the suite stays fast and network-independent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from metrics.accuracy import AccuracyScorer
from metrics.base import MetricResult


@pytest.fixture(autouse=True)
def _no_real_model():
    """Patch SentenceTransformer everywhere in this file, autouse.

    __init__ never touches real weights, even for tests that don't pass
    ``model=`` explicitly. Nested per-test ``patch`` calls (constructor
    assertions) still stack fine on top of this.
    """
    with patch("metrics.accuracy.SentenceTransformer") as mock_ctor:
        mock_ctor.return_value = MagicMock()
        yield mock_ctor


def _make_fake_model(response_vector: list[float], reference_vector: list[float]) -> MagicMock:
    """Build a mock SentenceTransformer whose ``encode`` returns fixed vectors.

    ``AccuracyScorer.evaluate`` calls
    ``model.encode([response, reference], convert_to_tensor=True)``, so the
    mock's ``encode`` returns a 2-row tensor: row 0 is the "response"
    embedding, row 1 is the "reference" embedding. Vectors below are
    pre-normalized (unit length) so the resulting dot product equals the
    intended cosine similarity directly.
    """
    fake_model = MagicMock()
    fake_model.encode.return_value = torch.tensor(
        [response_vector, reference_vector], dtype=torch.float32
    )
    return fake_model


class TestAccuracyScorerExactMatch:
    """Identical response/reference text should score near-perfect similarity."""

    def test_exact_match_scores_high(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [1.0, 0.0])
        scorer = AccuracyScorer(model=fake_model)

        result = scorer.evaluate(response="Paris", reference="Paris", metadata={})

        assert isinstance(result, MetricResult)
        assert result.metric_name == "semantic_accuracy"
        assert result.score == pytest.approx(1.0, abs=1e-6)
        assert result.score >= scorer.threshold
        assert "PASS" in result.explanation
        assert result.confidence == 1.0


class TestAccuracyScorerParaphrase:
    """Paraphrased answers with high semantic overlap should score high and pass."""

    def test_paraphrase_scores_high_and_passes(self) -> None:
        # Unit vectors with cos(theta) = 0.92, above the 0.85 default threshold.
        fake_model = _make_fake_model([1.0, 0.0], [0.92, 0.3919183588453085])
        scorer = AccuracyScorer(model=fake_model)

        result = scorer.evaluate(
            response="Paris",
            reference="The capital is Paris",
            metadata={},
        )

        assert result.score == pytest.approx(0.92, abs=1e-3)
        assert result.score >= scorer.threshold
        assert "PASS" in result.explanation

    def test_encode_called_once_with_response_and_reference(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [0.92, 0.3919183588453085])
        scorer = AccuracyScorer(model=fake_model)

        scorer.evaluate(response="Paris", reference="The capital is Paris", metadata={})

        fake_model.encode.assert_called_once_with(
            ["Paris", "The capital is Paris"], convert_to_tensor=True
        )


class TestAccuracyScorerWrongAnswer:
    """Semantically unrelated response/reference pairs should score low and fail."""

    def test_wrong_answer_scores_low_and_fails(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [0.0, 1.0])  # orthogonal -> similarity 0.0
        scorer = AccuracyScorer(model=fake_model)

        result = scorer.evaluate(response="Paris", reference="Tokyo", metadata={})

        assert result.score == pytest.approx(0.0, abs=1e-6)
        assert result.score < scorer.threshold
        assert "FAIL" in result.explanation


class TestAccuracyScorerConfiguration:
    """Behavioral tests around threshold configuration and input validation."""

    def test_custom_threshold_changes_pass_fail_verdict(self) -> None:
        # cos(theta) = 0.7, which fails the 0.85 default but passes a 0.5 threshold.
        fake_model = _make_fake_model([1.0, 0.0], [0.7, 0.7141428428542851])
        scorer = AccuracyScorer(model=fake_model, threshold=0.5)

        result = scorer.evaluate(response="a", reference="b", metadata={})

        assert result.score == pytest.approx(0.7, abs=1e-3)
        assert result.score >= scorer.threshold
        assert "PASS" in result.explanation

    def test_same_similarity_fails_default_threshold(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [0.7, 0.7141428428542851])
        scorer = AccuracyScorer(model=fake_model)  # default threshold 0.85

        result = scorer.evaluate(response="a", reference="b", metadata={})

        assert result.score < scorer.threshold
        assert "FAIL" in result.explanation

    def test_invalid_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AccuracyScorer(threshold=1.5)

    def test_negative_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AccuracyScorer(threshold=-0.1)

    def test_empty_response_raises_value_error(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [1.0, 0.0])
        scorer = AccuracyScorer(model=fake_model)

        with pytest.raises(ValueError):
            scorer.evaluate(response="   ", reference="Paris", metadata={})

    def test_empty_reference_raises_value_error(self) -> None:
        fake_model = _make_fake_model([1.0, 0.0], [1.0, 0.0])
        scorer = AccuracyScorer(model=fake_model)

        with pytest.raises(ValueError):
            scorer.evaluate(response="Paris", reference="", metadata={})


class TestAccuracyScorerModelLoading:
    """The embedding model must be loaded once, not on every evaluate() call."""

    def test_model_loaded_once_from_model_name(self) -> None:
        with patch("metrics.accuracy.SentenceTransformer") as mock_ctor:
            mock_ctor.return_value = _make_fake_model([1.0, 0.0], [1.0, 0.0])

            scorer = AccuracyScorer(model_name="all-MiniLM-L6-v2")
            scorer.evaluate(response="Paris", reference="Paris", metadata={})
            scorer.evaluate(response="Paris", reference="Paris", metadata={})

            # Constructor invoked exactly once, regardless of evaluate() call count.
            mock_ctor.assert_called_once_with("all-MiniLM-L6-v2")

    def test_injected_model_bypasses_constructor(self) -> None:
        with patch("metrics.accuracy.SentenceTransformer") as mock_ctor:
            fake_model = _make_fake_model([1.0, 0.0], [1.0, 0.0])
            AccuracyScorer(model=fake_model)

            mock_ctor.assert_not_called()