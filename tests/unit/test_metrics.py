"""Unit tests for all metric scorers.

All model/pipeline dependencies (SentenceTransformer, transformers NLI
pipeline) are mocked or injected via each scorer's dependency-injection
constructor argument, so these tests never download or load a real model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from metrics import (
    AccuracyScorer,
    CompositeScorer,
    CostScorer,
    HallucinationScorer,
    InstructionScorer,
    LatencyScorer,
    MetricResult,
    SafetyScorer,
)

# --------------------------------------------------------------------------- #
# Shared test doubles / helpers
# --------------------------------------------------------------------------- #


class _StubEmbeddingModel:
    """Stand-in for ``SentenceTransformer`` that returns fixed embeddings."""

    def __init__(self, embeddings: torch.Tensor) -> None:
        self._embeddings = embeddings

    def encode(self, texts: list[str], convert_to_tensor: bool = True) -> torch.Tensor:
        return self._embeddings


def _make_nli_pipeline(predictions: object) -> MagicMock:
    """Build a mock NLI pipeline whose call returns ``predictions`` verbatim."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = predictions
    return mock_pipeline


def _mk_result(name: str, score: float, confidence: float = 1.0) -> MetricResult:
    return MetricResult(metric_name=name, score=score, explanation="stub", confidence=confidence)


# --------------------------------------------------------------------------- #
# AccuracyScorer
# --------------------------------------------------------------------------- #


def test_accuracy_scorer_high_similarity_passes() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    scorer = AccuracyScorer(threshold=0.85, model=_StubEmbeddingModel(embeddings))
    result = scorer.evaluate("response text", "reference text", {})
    assert result.score == pytest.approx(1.0)
    assert result.metric_name == "semantic_accuracy"
    assert "PASS" in result.explanation


def test_accuracy_scorer_low_similarity_fails() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scorer = AccuracyScorer(threshold=0.85, model=_StubEmbeddingModel(embeddings))
    result = scorer.evaluate("response text", "reference text", {})
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert "FAIL" in result.explanation


def test_accuracy_scorer_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        AccuracyScorer(threshold=1.5, model=_StubEmbeddingModel(torch.tensor([[1.0, 0.0]])))


def test_accuracy_scorer_rejects_empty_response() -> None:
    scorer = AccuracyScorer(model=_StubEmbeddingModel(torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
    with pytest.raises(ValueError):
        scorer.evaluate("", "reference", {})


def test_accuracy_scorer_rejects_empty_reference() -> None:
    scorer = AccuracyScorer(model=_StubEmbeddingModel(torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
    with pytest.raises(ValueError):
        scorer.evaluate("response", "   ", {})


# --------------------------------------------------------------------------- #
# HallucinationScorer
# --------------------------------------------------------------------------- #


def test_hallucination_scorer_entailment_scores_one() -> None:
    predictions = [
        {"label": "ENTAILMENT", "score": 0.92},
        {"label": "NEUTRAL", "score": 0.05},
        {"label": "CONTRADICTION", "score": 0.03},
    ]
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline(predictions))
    result = scorer.evaluate("Paris is the capital of France.", "France's capital is Paris.", {})
    assert result.score == pytest.approx(1.0)
    assert result.confidence == pytest.approx(0.92)
    assert "entailment" in result.explanation.lower()


def test_hallucination_scorer_contradiction_scores_zero() -> None:
    predictions = [
        {"label": "entailment", "score": 0.02},
        {"label": "neutral", "score": 0.08},
        {"label": "contradiction", "score": 0.90},
    ]
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline(predictions))
    result = scorer.evaluate("The Eiffel Tower is in Berlin.", "The Eiffel Tower is in Paris.", {})
    assert result.score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.90)


def test_hallucination_scorer_neutral_scores_half() -> None:
    predictions = [
        {"label": "entailment", "score": 0.3},
        {"label": "neutral", "score": 0.6},
        {"label": "contradiction", "score": 0.1},
    ]
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline(predictions))
    result = scorer.evaluate("Some unrelated claim.", "Reference statement.", {})
    assert result.score == pytest.approx(0.5)


def test_hallucination_scorer_handles_nested_list_output() -> None:
    predictions = [
        [
            {"label": "entailment", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "contradiction", "score": 0.1},
        ]
    ]
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline(predictions))
    result = scorer.evaluate("resp", "ref", {})
    assert result.score == pytest.approx(1.0)


def test_hallucination_scorer_rejects_empty_response() -> None:
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline([{"label": "entailment", "score": 1.0}]))
    with pytest.raises(ValueError):
        scorer.evaluate("", "reference", {})


def test_hallucination_scorer_rejects_empty_reference() -> None:
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline([{"label": "entailment", "score": 1.0}]))
    with pytest.raises(ValueError):
        scorer.evaluate("response", "", {})


def test_hallucination_scorer_rejects_empty_pipeline_output() -> None:
    scorer = HallucinationScorer(nli_pipeline=_make_nli_pipeline([]))
    with pytest.raises(ValueError):
        scorer.evaluate("resp", "ref", {})


# --------------------------------------------------------------------------- #
# InstructionScorer
# --------------------------------------------------------------------------- #


def test_instruction_scorer_no_rules_is_trivial_pass() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("Hello world.", "reference", {})
    assert result.score == pytest.approx(1.0)
    assert result.confidence == pytest.approx(0.5)


def test_instruction_scorer_json_rule_passes() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate('{"a": 1}', "reference", {"expected_format": "json"})
    assert result.score == pytest.approx(1.0)


def test_instruction_scorer_json_rule_fails() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("not json", "reference", {"expected_format": "json"})
    assert result.score == pytest.approx(0.0)
    assert "FAIL" in result.explanation


def test_instruction_scorer_max_sentences_passes() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("One. Two.", "reference", {"max_sentences": 2})
    assert result.score == pytest.approx(1.0)


def test_instruction_scorer_max_sentences_fails() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("One. Two. Three.", "reference", {"max_sentences": 2})
    assert result.score == pytest.approx(0.0)


def test_instruction_scorer_required_keywords_partial_missing() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("The cat sat on the mat.", "reference", {"required_keywords": ["cat", "dog"]})
    assert result.score == pytest.approx(0.0)
    assert "missing keywords" in result.explanation


def test_instruction_scorer_required_keywords_empty_list_passes() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate("Anything.", "reference", {"required_keywords": []})
    assert result.score == pytest.approx(1.0)


def test_instruction_scorer_multiple_rules_partial_pass() -> None:
    scorer = InstructionScorer()
    result = scorer.evaluate(
        "The cat sat.",
        "reference",
        {"max_sentences": 1, "required_keywords": ["cat", "dog"]},
    )
    assert result.score == pytest.approx(0.5)


def test_instruction_scorer_rejects_empty_response() -> None:
    scorer = InstructionScorer()
    with pytest.raises(ValueError):
        scorer.evaluate("", "reference", {})


def test_instruction_scorer_rejects_bad_max_sentences_type() -> None:
    scorer = InstructionScorer()
    with pytest.raises(ValueError):
        scorer.evaluate("Hello.", "reference", {"max_sentences": "not-a-number"})


def test_instruction_scorer_handles_very_long_response() -> None:
    scorer = InstructionScorer()
    long_response = "This is a sentence. " * 5000
    result = scorer.evaluate(long_response, "reference", {"max_sentences": 100})
    assert result.score == pytest.approx(0.0)  # 5000 sentences > 100


# --------------------------------------------------------------------------- #
# SafetyScorer
# --------------------------------------------------------------------------- #

_TEST_BLOCKLIST = {
    "violence": ["kill you"],
    "self_harm": ["suicide"],
    "hate_speech": [],
    "sexual_content": [],
}


def test_safety_scorer_safe_response() -> None:
    scorer = SafetyScorer(blocklist=_TEST_BLOCKLIST)
    result = scorer.evaluate("Have a wonderful day!", "reference", {})
    assert result.score == pytest.approx(1.0)


def test_safety_scorer_flags_violence() -> None:
    scorer = SafetyScorer(blocklist=_TEST_BLOCKLIST)
    result = scorer.evaluate("I will kill you.", "reference", {})
    assert result.score == pytest.approx(0.0)
    assert "violence" in result.explanation


def test_safety_scorer_word_boundary_avoids_false_positive() -> None:
    scorer = SafetyScorer(blocklist={"violence": ["kill"]})
    result = scorer.evaluate("This dish is a killer recipe.", "reference", {})
    assert result.score == pytest.approx(1.0)


def test_safety_scorer_rejects_bad_blocklist_shape() -> None:
    with pytest.raises(ValueError):
        SafetyScorer(blocklist={"violence": "not-a-list"})


def test_safety_scorer_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SafetyScorer(blocklist_path="configs/does_not_exist.yaml")


def test_safety_scorer_rejects_empty_response() -> None:
    scorer = SafetyScorer(blocklist=_TEST_BLOCKLIST)
    with pytest.raises(ValueError):
        scorer.evaluate("", "reference", {})


def test_safety_scorer_handles_very_long_response() -> None:
    scorer = SafetyScorer(blocklist=_TEST_BLOCKLIST)
    long_response = ("This is a perfectly safe sentence. " * 5000).strip()
    result = scorer.evaluate(long_response, "reference", {})
    assert result.score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# LatencyScorer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("latency_ms", "expected_score", "expected_tier"),
    [
        (500, 1.0, "excellent"),
        (2500, 0.75, "good"),
        (4500, 0.5, "acceptable"),
        (6000, 0.25, "poor"),
    ],
)
def test_latency_scorer_tiers(latency_ms: float, expected_score: float, expected_tier: str) -> None:
    scorer = LatencyScorer()
    result = scorer.evaluate("resp", "ref", {"latency_ms": latency_ms})
    assert result.score == pytest.approx(expected_score)
    assert expected_tier in result.explanation


def test_latency_scorer_missing_metadata_raises() -> None:
    scorer = LatencyScorer()
    with pytest.raises(KeyError):
        scorer.evaluate("resp", "ref", {})


def test_latency_scorer_negative_latency_raises() -> None:
    scorer = LatencyScorer()
    with pytest.raises(ValueError):
        scorer.evaluate("resp", "ref", {"latency_ms": -1})


def test_latency_scorer_non_numeric_latency_raises() -> None:
    scorer = LatencyScorer()
    with pytest.raises(ValueError):
        scorer.evaluate("resp", "ref", {"latency_ms": "fast"})


def test_latency_scorer_custom_thresholds() -> None:
    scorer = LatencyScorer(excellent_ms=100, good_ms=200, acceptable_ms=300)
    result = scorer.evaluate("resp", "ref", {"latency_ms": 150})
    assert result.score == pytest.approx(0.75)


def test_latency_scorer_rejects_invalid_threshold_ordering() -> None:
    with pytest.raises(ValueError):
        LatencyScorer(excellent_ms=1000, good_ms=500, acceptable_ms=2000)


# --------------------------------------------------------------------------- #
# CostScorer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("total_tokens", "expected_score", "expected_tier"),
    [
        (200, 1.0, "excellent"),
        (750, 0.75, "good"),
        (1500, 0.5, "acceptable"),
        (3000, 0.25, "poor"),
    ],
)
def test_cost_scorer_tiers(total_tokens: int, expected_score: float, expected_tier: str) -> None:
    scorer = CostScorer()
    result = scorer.evaluate("resp", "ref", {"token_usage": {"total_tokens": total_tokens}})
    assert result.score == pytest.approx(expected_score)
    assert expected_tier in result.explanation


def test_cost_scorer_missing_token_usage_raises() -> None:
    scorer = CostScorer()
    with pytest.raises(KeyError):
        scorer.evaluate("resp", "ref", {})


def test_cost_scorer_missing_total_tokens_key_raises() -> None:
    scorer = CostScorer()
    with pytest.raises(KeyError):
        scorer.evaluate("resp", "ref", {"token_usage": {"prompt_tokens": 10}})


def test_cost_scorer_negative_tokens_raises() -> None:
    scorer = CostScorer()
    with pytest.raises(ValueError):
        scorer.evaluate("resp", "ref", {"token_usage": {"total_tokens": -5}})


def test_cost_scorer_custom_thresholds() -> None:
    scorer = CostScorer(excellent_tokens=10, good_tokens=20, acceptable_tokens=30)
    result = scorer.evaluate("resp", "ref", {"token_usage": {"total_tokens": 15}})
    assert result.score == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# CompositeScorer
# --------------------------------------------------------------------------- #


def test_composite_scorer_default_weights_sum_to_one() -> None:
    scorer = CompositeScorer()
    assert sum(scorer.weights.values()) == pytest.approx(1.0)


def test_composite_scorer_all_perfect_scores_gives_perfect_composite() -> None:
    scorer = CompositeScorer()
    results = {
        "accuracy": _mk_result("semantic_accuracy", 1.0),
        "hallucination": _mk_result("hallucination", 1.0),
        "instruction": _mk_result("instruction_following", 1.0),
        "safety": _mk_result("safety", 1.0),
        "latency": _mk_result("latency", 1.0),
        "cost": _mk_result("cost", 1.0),
    }
    composite = scorer.compute(results)
    assert composite.score == pytest.approx(1.0)
    assert composite.metric_name == "composite_score"


def test_composite_scorer_weighted_average_matches_manual_calculation() -> None:
    scorer = CompositeScorer()
    results = {
        "accuracy": _mk_result("semantic_accuracy", 0.9),
        "hallucination": _mk_result("hallucination", 0.8),
        "instruction": _mk_result("instruction_following", 1.0),
        "safety": _mk_result("safety", 1.0),
        "latency": _mk_result("latency", 0.5),
        "cost": _mk_result("cost", 0.75),
    }
    composite = scorer.compute(results)
    expected = 0.35 * 0.9 + 0.25 * 0.8 + 0.20 * 1.0 + 0.10 * 1.0 + 0.05 * 0.5 + 0.05 * 0.75
    assert composite.score == pytest.approx(expected)


def test_composite_scorer_missing_component_raises() -> None:
    scorer = CompositeScorer()
    with pytest.raises(KeyError):
        scorer.compute({"accuracy": _mk_result("semantic_accuracy", 1.0)})


def test_composite_scorer_ignores_extra_result_keys() -> None:
    scorer = CompositeScorer(weights={"accuracy": 1.0})
    results = {
        "accuracy": _mk_result("semantic_accuracy", 0.6),
        "unused_extra": _mk_result("something_else", 0.0),
    }
    composite = scorer.compute(results)
    assert composite.score == pytest.approx(0.6)


def test_composite_scorer_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        CompositeScorer(weights={"accuracy": 0.5, "safety": 0.4})


def test_composite_scorer_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        CompositeScorer(weights={"accuracy": -0.1, "safety": 1.1})


def test_composite_scorer_rejects_empty_weights() -> None:
    with pytest.raises(ValueError):
        CompositeScorer(weights={})