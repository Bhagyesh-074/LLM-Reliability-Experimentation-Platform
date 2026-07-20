"""core/evaluation/orchestrator.py

Evaluation orchestrator for a single evaluation run.

``EvaluationOrchestrator`` wires together the Prompt Registry, the
Benchmark Registry, the Provider layer, the metric engine (six scorers +
composite), and the persistence layer to execute one evaluation run end
to end: resolve the prompt version and dataset rows named by an
``EvaluationConfig``, build a request per row, call the configured
provider, score each response with every metric, aggregate a composite
score, and persist the run, its per-row results, and its aggregate
metrics.

Pipeline (SYSTEM_DESIGN.md "Evaluation Pipeline"):
    Dataset -> Prompt Resolution -> Request Builder -> Provider ->
    Response -> Metric Engine -> Statistics -> Persistence -> Dashboard.

Robustness contract:
    * A single row's failure (missing template variable, provider error,
      scorer error) is logged, recorded as a failed row, and never aborts
      the run.
    * If more than 50% of rows fail, the run's status is ``"failed"``, but
      partial results are still persisted.
    * Progress updates are emitted through an optional callback so a UI can
      render a progress bar.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

from core.evaluation.config import EvaluationConfig
from core.evaluation.persistence import EvaluationPersistenceService
from core.evaluation.request_builder import MissingTemplateVariableError, RequestBuilder
from metrics.base import Metric, MetricResult
from metrics.composite import CompositeScorer
from providers.base import BaseLLMProvider, LLMRequest, LLMResponse, ProviderError
from providers.factory import ProviderFactory
from registry.benchmark_service import BenchmarkService
from registry.prompt_service import PromptService
from registry.schemas import PromptVersionResponse

logger = logging.getLogger(__name__)

RunStatus = Literal["completed", "completed_with_errors", "failed"]

#: Canonical component keys shared by ``CompositeScorer.weights``,
#: ``RunMetrics`` columns, and the per-row ``metric_scores`` mapping. The
#: individual scorers report their own ``metric_name`` values (e.g.
#: "semantic_accuracy"); the orchestrator maps every scorer onto one of
#: these stable keys so the composite scorer and DB layer agree.
COMPONENT_KEYS: tuple[str, ...] = (
    "accuracy",
    "hallucination",
    "instruction",
    "safety",
    "latency",
    "cost",
)

#: Fraction of failed rows strictly above which the whole run is marked
#: ``"failed"`` (per the ">50% rows fail" rule).
FAILURE_THRESHOLD = 0.5

ProgressCallback = Callable[[Dict[str, Any]], None]


class EvaluationRowResult(BaseModel):
    """The outcome of evaluating a single dataset row.

    ``request`` is ``None`` only when the row failed before a request
    could even be built (e.g. a missing template variable). On success,
    ``response`` and ``metric_scores`` are populated and ``composite_score``
    is set; on failure, ``error`` is populated instead.
    """

    row_index: int = Field(..., description="Zero-based position of this row within the dataset.")
    question_row: Dict[str, Any] = Field(..., description="The raw dataset row, as loaded.")
    request: Optional[LLMRequest] = Field(
        default=None, description="The request sent to the provider, if one was built."
    )
    response: Optional[LLMResponse] = Field(
        default=None, description="The provider's raw response, if generation succeeded."
    )
    metric_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component score in [0, 1], keyed by COMPONENT_KEYS. Empty on failure.",
    )
    composite_score: Optional[float] = Field(
        default=None, description="Weighted composite of metric_scores, if scoring succeeded."
    )
    success: bool = Field(..., description="Whether this row completed (generation + scoring) without error.")
    error: Optional[str] = Field(default=None, description="Human-readable error, if this row failed.")


class EvaluationRun(BaseModel):
    """In-memory result of executing one evaluation run.

    This is a lightweight DTO, distinct from the ``database.models.EvaluationRun``
    ORM row it is persisted into. ``run_id`` is populated with the
    persisted primary key after ``EvaluationPersistenceService.save_run``
    runs.
    """

    run_id: str = Field(..., description="Persisted primary key of this run (from the DB).")
    config: EvaluationConfig
    status: RunStatus = Field(..., description="'completed', 'completed_with_errors', or 'failed'.")
    composite_score: Optional[float] = Field(
        default=None, description="Mean composite score across successful rows, or None if none succeeded."
    )
    metric_averages: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean of each component score across successful rows only, keyed by COMPONENT_KEYS.",
    )
    started_at: datetime
    completed_at: datetime
    results: List[EvaluationRowResult] = Field(default_factory=list)

    @property
    def raw_responses(self) -> List[LLMResponse]:
        """Convenience view: the successful raw ``LLMResponse`` objects, in row order."""
        return [row.response for row in self.results if row.response is not None]


class EvaluationOrchestrator:
    """Coordinates a single evaluation run end to end.

    Loads the prompt version and dataset rows, resolves a provider,
    dispatches a request per row, scores each response with all six
    metrics plus the composite, aggregates run-level averages, and
    persists the run, its results, and its aggregate metrics.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        session: Session,
        prompt_service: Optional[PromptService] = None,
        benchmark_service: Optional[BenchmarkService] = None,
        provider: Optional[BaseLLMProvider] = None,
        scorers: Optional[Dict[str, Metric]] = None,
        composite_scorer: Optional[CompositeScorer] = None,
        persistence: Optional[EvaluationPersistenceService] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the orchestrator for one evaluation run.

        Args:
            config: The validated run configuration.
            session: An active SQLAlchemy session, used to construct the
                default registry services and persistence service.
            prompt_service: Optional pre-built ``PromptService``. Defaults
                to a new instance bound to ``session``.
            benchmark_service: Optional pre-built ``BenchmarkService``.
                Defaults to a new instance bound to ``session``.
            provider: Optional pre-built provider instance, bypassing
                ``ProviderFactory``. Useful for tests. Defaults to
                ``None``, meaning the provider is constructed in ``run()``.
            scorers: Optional mapping of ``COMPONENT_KEYS`` -> ``Metric``
                instance. Injecting this lets tests avoid loading heavy ML
                models, and lets callers (e.g. the Streamlit dashboard)
                supply a cached scorer set so models load once per process
                rather than once per orchestrator. Defaults to ``None``,
                meaning the six default scorers are constructed here, in
                ``__init__``, and stored on ``self._scorers``.
            composite_scorer: Optional pre-built ``CompositeScorer``.
                Defaults to a new instance with default weights.
            persistence: Optional pre-built persistence service. Defaults
                to a new instance bound to ``session``.
            progress_callback: Optional callable invoked once before the
                first row, once after each row, and once when the run
                finishes, with a ``{"current", "total", "status"}`` dict.
                Exceptions raised by the callback are swallowed so a faulty
                UI hook cannot abort a run.
        """
        self.config = config
        self.session = session
        self.prompt_service = prompt_service or PromptService(session)
        self.benchmark_service = benchmark_service or BenchmarkService(session)
        self._injected_provider = provider
        self._injected_scorers = scorers
        self.composite_scorer = composite_scorer or CompositeScorer()
        self.persistence = persistence or EvaluationPersistenceService(session)
        self.progress_callback = progress_callback
        self.request_builder = RequestBuilder(config)

        # Resolved once, here, rather than lazily inside run(): the two
        # ML-backed scorers (Accuracy, Hallucination) load real models, and
        # doing that at construction time lets a caller build one
        # orchestrator per cached scorer set (e.g. Streamlit's
        # @st.cache_resource) instead of reloading models on every run.
        self._scorers: Dict[str, Metric] = self._get_scorers()

    async def run(self) -> EvaluationRun:
        """Execute the evaluation run end to end.

        Returns:
            An ``EvaluationRun`` DTO carrying the persisted ``run_id``,
            per-row results, mean composite score, per-component averages,
            timestamps, and an overall ``status`` of ``"completed"`` (no
            failures), ``"completed_with_errors"`` (some rows failed but
            <= 50%), or ``"failed"`` (> 50% of rows failed).

        Raises:
            PromptNotFoundError: If ``config.prompt_version_id`` does not
                resolve. This is a setup error, not a per-row error, so it
                propagates.
            FileNotFoundError: If ``config.dataset_version_id`` has no
                stored dataset file.
            ProviderError: If ``config.provider_name`` does not match a
                known provider (raised before any row is processed).
        """
        started_at = datetime.now(timezone.utc)
        logger.info(
            "Starting evaluation run: provider=%s model=%s prompt_version_id=%s dataset_version_id=%s",
            self.config.provider_name,
            self.config.model_name,
            self.config.prompt_version_id,
            self.config.dataset_version_id,
        )

        # Setup-phase resolution: any failure here is fatal and propagates.
        # Scorers are not resolved here -- they were already built once in
        # __init__ and live on self._scorers.
        prompt_version = self._load_prompt_version()
        rows = self._load_dataset_rows()
        provider = self._get_provider()

        total = len(rows)
        self._emit_progress(current=0, total=total, status="started")

        results: List[EvaluationRowResult] = []
        for index, row in enumerate(rows):
            result = await self._evaluate_row(index, row, prompt_version, provider)
            results.append(result)
            self._emit_progress(
                current=index + 1,
                total=total,
                status="ok" if result.success else "row_failed",
            )

        completed_at = datetime.now(timezone.utc)
        status = self._determine_status(results)
        metric_averages = self._aggregate_metric_averages(results)
        run_composite = self._aggregate_composite(results)

        run = EvaluationRun(
            run_id=str(uuid4()),  # placeholder; replaced by the persisted id below
            config=self.config,
            status=status,
            composite_score=run_composite,
            metric_averages=metric_averages,
            started_at=started_at,
            completed_at=completed_at,
            results=results,
        )

        run_id = self._persist(run)
        run = run.model_copy(update={"run_id": run_id})

        logger.info(
            "Finished evaluation run %s: %d row(s), status=%s (%d failed), composite=%s",
            run_id,
            total,
            status,
            sum(1 for r in results if not r.success),
            f"{run_composite:.4f}" if run_composite is not None else "n/a",
        )
        self._emit_progress(current=total, total=total, status=status)

        return run

    async def _evaluate_row(
        self,
        index: int,
        row: Dict[str, Any],
        prompt_version: PromptVersionResponse,
        provider: BaseLLMProvider,
    ) -> EvaluationRowResult:
        """Build, dispatch, and score a single row, capturing any failure.

        Three independent failure points are handled without raising:
        template rendering (a missing ``{variable}``), the provider call
        (``ProviderError`` or any unexpected exception), and metric
        scoring (against ``self._scorers``, resolved once in ``__init__``).
        Each is logged and turned into a failed ``EvaluationRowResult`` so
        the run continues.
        """
        # 1. Build the request.
        try:
            request = self.request_builder.build_request(prompt_version, row)
        except MissingTemplateVariableError as exc:
            logger.error("Row %d skipped: %s", index, exc)
            return EvaluationRowResult(row_index=index, question_row=row, success=False, error=str(exc))

        # 2. Call the provider, timing the call. ``LLMResponse`` already
        #    carries ``latency_ms``, but we time defensively in case a
        #    provider under-reports it.
        loop = asyncio.get_running_loop()
        call_start = loop.time()
        try:
            # BaseLLMProvider.generate() is synchronous; run it in a worker
            # thread so a slow provider call doesn't block the event loop.
            response = await asyncio.to_thread(provider.generate, request)
        except ProviderError as exc:
            logger.error("Row %d failed calling provider: %s", index, exc)
            return EvaluationRowResult(
                row_index=index, question_row=row, request=request, success=False, error=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - a single row must never crash the whole run
            logger.exception("Row %d failed with an unexpected error", index)
            return EvaluationRowResult(
                row_index=index,
                question_row=row,
                request=request,
                success=False,
                error=f"Unexpected error: {exc}",
            )
        measured_latency_ms = (loop.time() - call_start) * 1000.0

        # 3. Score the response with all metrics + composite.
        try:
            metric_scores, composite = self._score_row(row, response, measured_latency_ms)
        except Exception as exc:  # noqa: BLE001 - scoring failure must not crash the run
            logger.exception("Row %d failed during metric scoring", index)
            return EvaluationRowResult(
                row_index=index,
                question_row=row,
                request=request,
                response=response,
                success=False,
                error=f"Scoring error: {exc}",
            )

        return EvaluationRowResult(
            row_index=index,
            question_row=row,
            request=request,
            response=response,
            metric_scores=metric_scores,
            composite_score=composite,
            success=True,
        )

    def _score_row(
        self,
        row: Dict[str, Any],
        response: LLMResponse,
        measured_latency_ms: float,
    ) -> tuple[Dict[str, float], float]:
        """Run every scorer for one row and aggregate the composite score.

        Builds the scorer ``metadata`` dict from the response and dataset
        row, invokes each of the six scorers in ``self._scorers`` (resolved
        once in ``__init__``), and feeds their scores into the composite
        scorer.

        Args:
            row: The raw dataset row (source of ground truth and rule
                metadata such as ``domain``, ``difficulty``,
                ``expected_format``).
            response: The provider's response.
            measured_latency_ms: Orchestrator-measured wall-clock latency,
                used when the provider under-reports ``latency_ms``.

        Returns:
            A ``(metric_scores, composite_score)`` tuple, where
            ``metric_scores`` maps each ``COMPONENT_KEYS`` entry to its
            score in ``[0, 1]``.
        """
        reference = self._reference_for(row)
        metadata = self._build_metadata(row, response, measured_latency_ms)

        metric_results: Dict[str, MetricResult] = {}
        for key in COMPONENT_KEYS:
            scorer = self._scorers[key]
            metric_results[key] = scorer.evaluate(response.text, reference, metadata)

        composite_result = self.composite_scorer.compute(metric_results)
        metric_scores = {key: result.score for key, result in metric_results.items()}
        return metric_scores, composite_result.score

    @staticmethod
    def _build_metadata(
        row: Dict[str, Any], response: LLMResponse, measured_latency_ms: float
    ) -> Dict[str, Any]:
        """Assemble the ``metadata`` dict passed to every scorer.

        Includes latency and token usage (consumed by ``LatencyScorer`` /
        ``CostScorer``), dataset descriptors (``domain``, ``difficulty``),
        and the optional ``expected_format`` rule (consumed by
        ``InstructionScorer``) only when present on the row.
        """
        latency_ms = response.latency_ms if response.latency_ms else measured_latency_ms
        metadata: Dict[str, Any] = {
            "latency_ms": latency_ms,
            "token_usage": dict(response.token_usage),
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
        }
        if "expected_format" in row:
            metadata["expected_format"] = row["expected_format"]
        return metadata

    @staticmethod
    def _reference_for(row: Dict[str, Any]) -> str:
        """Extract the ground-truth reference text from a dataset row.

        Falls back to an empty-safe placeholder only when a row carries no
        ground truth; scorers that require a non-empty reference will then
        raise, which the caller records as a per-row scoring failure.
        """
        return str(row.get("ground_truth") or "")

    def _get_scorers(self) -> Dict[str, Metric]:
        """Resolve the six scorers once, honoring an injected override.

        Called exactly once, from ``__init__``, and the result is cached on
        ``self._scorers`` for the lifetime of this orchestrator instance.
        Constructing the default scorers loads real ML models (embeddings,
        NLI), which is why callers that build many orchestrators (or an
        orchestrator per evaluation run, as the dashboard does) should
        inject an already-built ``scorers`` dict -- e.g. one cached with
        Streamlit's ``@st.cache_resource`` -- rather than let this build a
        fresh set every time.
        """
        if self._injected_scorers is not None:
            missing = [key for key in COMPONENT_KEYS if key not in self._injected_scorers]
            if missing:
                raise ValueError(f"injected scorers is missing component(s): {missing}")
            return self._injected_scorers

        # Imported here so unit tests that inject scorers never trigger the
        # heavy transformers / sentence-transformers imports.
        from metrics.accuracy import AccuracyScorer
        from metrics.cost import CostScorer
        from metrics.hallucination import HallucinationScorer
        from metrics.instruction import InstructionScorer
        from metrics.latency import LatencyScorer
        from metrics.safety import SafetyScorer

        return {
            "accuracy": AccuracyScorer(),
            "hallucination": HallucinationScorer(),
            "instruction": InstructionScorer(),
            "safety": SafetyScorer(),
            "latency": LatencyScorer(),
            "cost": CostScorer(),
        }

    def _load_prompt_version(self) -> PromptVersionResponse:
        """Resolve ``config.prompt_version_id`` via ``PromptService``."""
        return self.prompt_service.get_version_by_id(self.config.prompt_version_id)

    def _load_dataset_rows(self) -> List[Dict[str, Any]]:
        """Load every row of ``config.dataset_version_id`` via ``BenchmarkService``."""
        return self.benchmark_service.load_dataset_rows(self.config.dataset_version_id)

    def _get_provider(self) -> BaseLLMProvider:
        """Resolve the provider instance for this run, honoring an injected override."""
        if self._injected_provider is not None:
            return self._injected_provider
        return ProviderFactory.create(self.config.provider_name, self.config.model_name)

    def _persist(self, run: EvaluationRun) -> str:
        """Persist the run, its results, and its aggregate metrics.

        Partial results are always persisted, including for a ``"failed"``
        run (> 50% of rows failed), per the error-handling contract.

        Returns:
            The persisted ``run_id`` (DB primary key).
        """
        run_id = self.persistence.save_run(run)
        self.persistence.save_results(run_id, run.results)
        self.persistence.save_metrics(run_id, run.metric_averages)
        return run_id

    def _emit_progress(self, current: int, total: int, status: str) -> None:
        """Invoke the progress callback, swallowing any callback error.

        A faulty UI hook must never be able to abort an evaluation run, so
        exceptions from the callback are logged and suppressed.
        """
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({"current": current, "total": total, "status": status})
        except Exception:  # noqa: BLE001 - progress reporting is best-effort
            logger.warning("progress_callback raised; continuing run", exc_info=True)

    @staticmethod
    def _aggregate_metric_averages(results: List[EvaluationRowResult]) -> Dict[str, float]:
        """Mean of each component score across successful rows only.

        Failed rows contribute nothing to the average (per spec). Returns
        an empty dict if no rows succeeded.
        """
        successful = [r for r in results if r.success]
        if not successful:
            return {}
        averages: Dict[str, float] = {}
        for key in COMPONENT_KEYS:
            scores = [r.metric_scores[key] for r in successful if key in r.metric_scores]
            if scores:
                averages[key] = sum(scores) / len(scores)
        return averages

    @staticmethod
    def _aggregate_composite(results: List[EvaluationRowResult]) -> Optional[float]:
        """Mean composite score across successful rows only, or ``None``."""
        composites = [r.composite_score for r in results if r.success and r.composite_score is not None]
        if not composites:
            return None
        return sum(composites) / len(composites)

    @staticmethod
    def _determine_status(results: List[EvaluationRowResult]) -> RunStatus:
        """Derive the run's overall status from its per-row outcomes.

        Applies the ">50% rows fail -> failed" rule; any failures at or
        below that threshold yield ``"completed_with_errors"``, and none
        yields ``"completed"``.
        """
        if not results:
            return "completed"
        failures = sum(1 for r in results if not r.success)
        if failures == 0:
            return "completed"
        if failures / len(results) > FAILURE_THRESHOLD:
            return "failed"
        return "completed_with_errors"