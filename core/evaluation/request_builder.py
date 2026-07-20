"""core/evaluation/request_builder.py

Turns a ``(prompt_version, question_row)`` pair into a normalized
``LLMRequest`` ready to hand to a ``BaseLLMProvider``.
"""

from __future__ import annotations

import re
import string
from typing import Any, List, Mapping, Set

from providers.base import LLMRequest
from registry.schemas import PromptVersionResponse

from core.evaluation.config import EvaluationConfig

# Splits a str.format field name like "user.name" or "items[0]" down to its
# base name ("user", "items") for the presence check against a flat row.
_FIELD_NAME_SPLIT_RE = re.compile(r"[.\[]")


class MissingTemplateVariableError(ValueError):
    """Raised when a prompt template references a variable absent from the row.

    Attributes:
        variable_name: The missing placeholder's name.
        available_columns: The columns actually present on the offending row.
    """

    def __init__(self, variable_name: str, available_columns: List[str]) -> None:
        self.variable_name = variable_name
        self.available_columns = available_columns
        super().__init__(
            f"Prompt template references '{{{variable_name}}}', but the "
            f"dataset row has no such column. Available columns: "
            f"{available_columns}."
        )


class RequestBuilder:
    """Fills a prompt template with dataset row values to build ``LLMRequest``s.

    Generation parameters (``temperature``, ``max_tokens``) come from the
    bound ``EvaluationConfig`` and are applied identically to every
    request this instance builds; only the rendered prompt text varies
    per row.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Bind this builder to a run's config, for temperature/max_tokens.

        Args:
            config: The evaluation run's configuration. Its
                ``temperature`` and ``max_tokens`` are applied to every
                ``LLMRequest`` this builder produces.
        """
        self.config = config

    def build_request(
        self,
        prompt_version: PromptVersionResponse,
        question_row: Mapping[str, Any],
    ) -> LLMRequest:
        """Build a normalized ``LLMRequest`` from a prompt version and a row.

        ``prompt_version.content`` is treated as a ``str.format``-style
        template: ``{variable}`` placeholders are substituted with values
        from ``question_row`` (e.g. ``{context}``, ``{question}``), and
        ``{{`` / ``}}`` are literal braces -- matching the escaping rules
        ``PromptService.validate_prompt_syntax`` already enforces at
        prompt-creation time.

        The rendered template becomes the request's ``user_prompt``.
        ``system_prompt`` is left unset (``None``): neither
        ``PromptVersion`` nor ``EvaluationConfig`` currently carries a
        separate system-prompt field, so there is nothing to populate it
        with at this stage.

        Args:
            prompt_version: The immutable prompt version to render.
            question_row: One dataset row, as a flat mapping of column
                name to value (e.g. one record loaded from a dataset
                CSV).

        Returns:
            An ``LLMRequest`` with the rendered template as
            ``user_prompt`` and this builder's config applied as
            ``temperature``/``max_tokens``.

        Raises:
            MissingTemplateVariableError: If the template references a
                placeholder that is not a key on ``question_row``.
        """
        template = prompt_version.content
        required_variables = self._extract_variable_names(template)

        missing = sorted(name for name in required_variables if name not in question_row)
        if missing:
            raise MissingTemplateVariableError(
                variable_name=missing[0],
                available_columns=list(question_row.keys()),
            )

        rendered = template.format(**question_row)

        return LLMRequest(
            user_prompt=rendered,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @staticmethod
    def _extract_variable_names(template: str) -> Set[str]:
        """Return the set of top-level ``{variable}`` names referenced by a template.

        Uses ``string.Formatter`` to parse the template exactly the way
        ``str.format`` will, so ``{{``/``}}`` escaping is honored
        automatically and literal text is ignored. Dotted/indexed field
        names (e.g. ``{user.name}``) are reduced to their base name for
        the presence check, since dataset rows are flat mappings.
        """
        names: Set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(template):
            if not field_name:
                continue
            base_name = _FIELD_NAME_SPLIT_RE.split(field_name, maxsplit=1)[0]
            if base_name:
                names.add(base_name)
        return names