"""
registry/dataset_validator.py

Schema validation for benchmark dataset files before they are persisted
as an immutable ``DatasetVersion`` (see database/models.py and
DATABASE_SCHEMA.md — dataset versions must never be edited in place, so
bad data must be caught here, before ``BenchmarkService.upload_dataset``
ever writes a row).

Validation is column-driven, not format-driven: callers are expected to
parse CSV or JSON into a ``pandas.DataFrame`` first (e.g. via
``pd.read_csv`` or ``pd.DataFrame.from_records(json.load(...))``) and
pass that DataFrame in here.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, Field

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "question",
    "ground_truth",
    "domain",
    "category",
    "difficulty",
)

VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})


class RowError(BaseModel):
    """A single row-level validation failure."""

    row_index: int = Field(..., description="Zero-based row index within the DataFrame.")
    column: str = Field(..., description="Column the error applies to.")
    message: str = Field(..., description="Human-readable description of the error.")


class ValidationResult(BaseModel):
    """Outcome of validating a candidate benchmark dataset."""

    is_valid: bool
    missing_columns: List[str] = Field(default_factory=list)
    row_errors: List[RowError] = Field(default_factory=list)

    @property
    def error_summary(self) -> List[str]:
        """Flat, human-readable error strings — missing columns first, then row errors."""
        summary = [f"Missing column: {col}" for col in self.missing_columns]
        summary.extend(f"Row {err.row_index}: {err.message}" for err in self.row_errors)
        return summary


def _is_blank(value: object) -> bool:
    """Return True if a cell is NaN/None or an empty/whitespace-only string."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def validate(df: pd.DataFrame) -> ValidationResult:
    """
    Validate a candidate benchmark dataset.

    Checks, in order:
      1. All of ``REQUIRED_COLUMNS`` are present. If any are missing,
         row-level checks are skipped entirely (there is nothing useful
         to check per-row for a column that doesn't exist).
      2. ``question`` and ``ground_truth`` cells are non-empty for every row.
      3. ``difficulty`` values are one of ``VALID_DIFFICULTIES``
         (case-insensitive).

    Args:
        df: The parsed dataset, e.g. from ``pd.read_csv(uploaded_file)``.

    Returns:
        A ``ValidationResult``. ``is_valid`` is False if any required
        column is missing or any row fails a check.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    row_errors: List[RowError] = []

    if not missing_columns:
        for idx, row in df.iterrows():
            row_index = int(idx)  # type: ignore[arg-type]

            if _is_blank(row["question"]):
                row_errors.append(
                    RowError(row_index=row_index, column="question", message="Empty question cell")
                )

            if _is_blank(row["ground_truth"]):
                row_errors.append(
                    RowError(
                        row_index=row_index,
                        column="ground_truth",
                        message="Empty ground_truth cell",
                    )
                )

            difficulty = row["difficulty"]
            normalized_difficulty = "" if pd.isna(difficulty) else str(difficulty).strip().lower()
            if normalized_difficulty not in VALID_DIFFICULTIES:
                row_errors.append(
                    RowError(
                        row_index=row_index,
                        column="difficulty",
                        message=(
                            f"Invalid difficulty {difficulty!r}; must be one of "
                            f"{sorted(VALID_DIFFICULTIES)}"
                        ),
                    )
                )

    is_valid = not missing_columns and not row_errors
    return ValidationResult(is_valid=is_valid, missing_columns=missing_columns, row_errors=row_errors)


__all__ = ["REQUIRED_COLUMNS", "VALID_DIFFICULTIES", "RowError", "ValidationResult", "validate"]