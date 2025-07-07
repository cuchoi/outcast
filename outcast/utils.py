from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


def dummify_object_cols(X: pd.DataFrame, context_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode object columns in the context_cols."""
    Y = X.copy()
    new_context_cols = list(context_cols)
    for col in context_cols:
        if Y[col].dtype == "object":
            # one hot encode categorical columns
            Y = pd.get_dummies(Y, columns=[col], prefix=f"__outcast_{col}")

            # Add new column names
            new_context_cols.extend(val for val in Y.columns if f"__outcast_{col}" in val)
            new_context_cols.remove(col)
    return Y, new_context_cols


def _validate_context_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"X must be a pandas.DataFrame, got {type(df)}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"context columns missing from X: {missing}")
    return list(cols)


def _quantile_threshold(scores: npt.NDArray[np.float64], contamination: float) -> float:
    MAX_CONTAMINATION = 0.5
    if not 0.0 < contamination < MAX_CONTAMINATION:
        raise ValueError(f"contamination must be in (0, {MAX_CONTAMINATION})")
    return float(np.quantile(scores, 1.0 - contamination))
