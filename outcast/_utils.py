
from typing import Sequence
import numpy as np
import pandas as pd


def _validate_context_cols(df: pd.DataFrame, cols: Sequence[str]) -> Sequence[str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame, got %r" % type(df))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"context columns missing from X: {missing}")
    return list(cols)


def _quantile_threshold(scores: np.ndarray, contamination: float) -> float:
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be in (0, 0.5)")
    return np.quantile(scores, 1.0 - contamination)
