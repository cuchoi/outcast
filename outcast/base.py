from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseConditionalDetector(BaseEstimator):
    """abstract mix‑in for conditional outlier detectors.

    Parameters
    ----------
    context_cols : list[str] | None
        Columns in *X* that define the conditioning context. If *None*, the
        detector reduces to its global (non‑conditional) variant.
    contamination : float, default=0.1
        Expected proportion of outliers in the data. Used to set the detection
        threshold after *fit*.
    random_state : int | None, default=None
        Reproducible randomness for child estimators.
    """

    def __init__(
        self,
        *,
        context_cols: Optional[Sequence[str]] = None,
        contamination: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.context_cols = context_cols
        self.contamination = contamination
        self.random_state = random_state
        # attributes set during fit
        self.threshold_: float  # percentile threshold

    # ---------------------------------------------------------------------
    # public api – subclasses must implement _fit_core() and _score_core()
    # ---------------------------------------------------------------------
    def fit(self, X, y=None):  # noqa: D401, N802 – sklearn signature
        """Fit detector and establish *threshold_* based on *contamination*."""
        X = self._validate_input(X)
        self._fit_core(X)
        scores = self._score_core(X)
        # lower scores ⇒ more normal in IsolationForest; invert so higher ⇒ anomalous
        percentile = (1.0 - self.contamination) * 100.0
        self.threshold_ = np.percentile(scores, percentile)
        print(f"Threshold set to: {self.threshold_}")
        return self

    def score_samples(self, X):  # noqa: N802 – sklearn signature
        """Return anomaly scores (higher ⇒ more anomalous)."""
        X = self._validate_input(X)
        return self._score_core(X)

    def predict(self, X):  # noqa: N802 – sklearn signature
        """Return 1 for outliers, 0 for inliers, using learned *threshold_*."""
        scores = self.score_samples(X)
        return (scores > self.threshold_).astype(int)

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------
    def _fit_core(self, X: pd.DataFrame) -> None:  # pragma: no cover – abstract
        raise NotImplementedError

    def _score_core(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.context_cols is not None:
            missing = set(self.context_cols) - set(X.columns)
            if missing:
                raise ValueError(f"context_cols missing from input: {missing}")
        return X.copy()
