import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from outcast.contamination import ContaminationStrategy, _estimate_contamination


class BaseConditionalDetector:
    """abstract mix-in for conditional outlier detectors.

    Parameters
    ----------
    context_cols : list[str] | None
        Columns in *X* that define the conditioning context. For now,
        we don't allow for none contextual outliers.
    contamination : float, default=0.1
        Expected proportion of outliers in the data. Used for generating
        binary outlier labels.
        If None, the contamination rate is estimated from the data.
    random_state : int | None, default=None
        Reproducible randomness for child estimators.

    """

    def __init__(
        self,
        *,
        target_col: str,
        context_cols: list[str],
        contamination: float | None = None,
        contamination_strategy: ContaminationStrategy = ContaminationStrategy.QUADRUPLE_MAD,
        random_state: int | None = None,
    ) -> None:
        """Initialize base conditional detector."""
        self.target_col = target_col
        self.context_cols = context_cols
        self._contamination = contamination
        self.contamination_strategy = contamination_strategy
        self.random_state = random_state
        # attributes set during fit
        self.threshold_: float  # percentile threshold

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseConditionalDetector":  # noqa: ARG002
        """Fit detector and establish *threshold_* based on *contamination*."""
        X = self._validate_input(X)
        self._fit_core(X)
        scores = self._score_core(X)
        self._scores_fit_ = scores
        percentile = (1.0 - self.contamination) * 100.0
        self.threshold_ = float(np.percentile(scores, percentile))
        return self

    def score_samples(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return anomaly scores (higher ⇒ more anomalous)."""
        X = self._validate_input(X)
        return self._score_core(X)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return 1 for outliers, 0 for inliers, using learned *threshold_*."""
        scores = self.score_samples(X)
        return (scores > self.threshold_).astype(int)

    @property
    def contamination(self) -> float:
        """Return contamination rate, either set by user or estimated from data.

        Note: For contamination to be calculated from data, the detector must be fitted first.
        """
        if self._contamination is not None:
            return self._contamination

        return _estimate_contamination(self._scores_fit_, self.contamination_strategy)

    # subclasses must implement _fit_core() and _score_core()
    def _fit_core(self: Self, X: pd.DataFrame) -> Self:  # pragma: no cover - abstract
        raise NotImplementedError

    def _score_core(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    # Private utils
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.context_cols is not None:
            missing = set(self.context_cols) - set(X.columns)
            if missing:
                raise ValueError(f"context_cols missing from input: {missing}")
        return X.copy()
