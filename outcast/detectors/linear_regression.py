import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing_extensions import Self

from outcast.base import BaseConditionalDetector
from outcast.utils import _validate_context_cols


class ConditionalLinearRegression(BaseConditionalDetector):
    """Flags points whose target deviates from a linear model *conditioned on context*.

    target_col is modelled as a linear function of context_cols. the absolute
    standardised residual provides the anomaly score.
    """

    def _fit_core(self, X: pd.DataFrame, y: None = None) -> Self:  # noqa: ARG002
        self.context_cols_ = _validate_context_cols(X, self.context_cols)
        if self.target_col not in X.columns:
            raise ValueError(f"target column '{self.target_col}' missing from X")

        Xc = X[self.context_cols_].to_numpy()
        y_t = X[self.target_col].to_numpy()
        # linear model p(target | context)
        self._lr = LinearRegression()
        self._lr.fit(X=Xc, y=y_t)

        # residuals normalised by std dev
        resid = y_t - self._lr.predict(Xc)
        self._resid_std_ = resid.std(ddof=1) or 1e-12  # avoid div zero
        self._scores_fit_ = np.abs(resid) / self._resid_std_

        self.n_features_in_ = X.shape[1]
        return self

    def _score_core(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        _validate_context_cols(X, self.context_cols_)
        if self.target_col not in X.columns:
            raise ValueError(f"target column '{self.target_col}' missing from X")

        Xc = X[self.context_cols_].to_numpy()
        y_t = X[self.target_col].to_numpy()
        resid: npt.NDArray[np.float64] = y_t - self._lr.predict(Xc)
        return np.abs(resid) / self._resid_std_
