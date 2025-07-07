from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from outcast.base import BaseConditionalDetector
from outcast.contamination import ContaminationStrategy


class ConditionalGradientBoostingQuantile(BaseConditionalDetector):
    """Tree-based conditional outlier detector for tabular data.

    Parameters
    ----------
    target_col : str
    context_cols : list[str]
    quantile : float, default=0.5
        Which conditional quantile to model (0 < q < 1).
    contamination : float | None, default=None
    random_state : int | None, default=None
    gbr_kwargs : dict[str, Any], passed to HistGradientBoostingRegressor

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        target_col: str,
        context_cols: list[str],
        quantile: float = 0.5,
        contamination: float | None = None,
        contamination_strategy: ContaminationStrategy = ContaminationStrategy.QUADRUPLE_MAD,
        random_state: int | None = None,
        gbr_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            target_col=target_col,
            context_cols=context_cols,
            contamination=contamination,
            contamination_strategy=contamination_strategy,
            random_state=random_state,
        )
        self.quantile = quantile
        self.gbr_kwargs = gbr_kwargs or {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        num_cols = [c for c in self.context_cols if X[c].dtype.kind in "fiu"]
        cat_cols = list(set(self.context_cols) - set(num_cols))

        transformers = []
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))

        encoder = ColumnTransformer(transformers)
        gbr = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile,
            random_state=self.random_state,
            **self.gbr_kwargs,
        )
        return Pipeline([("encode", encoder), ("gbr", gbr)])

    # ------------------------------------------------------------------
    # required abstract methods
    # ------------------------------------------------------------------
    def _fit_core(self: Self, X: pd.DataFrame, y: None = None) -> Self:  # noqa: ARG002
        if self.target_col not in X.columns:
            raise ValueError(f"target column '{self.target_col}' missing from X")

        self._pipe = self._build_pipeline(X)
        self._pipe.fit(X[self.context_cols], X[self.target_col])

        resid = X[self.target_col].to_numpy() - self._pipe.predict(X[self.context_cols])
        scale = np.median(np.abs(resid)) * 1.4826 or 1e-12
        self._scale_ = scale
        self._scores_fit_ = np.abs(resid) / scale
        self.n_features_in_ = X.shape[1]
        return self

    def _score_core(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        check_is_fitted(self, ["_pipe", "_scale_"])
        if self.target_col not in X.columns:
            raise ValueError(f"target column '{self.target_col}' missing from X")

        resid: npt.NDArray[np.float64] = X[self.target_col].to_numpy() - self._pipe.predict(X[self.context_cols])
        return np.abs(resid) / self._scale_
