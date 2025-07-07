from collections.abc import Callable
from enum import Enum, auto

import numpy as np
import numpy.typing as npt


class ContaminationStrategy(Enum):
    """Enumeration of built-in auto-contamination heuristics."""

    QUADRUPLE_MAD = auto()
    DOUBLE_MAD = auto()
    TUKEY = auto()


def _contam_quadruple_mad(scores: npt.NDArray[np.float64]) -> float:
    """Median + 4xMAD (scaled) → robust proportion."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) * 1.4826 or 1e-12
    cutoff = med + 4 * mad
    return float((scores > cutoff).mean())


def _contam_double_mad(scores: npt.NDArray[np.float64]) -> float:
    """Median + 2xMAD (scaled) → robust proportion."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) * 1.4826 or 1e-12
    cutoff = med + 2 * mad
    return float((scores > cutoff).mean())


def _contam_tukey(scores: npt.NDArray[np.float64]) -> float:
    """Q3 + 1.5xIQR (Tukey box plot whisker)."""
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1 or 1e-12
    cutoff = q3 + 1.5 * iqr
    return float((scores > cutoff).mean())


_CONTAMINATION_FUNCTIONS: dict[ContaminationStrategy, Callable[[npt.NDArray[np.float64]], float]] = {
    ContaminationStrategy.QUADRUPLE_MAD: _contam_quadruple_mad,
    ContaminationStrategy.DOUBLE_MAD: _contam_double_mad,
    ContaminationStrategy.TUKEY: _contam_tukey,
}


def _estimate_contamination(
    scores: npt.NDArray[np.float64],
    strategy: ContaminationStrategy,
) -> float:
    """Return estimated contamination, clipped to [0.005, 0.4]."""
    if callable(strategy):
        prop = float(strategy(scores))
    else:
        try:
            prop = _CONTAMINATION_FUNCTIONS[strategy](scores)
        except KeyError as exc:
            raise ValueError(
                f"unknown auto_strategy '{strategy}'. choose one of {list(_CONTAMINATION_FUNCTIONS)}"
            ) from exc

    return float(np.clip(prop, 0.005, 0.4))
