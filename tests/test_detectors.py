import numpy as np
import pandas as pd

from outcast import ConditionalLinearRegression
from outcast.contamination import ContaminationStrategy
from outcast.detectors.random_forest import ConditionalGradientBoostingQuantile
from outcast.utils import dummify_object_cols


def _numeric_toy_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "alt": rng.normal(800, 100, n),
            "rain": rng.normal(1500, 200, n),
        }
    )
    df["yield_kg"] = 0.4 * df.alt - 0.2 * df.rain + rng.normal(0, 50, n)

    outlier_mask = df["id"].isin([0, 1, 2])
    df.loc[outlier_mask, "yield_kg"] += 500  # inject anomalies
    return df


def _toy_categorical_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "alt": rng.normal(800, 300, n),
            "rain": rng.normal(1500, 200, n),
            "country": rng.choice(["brazil", "colombia", "uganda"], size=n),
        }
    )
    df["yield_kg"] = (
        (df["country"] == "brazil") * (0.2 * df.alt - 0.1 * df.rain + rng.normal(800, 350, n))
        + (df["country"] == "colombia") * (0.2 * df.alt - 0.1 * df.rain + rng.normal(400, 150, n))
        + (df["country"] == "uganda") * (0.2 * df.alt - 0.1 * df.rain + rng.normal(200, 150, n))
    )
    outlier_mask = df["id"].isin([0, 1, 2])
    df.loc[outlier_mask, "yield_kg"] += 1000  # inject anomalies
    return df


def test_linear_detector_flags_planted_outliers_contamination() -> None:
    df = _numeric_toy_df(n=300)

    clf = ConditionalLinearRegression(
        context_cols=["alt", "rain"],
        target_col="yield_kg",
        contamination=0.01,
    ).fit(df)

    labels = clf.predict(df)  # 1 = outlier
    flagged_ids = set(df.loc[labels == 1, "id"])  # which rows got flagged

    assert flagged_ids == {0, 1, 2}, f"unexpected flags: {flagged_ids}"


def test_linear_detector_flags_planted_outliers_no_contamination() -> None:
    """Test that the detector flags the same outliers as above, but without contamination.

    Uses the QUADRUPLE_MAD strategy to estimate contamination.
    """
    df = _numeric_toy_df(n=300, seed=15)  # seed is explicitly set to detect the outliers

    clf = ConditionalLinearRegression(
        context_cols=["alt", "rain"],
        target_col="yield_kg",
        contamination_strategy=ContaminationStrategy.QUADRUPLE_MAD,
    ).fit(df)

    labels = clf.predict(df)  # 1 = outlier
    flagged_ids = set(df.loc[labels == 1, "id"])  # which rows got flagged
    assert {0, 1, 2} == flagged_ids, f"unexpected flags: {flagged_ids}"


def test_linear_detector_categorical_context() -> None:
    """Test that the detector works with categorical context columns."""
    df = _toy_categorical_df(n=300, seed=10)  # seed is explicitly set to detect the outliers
    df, context_cols = dummify_object_cols(
        df, context_cols=["alt", "rain", "country"]
    )  # one-hot encode categorical columns

    clf = ConditionalLinearRegression(
        context_cols=context_cols,
        target_col="yield_kg",
        contamination=0.01,
    ).fit(df)

    labels = clf.predict(df)  # 1 = outlier
    flagged_ids = set(df.loc[labels == 1, "id"])  # which rows got flagged

    assert {0, 1, 2} == flagged_ids, f"unexpected flags: {flagged_ids}"


def test_random_forest_categorical() -> None:
    """Test that the detector works with categorical context columns."""
    df = _toy_categorical_df(n=300, seed=10)  # seed is explicitly set to detect the outliers
    df, context_cols = dummify_object_cols(
        X=df, context_cols=["alt", "rain", "country"]
    )  # one-hot encode categorical columns

    clf = ConditionalGradientBoostingQuantile(
        context_cols=context_cols,
        target_col="yield_kg",
        contamination=0.015,
    ).fit(df)

    labels = clf.predict(df)  # 1 = outlier
    flagged_ids = set(df.loc[labels == 1, "id"])  # which rows got flagged

    assert {0, 1, 2}.intersection(flagged_ids), f"unexpected flags: {flagged_ids}"
