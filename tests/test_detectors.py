import numpy as np
import pandas as pd
from outcast import ConditionalLinearRegression


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "alt": rng.normal(800, 100, n),
        "rain": rng.normal(1500, 200, n),
    })
    df["yield_kg"] = 0.4*df.alt - 0.2*df.rain + rng.normal(0, 50, n)

    outlier_mask = df["id"].isin([0, 1, 2])
    df.loc[outlier_mask, "yield_kg"] += 500        # inject anomalies
    return df


def test_linear_detector_flags_planted_outliers():
    df = _toy_df(n=300)

    clf = ConditionalLinearRegression(
        context_cols=["alt", "rain"],
        target_col="yield_kg",
        contamination=0.01,
    ).fit(df)

    labels = clf.predict(df)                      # 1 = outlier
    flagged_ids = set(df.loc[labels == 1, "id"])  # which rows got flagged

    assert flagged_ids == {0, 1, 2}, f"unexpected flags: {flagged_ids}"
