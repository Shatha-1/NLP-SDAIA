"""Lab 6 starter: sliced evaluation report."""
import pandas as pd

from bayan.evaluation.bootstrap import bootstrap_ci

SMALL_SLICE_N = 30


def sliced_report(df: pd.DataFrame, slice_cols: list[str], *, min_n: int = SMALL_SLICE_N,
                   n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per-slice accuracy with a bootstrap CI, flagging slices too small to trust.

    df must have y_true/y_pred columns plus each column named in slice_cols
    (e.g. language, dialect, class/topic, length bucket).
    """
    rows = []
    correct = (df["y_true"] == df["y_pred"]).astype(int)

    for col in slice_cols:
        for value, group_idx in df.groupby(col).groups.items():
            group_correct = correct.loc[group_idx].tolist()
            n = len(group_correct)
            point, lo, hi = bootstrap_ci(group_correct, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "slice_col": col,
                    "slice_value": value,
                    "n": n,
                    "accuracy": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "small_slice": n < min_n,
                }
            )

    return pd.DataFrame(rows)
