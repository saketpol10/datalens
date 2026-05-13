import numpy as np
import pandas as pd

from models.schemas import ColumnProfile, DatasetProfile

_MAX_SAMPLE_ROWS = 5
_MAX_TOP_VALUES = 5
_MAX_CORRELATIONS = 6


def _profile_numeric(series: pd.Series) -> dict:
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    outliers = int(((series < lower) | (series > upper)).sum())
    return {
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "median": round(float(series.median()), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "outlier_count": outliers,
    }


def _profile_categorical(series: pd.Series) -> dict:
    counts = series.value_counts()
    top = [(str(k), int(v)) for k, v in counts.head(_MAX_TOP_VALUES).items()]
    return {"top_values": top}


def _top_correlations(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "r"})
        .sort_values("r", ascending=False)
        .head(_MAX_CORRELATIONS)
    )
    return [(row.a, row.b, round(row.r, 3)) for row in pairs.itertuples()]


def profile(df: pd.DataFrame) -> DatasetProfile:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = [
        c for c in df.columns if c not in numeric_cols and c not in datetime_cols
    ]

    columns = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        base = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            missing_count=missing,
            missing_pct=round(missing / len(df) * 100, 2),
            unique_count=int(s.nunique()),
        )
        if col in numeric_cols:
            base = base.model_copy(update=_profile_numeric(s.dropna()))
        elif col in categorical_cols:
            base = base.model_copy(update=_profile_categorical(s.dropna()))
        columns.append(base)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    sample = df.head(_MAX_SAMPLE_ROWS).fillna("").astype(str).to_dict(orient="records")

    return DatasetProfile(
        rows=len(df),
        cols=len(df.columns),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=round(missing_cells / total_cells * 100, 2),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        columns=columns,
        top_correlations=_top_correlations(df),
        sample_rows=sample,
        duplicate_rows=int(df.duplicated().sum()),
    )
