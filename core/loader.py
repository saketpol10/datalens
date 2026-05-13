from pathlib import Path

import pandas as pd


def load(path: str) -> tuple[pd.DataFrame, str]:
    """Load a CSV and return (dataframe, filename). Raises on failure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in (".csv", ".tsv", ".txt"):
        raise ValueError(f"Expected a CSV file, got: {p.suffix}")

    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(p, sep=sep, low_memory=False)

    if df.empty:
        raise ValueError("File is empty.")

    # Parse columns that look like dates
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            try:
                converted = pd.to_datetime(sample, infer_datetime_format=True)
                if converted.notna().all():
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass

    return df, p.name
