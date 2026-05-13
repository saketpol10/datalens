from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    # numeric
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    outlier_count: int | None = None
    # categorical
    top_values: list[tuple[str, int]] | None = None


class DatasetProfile(BaseModel):
    rows: int
    cols: int
    total_cells: int
    missing_cells: int
    missing_pct: float
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    columns: list[ColumnProfile]
    top_correlations: list[tuple[str, str, float]]
    sample_rows: list[dict]
    duplicate_rows: int


class DomainInference(BaseModel):
    domain: str = Field(description="Industry or domain, e.g. 'E-commerce', 'Healthcare'")
    description: str = Field(description="One sentence on what this dataset represents")
    likely_target: str | None = Field(description="Most likely target variable for ML, if any")
    key_entities: list[str] = Field(description="Main real-world entities in the data")


class DataInsights(BaseModel):
    key_findings: list[str] = Field(description="3-5 most notable patterns or facts")
    anomalies: list[str] = Field(description="Outliers, skew, or unexpected distributions")
    data_quality_issues: list[str] = Field(description="Missing data, duplicates, type issues")
    recommended_analyses: list[str] = Field(description="Next analyses worth doing")
    executive_summary: str = Field(description="2-3 sentence narrative for a non-technical reader")


class AnalysisReport(BaseModel):
    filename: str
    profile: DatasetProfile
    domain: DomainInference
    insights: DataInsights
