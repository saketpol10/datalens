# Datalens

A CLI tool that takes any CSV file and produces a full data analysis report — automatically profiling the dataset with pandas, inferring its domain, and generating insights using an LLM.

No configuration needed. Works on any CSV.

Built with Groq (Llama 3.3 70B), pandas, and Pydantic.

## Architecture

```
CSV file
    |
    v
 Loader        Reads CSV, infers column types, parses dates
    |
    v
 Profiler      Pure pandas: shape, missing values, distributions,
               outliers (IQR), correlations, top categorical values
    |
    v
 Domain        LLM infers what the dataset is about from column
 Inference     names, types, and sample rows alone
    |
    v
 Insight       LLM reasons over the statistical profile in domain
 Generation    context: key findings, anomalies, data quality
               issues, recommended next analyses
    |
    v
 Report        Rich terminal output + saved Markdown file
```

## Setup

```bash
git clone https://github.com/saketpol10/datalens
cd datalens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Usage

```bash
python main.py path/to/data.csv
```

Output streams to terminal and is saved as `<filename>_analysis.md`.

## Sample output

```
Step 1/3 — Profiling
Done (0.01s)

Dataset Overview
  Rows            891
  Columns         12
  Missing cells   866 (8.1%)
  Duplicate rows  0
  Numeric cols    7
  Categorical cols 5

Column Profiles
  Column       Type     Missing  Unique  Stats
  Survived     int64    0%       2       mean=0.38, std=0.49, [0, 1]
  Pclass       int64    0%       3       mean=2.31, std=0.84, [1, 3]
  Age          float64  19.87%   88      mean=29.70, std=14.53, 11 outliers
  Fare         float64  0%       248     mean=32.20, std=49.69, 116 outliers
  Cabin        str      77.1%    147     G6(4)  C23 C25 C27(4)
  ...

Top Correlations
  Pclass  <->  Fare      0.549
  SibSp   <->  Parch     0.415
  Pclass  <->  Age       0.369
  Survived <-> Pclass    0.338

Step 2/3 — Inferring domain
Done (0.82s)

Domain: Transportation
The dataset contains Titanic passenger records including demographics,
ticket details, and survival status.
Likely target: Survived

Step 3/3 — Generating insights
Done (2.14s)

Executive Summary
  Only 38.38% of passengers survived, with significant correlations between
  survival and class and fare. Data quality issues — particularly 77.1% missing
  cabin data and 19.87% missing age — limit deeper analysis.

Key Findings
  - Only 38.38% of passengers survived; Pclass correlates with survival (r=0.338)
  - Average fare was $32.20 with high std ($49.69); strong Pclass-Fare link (r=0.549)
  - Age missing for 19.87% of passengers, with a range of 0.42 to 80 years

Anomalies
  - Parch has 213 outliers — some passengers had unusually large travel groups
  - Fare is highly skewed: median $14.45 vs mean $32.20
  - Embarked has 2 missing values, likely data entry errors

Data Quality Issues
  - 19.87% of Age values missing — impacts age-related modelling
  - 77.1% of Cabin data missing — limits location-based analysis

Recommended Analyses
  - Survival rate breakdown by Sex and Age cohort
  - Fare distribution by Pclass and embarkation port
  - Age imputation using KNN or regression before modelling
  - Cabin deck extraction from partial cabin values

Saved → titanic_analysis.md
Total: 3.0s
```

## Design decisions

**No LLM for profiling** — all statistics are computed with pandas. The LLM only receives a compact digest of the results, keeping token usage low and latency fast regardless of dataset size.

**Two-stage LLM reasoning** — domain inference runs first so the insight generation prompt is grounded in context. A generic insight prompt produces generic output; a domain-aware one produces specific, actionable findings.

**IQR outlier detection** — uses the standard 1.5×IQR method per numeric column, reported as a count rather than a list to keep the LLM digest compact.

**Typed data flow** — Pydantic models (`ColumnProfile`, `DatasetProfile`, `DomainInference`, `DataInsights`) enforce structure at every stage, making it straightforward to extend or swap components.

**Graceful handling of large files** — the profiler works on the full dataframe; only a 5-row sample and summary statistics are sent to the LLM, so the tool handles arbitrarily large CSVs without hitting token limits.

## Stack

| Component          | Tool                     |
|--------------------|--------------------------|
| LLM                | Groq API (Llama 3.3 70B) |
| Data profiling     | pandas + numpy           |
| Data validation    | Pydantic v2              |
| Terminal UI        | Rich                     |
| Async runtime      | asyncio                  |
