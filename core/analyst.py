import json

from openai import AsyncOpenAI

from models.schemas import DataInsights, DatasetProfile, DomainInference


def _format_digest(profile: DatasetProfile) -> str:
    lines = [
        f"Shape: {profile.rows} rows x {profile.cols} columns",
        f"Missing: {profile.missing_pct}% of all cells",
        f"Duplicates: {profile.duplicate_rows} rows",
        "",
    ]

    if profile.numeric_cols:
        lines.append("NUMERIC COLUMNS:")
        for col in profile.columns:
            if col.name not in profile.numeric_cols:
                continue
            lines.append(
                f"  {col.name}: mean={col.mean}, std={col.std}, "
                f"range=[{col.min}, {col.max}], median={col.median}, "
                f"missing={col.missing_pct}%, outliers={col.outlier_count}"
            )

    if profile.categorical_cols:
        lines.append("\nCATEGORICAL COLUMNS:")
        for col in profile.columns:
            if col.name not in profile.categorical_cols:
                continue
            top = ", ".join(f"{v}({c})" for v, c in (col.top_values or []))
            lines.append(
                f"  {col.name}: {col.unique_count} unique, missing={col.missing_pct}%"
                + (f", top: {top}" if top else "")
            )

    if profile.datetime_cols:
        lines.append(f"\nDATETIME COLUMNS: {', '.join(profile.datetime_cols)}")

    if profile.top_correlations:
        lines.append("\nTOP CORRELATIONS:")
        for a, b, r in profile.top_correlations:
            lines.append(f"  {a} <-> {b}: {r}")

    if profile.sample_rows:
        lines.append("\nSAMPLE ROWS (first 5):")
        for row in profile.sample_rows:
            lines.append(f"  {row}")

    return "\n".join(lines)


_DOMAIN_SYSTEM = "You are a data scientist. Infer the domain and context of a dataset from its structure. Return only valid JSON."

_DOMAIN_USER = """Identify the domain and context of this dataset.

{digest}

Return JSON:
{{
  "domain": "...",
  "description": "...",
  "likely_target": "...",
  "key_entities": ["...", "..."]
}}"""

_INSIGHTS_SYSTEM = "You are a senior data analyst. Generate precise, non-obvious insights from dataset statistics. Return only valid JSON."

_INSIGHTS_USER = """Dataset domain: {domain} — {description}

{digest}

Analyze this dataset and return JSON:
{{
  "key_findings": ["3-5 specific, non-obvious findings with numbers"],
  "anomalies": ["outliers, skewed distributions, unexpected patterns"],
  "data_quality_issues": ["concrete issues: missing data, duplicates, type problems"],
  "recommended_analyses": ["3-4 specific next analyses worth doing"],
  "executive_summary": "2-3 sentences for a non-technical reader"
}}"""


async def infer_domain(
    client: AsyncOpenAI, profile: DatasetProfile, model: str
) -> DomainInference:
    digest = _format_digest(profile)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DOMAIN_SYSTEM},
            {"role": "user", "content": _DOMAIN_USER.format(digest=digest)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return DomainInference(**json.loads(response.choices[0].message.content))


async def generate_insights(
    client: AsyncOpenAI,
    profile: DatasetProfile,
    domain: DomainInference,
    model: str,
) -> DataInsights:
    digest = _format_digest(profile)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _INSIGHTS_SYSTEM},
            {
                "role": "user",
                "content": _INSIGHTS_USER.format(
                    domain=domain.domain,
                    description=domain.description,
                    digest=digest,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return DataInsights(**json.loads(response.choices[0].message.content))
