import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from core import generate_insights, infer_domain, load, profile
from models.schemas import AnalysisReport, DatasetProfile, DomainInference, DataInsights

load_dotenv()

console = Console()
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _render_profile(p: DatasetProfile) -> None:
    # Overview table
    overview = Table(show_header=False, box=None, padding=(0, 2))
    overview.add_column(style="dim")
    overview.add_column()
    overview.add_row("Rows", f"{p.rows:,}")
    overview.add_row("Columns", str(p.cols))
    overview.add_row("Missing cells", f"{p.missing_cells:,} ({p.missing_pct}%)")
    overview.add_row("Duplicate rows", str(p.duplicate_rows))
    overview.add_row("Numeric cols", str(len(p.numeric_cols)))
    overview.add_row("Categorical cols", str(len(p.categorical_cols)))
    if p.datetime_cols:
        overview.add_row("Datetime cols", str(len(p.datetime_cols)))
    console.print(Panel(overview, title="Dataset Overview", border_style="blue"))

    # Column table
    col_table = Table(title="Column Profiles", border_style="dim")
    col_table.add_column("Column", style="bold")
    col_table.add_column("Type", style="dim")
    col_table.add_column("Missing", justify="right")
    col_table.add_column("Unique", justify="right")
    col_table.add_column("Stats / Top values")

    for col in p.columns:
        missing_str = f"{col.missing_pct}%" if col.missing_pct > 0 else "[green]0%[/green]"
        if col.name in p.numeric_cols:
            stats = f"mean={col.mean}  std={col.std}  [{col.min}, {col.max}]"
            if col.outlier_count:
                stats += f"  [yellow]{col.outlier_count} outliers[/yellow]"
        elif col.top_values:
            stats = "  ".join(f"{v}({c})" for v, c in col.top_values[:3])
        else:
            stats = ""
        col_table.add_row(col.name, col.dtype, missing_str, str(col.unique_count), stats)

    console.print(col_table)

    # Correlations
    if p.top_correlations:
        corr_table = Table(title="Top Correlations", border_style="dim")
        corr_table.add_column("Column A", style="bold")
        corr_table.add_column("Column B", style="bold")
        corr_table.add_column("r", justify="right")
        for a, b, r in p.top_correlations:
            color = "green" if r > 0.7 else "yellow" if r > 0.4 else "white"
            corr_table.add_row(a, b, f"[{color}]{r}[/{color}]")
        console.print(corr_table)


def _render_insights(domain: DomainInference, insights: DataInsights) -> None:
    console.print(
        Panel(
            f"[bold]{domain.domain}[/bold]\n{domain.description}"
            + (f"\n[dim]Likely target: {domain.likely_target}[/dim]" if domain.likely_target else "")
            + (f"\n[dim]Entities: {', '.join(domain.key_entities)}[/dim]" if domain.key_entities else ""),
            title="Domain",
            border_style="cyan",
        )
    )
    console.print(
        Panel(insights.executive_summary, title="Executive Summary", border_style="green")
    )

    def _section(title: str, items: list[str], color: str = "white") -> None:
        if not items:
            return
        content = "\n".join(f"[{color}]•[/{color}] {item}" for item in items)
        console.print(Panel(content, title=title, border_style="dim"))

    _section("Key Findings", insights.key_findings, "cyan")
    _section("Anomalies", insights.anomalies, "yellow")
    _section("Data Quality Issues", insights.data_quality_issues, "red")
    _section("Recommended Analyses", insights.recommended_analyses, "green")


def _save_report(report: AnalysisReport) -> Path:
    p = report.profile
    d = report.domain
    ins = report.insights

    lines = [
        f"# Data Analysis: {report.filename}",
        "",
        f"**Domain:** {d.domain}  ",
        f"**Description:** {d.description}  ",
        f"**Likely target:** {d.likely_target or 'N/A'}  ",
        f"**Key entities:** {', '.join(d.key_entities)}",
        "",
        "## Executive Summary",
        "",
        ins.executive_summary,
        "",
        "## Dataset Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rows | {p.rows:,} |",
        f"| Columns | {p.cols} |",
        f"| Missing cells | {p.missing_cells:,} ({p.missing_pct}%) |",
        f"| Duplicate rows | {p.duplicate_rows} |",
        "",
        "## Key Findings",
        "",
        *[f"- {f}" for f in ins.key_findings],
        "",
        "## Anomalies",
        "",
        *[f"- {a}" for a in ins.anomalies],
        "",
        "## Data Quality Issues",
        "",
        *[f"- {q}" for q in ins.data_quality_issues],
        "",
        "## Recommended Analyses",
        "",
        *[f"- {r}" for r in ins.recommended_analyses],
        "",
        "## Column Profiles",
        "",
    ]

    for col in p.columns:
        lines.append(f"### {col.name} (`{col.dtype}`)")
        lines.append(f"- Missing: {col.missing_count} ({col.missing_pct}%)")
        lines.append(f"- Unique values: {col.unique_count}")
        if col.name in p.numeric_cols:
            lines += [
                f"- Mean: {col.mean}, Std: {col.std}",
                f"- Range: [{col.min}, {col.max}], Median: {col.median}",
                f"- Outliers (IQR): {col.outlier_count}",
            ]
        elif col.top_values:
            top_str = ", ".join(f"{v} ({c})" for v, c in col.top_values)
            lines.append(f"- Top values: {top_str}")
        lines.append("")

    slug = Path(report.filename).stem.replace(" ", "_")[:50]
    out = Path(f"{slug}_analysis.md")
    out.write_text("\n".join(lines))
    return out


async def run(csv_path: str) -> AnalysisReport:
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    # Load
    console.print(f"\n[dim]Loading {csv_path}...[/dim]")
    df, filename = load(csv_path)
    console.print(
        Panel(f"[bold cyan]{filename}[/bold cyan]", title="Data Analyst", padding=(1, 4))
    )
    console.print(f"[dim]Model: {MODEL}[/dim]\n")

    # Profile (pandas — no LLM)
    console.print("[bold yellow]Step 1/3 — Profiling[/bold yellow]")
    t0 = time.perf_counter()
    p = profile(df)
    console.print(f"[green]Done ({time.perf_counter()-t0:.2f}s)[/green]\n")
    _render_profile(p)

    # Domain inference
    console.print(Rule(style="dim"))
    console.print("\n[bold yellow]Step 2/3 — Inferring domain[/bold yellow]")
    t1 = time.perf_counter()
    domain = await infer_domain(client, p, MODEL)
    console.print(f"[green]Done ({time.perf_counter()-t1:.2f}s)[/green]")

    # Insights
    console.print("\n[bold yellow]Step 3/3 — Generating insights[/bold yellow]")
    t2 = time.perf_counter()
    insights = await generate_insights(client, p, domain, MODEL)
    console.print(f"[green]Done ({time.perf_counter()-t2:.2f}s)[/green]\n")

    console.print(Rule(style="dim"))
    _render_insights(domain, insights)

    # Save
    report = AnalysisReport(filename=filename, profile=p, domain=domain, insights=insights)
    out = _save_report(report)
    total = time.perf_counter() - t0
    console.print(f"\n[blue]Saved → {out.resolve()}[/blue]")
    console.print(f"[dim]Total: {total:.1f}s[/dim]")

    return report


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python main.py <path/to/file.csv>[/red]")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
