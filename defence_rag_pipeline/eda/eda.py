"""
EDA & visualizations for SIPRI + GDELT.

Produces a handful of plots analysts (and your teammate's Analysis /
Risk Assessment agents) actually care about, and saves them as PNGs plus
a short markdown summary with the key numbers.
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_top_spenders(sipri_df: pd.DataFrame, year: int, out_path: Path, top_n: int = 10) -> None:
    year_df = sipri_df[sipri_df["year"] == year].nlargest(top_n, "expenditure_usd")
    if year_df.empty:
        logger.warning("No SIPRI data for year %d; skipping top-spenders plot", year)
        return

    plt.figure(figsize=(9, 5))
    plt.barh(year_df["country"], year_df["expenditure_usd"] / 1e9)
    plt.xlabel("Military expenditure (US$ billions)")
    plt.title(f"Top {top_n} military spenders — {year}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_expenditure_trend(sipri_df: pd.DataFrame, countries: list[str], out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for c in countries:
        sub = sipri_df[sipri_df["country"] == c].sort_values("year")
        if sub.empty:
            continue
        plt.plot(sub["year"], sub["expenditure_usd"] / 1e9, marker="o", label=c)
    plt.xlabel("Year")
    plt.ylabel("Expenditure (US$ billions)")
    plt.title("Military expenditure trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_gdelt_volume_over_time(gdelt_df: pd.DataFrame, out_path: Path) -> None:
    if gdelt_df.empty or "seendate" not in gdelt_df.columns:
        logger.warning("No GDELT data with dates; skipping volume plot")
        return

    daily = gdelt_df.set_index("seendate").resample("D").size()
    plt.figure(figsize=(9, 5))
    daily.plot()
    plt.xlabel("Date")
    plt.ylabel("Article count")
    plt.title("GDELT article volume over time")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_gdelt_by_query(gdelt_df: pd.DataFrame, out_path: Path) -> None:
    if gdelt_df.empty or "country_query" not in gdelt_df.columns:
        return
    counts = gdelt_df["country_query"].value_counts()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.ylabel("Article count")
    plt.title("GDELT article volume by query")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved %s", out_path)


def write_summary(sipri_df: pd.DataFrame, gdelt_df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# EDA Summary\n"]
    if not sipri_df.empty:
        lines.append(f"- SIPRI: {sipri_df['country'].nunique()} countries, "
                      f"years {int(sipri_df['year'].min())}–{int(sipri_df['year'].max())}\n")
        latest_year = int(sipri_df["year"].max())
        top = sipri_df[sipri_df["year"] == latest_year].nlargest(5, "expenditure_usd")
        lines.append(f"- Top 5 spenders in {latest_year}:\n")
        for _, row in top.iterrows():
            lines.append(f"  - {row['country']}: ${row['expenditure_usd']/1e9:.1f}B\n")
    if not gdelt_df.empty:
        lines.append(f"- GDELT: {len(gdelt_df)} articles across "
                      f"{gdelt_df['country_query'].nunique()} queries\n")
        if "seendate" in gdelt_df.columns:
            lines.append(f"  - Date range: {gdelt_df['seendate'].min()} to {gdelt_df['seendate'].max()}\n")

    out_path.write_text("".join(lines))
    logger.info("Saved EDA summary to %s", out_path)


def run_eda(sipri_df: pd.DataFrame, gdelt_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sipri_df.empty:
        latest_year = int(sipri_df["year"].max())
        plot_top_spenders(sipri_df, latest_year, out_dir / "top_spenders.png")
        top_countries = sipri_df[sipri_df["year"] == latest_year].nlargest(5, "expenditure_usd")["country"].tolist()
        plot_expenditure_trend(sipri_df, top_countries, out_dir / "expenditure_trend.png")

    if not gdelt_df.empty:
        plot_gdelt_volume_over_time(gdelt_df, out_dir / "gdelt_volume.png")
        plot_gdelt_by_query(gdelt_df, out_dir / "gdelt_by_query.png")

    write_summary(sipri_df, gdelt_df, out_dir / "summary.md")
