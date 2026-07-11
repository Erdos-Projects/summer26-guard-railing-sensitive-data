"""Build portfolio-ready privacy/utility reports from experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReportOutputs:
    report_path: Path
    frontier_path: Path
    plot_path: Path | None


def _read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.1f}"
        if abs(value) >= 10:
            return f"{value:,.2f}"
        return f"{value:,.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"

    data = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows).copy()
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _pareto_efficient(frame: pd.DataFrame, risk_col: str, utility_col: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="bool")
    efficient = []
    values = frame[[risk_col, utility_col]].to_numpy(dtype=float)
    for index, point in enumerate(values):
        if np.isnan(point).any():
            efficient.append(False)
            continue
        others = np.delete(values, index, axis=0)
        dominates = (
            (others[:, 0] <= point[0])
            & (others[:, 1] <= point[1])
            & ((others[:, 0] < point[0]) | (others[:, 1] < point[1]))
        )
        efficient.append(bool(~dominates.any()))
    return pd.Series(efficient, index=frame.index)


def build_frontier_table(
    k_summary: pd.DataFrame,
    linkage_summary: pd.DataFrame,
    utility_summary: pd.DataFrame,
    *,
    n_known: int = 3,
) -> pd.DataFrame:
    """Merge privacy and utility summaries into one frontier table."""

    if linkage_summary.empty or utility_summary.empty:
        return pd.DataFrame()

    privacy = linkage_summary[linkage_summary["n_known"] == n_known].copy()
    privacy = privacy.rename(
        columns={
            "pct_unique": f"linkage_unique_pct_at_{n_known}_facts",
            "median_candidate_size": f"median_candidates_at_{n_known}_facts",
            "avg_candidate_size": f"avg_candidates_at_{n_known}_facts",
        }
    )
    keep_privacy = [
        "release_name",
        f"linkage_unique_pct_at_{n_known}_facts",
        f"median_candidates_at_{n_known}_facts",
        f"avg_candidates_at_{n_known}_facts",
    ]
    privacy = privacy[[column for column in keep_privacy if column in privacy.columns]]

    utility = utility_summary[utility_summary["status"] == "ok"].copy()
    keep_utility = [
        "release_name",
        "rmse",
        "mae",
        "hit_rate_at_k",
        "ndcg_at_k",
        "ranking_users",
    ]
    utility = utility[[column for column in keep_utility if column in utility.columns]]

    frontier = privacy.merge(utility, on="release_name", how="inner")
    if not k_summary.empty:
        k_keep = ["release_name", "rows_released", "min_k", "median_k", "pct_unique_facts"]
        frontier = frontier.merge(k_summary[[column for column in k_keep if column in k_summary.columns]], on="release_name", how="left")

    baseline = frontier.loc[frontier["release_name"] == "original_movie_rating_month", "rmse"]
    baseline_rmse = float(baseline.iloc[0]) if not baseline.empty else float(frontier["rmse"].min())
    frontier["rmse_delta_vs_original"] = frontier["rmse"] - baseline_rmse
    frontier["utility_loss_pct_vs_original"] = frontier["rmse_delta_vs_original"] / baseline_rmse * 100
    risk_col = f"linkage_unique_pct_at_{n_known}_facts"
    frontier["pareto_frontier"] = _pareto_efficient(frontier, risk_col, "rmse")
    return frontier.sort_values([risk_col, "rmse"], na_position="last").reset_index(drop=True)


def write_frontier_plot(frontier: pd.DataFrame, path: Path, *, n_known: int = 3) -> Path | None:
    """Write a privacy/utility scatter plot when matplotlib is available."""

    if frontier.empty:
        return None

    try:
        cache_dir = Path(tempfile.gettempdir()) / "guardrails_sensitive_data_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    risk_col = f"linkage_unique_pct_at_{n_known}_facts"
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    colors = np.where(frontier["pareto_frontier"], "#0f766e", "#64748b")
    ax.scatter(frontier["rmse"], frontier[risk_col], s=70, c=colors, edgecolor="white", linewidth=0.8)
    for row in frontier.itertuples(index=False):
        label = str(row.release_name).replace("_", " ")
        ax.annotate(label, (float(row.rmse), float(getattr(row, risk_col))), fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("RMSE on held-out true ratings")
    ax.set_ylabel(f"Unique linkage rate with {n_known} known facts (%)")
    ax.set_title("Privacy-Utility Frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _headline_findings(
    frontier: pd.DataFrame,
    planted_summary: pd.DataFrame,
    *,
    n_known: int,
) -> list[str]:
    findings: list[str] = []
    if not frontier.empty:
        risk_col = f"linkage_unique_pct_at_{n_known}_facts"
        original = frontier[frontier["release_name"] == "original_movie_rating_month"]
        if not original.empty:
            row = original.iloc[0]
            findings.append(
                f"Original-style movie/rating/month facts uniquely identify {row[risk_col]:.1f}% "
                f"of sampled users with {n_known} known facts."
            )
        safest = frontier.sort_values([risk_col, "rmse"]).iloc[0]
        findings.append(
            f"The lowest-risk evaluated release is `{safest['release_name']}` "
            f"({safest[risk_col]:.1f}% unique; RMSE {safest['rmse']:.3f})."
        )
        best_utility = frontier.sort_values("rmse").iloc[0]
        findings.append(
            f"The strongest bias-model utility is `{best_utility['release_name']}` "
            f"(RMSE {best_utility['rmse']:.3f})."
        )
        pareto = frontier[frontier["pareto_frontier"]]["release_name"].tolist()
        if pareto:
            findings.append("Pareto frontier releases: " + ", ".join(f"`{name}`" for name in pareto) + ".")
    if not planted_summary.empty:
        row = planted_summary.iloc[0]
        findings.append(
            f"The planted synthetic attack recovered source users at rank 1 in "
            f"{row['top_1_rate']:.1f}% of profiles and top 5 in {row['top_5_rate']:.1f}%."
        )
    return findings


def build_report(
    output_dir: Path,
    *,
    prefix: str = "",
    n_known: int = 3,
    title: str = "Netflix Prize Privacy-Utility Audit",
) -> ReportOutputs:
    """Build a Markdown report, frontier CSV, and optional frontier plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_summary = _read_csv_optional(output_dir / f"{prefix}privacy_k_anonymity_summary.csv")
    linkage_summary = _read_csv_optional(output_dir / f"{prefix}privacy_linkage_summary.csv")
    utility_summary = _read_csv_optional(output_dir / f"{prefix}utility_rmse_summary.csv")
    planted_summary = _read_csv_optional(output_dir / f"{prefix}planted_attack_summary.csv")

    frontier = build_frontier_table(k_summary, linkage_summary, utility_summary, n_known=n_known)
    frontier_path = output_dir / f"{prefix}privacy_utility_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    plot_path = write_frontier_plot(frontier, output_dir / f"{prefix}privacy_utility_frontier.png", n_known=n_known)

    findings = _headline_findings(frontier, planted_summary, n_known=n_known)
    report_path = output_dir / f"{prefix}privacy_utility_report.md"
    lines = [
        f"# {title}",
        "",
        "This report summarizes the public-safe outputs produced by the privacy audit CLI.",
        "",
        "## Headline Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings or ["No complete frontier data was available."])
    lines.extend(
        [
            "",
            "## Privacy-Utility Frontier",
            "",
            _markdown_table(
                frontier,
                [
                    "release_name",
                    f"linkage_unique_pct_at_{n_known}_facts",
                    f"median_candidates_at_{n_known}_facts",
                    "rmse",
                    "mae",
                    "hit_rate_at_k",
                    "ndcg_at_k",
                    "utility_loss_pct_vs_original",
                    "pareto_frontier",
                ],
            ),
            "",
            "## k-Anonymity Summary",
            "",
            _markdown_table(
                k_summary.sort_values("min_k", ascending=False) if not k_summary.empty else k_summary,
                ["release_name", "rows_released", "min_k", "median_k", "pct_unique_facts"],
            ),
            "",
            "## Planted Synthetic Attack",
            "",
            _markdown_table(planted_summary, ["n_profiles", "n_known", "mean_rank", "top_1_rate", "top_5_rate", "top_10_rate"]),
            "",
            "## Artifacts",
            "",
            f"- Frontier CSV: `{frontier_path.name}`",
        ]
    )
    if plot_path is not None:
        lines.append(f"- Frontier plot: `{plot_path.name}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReportOutputs(report_path=report_path, frontier_path=frontier_path, plot_path=plot_path)
