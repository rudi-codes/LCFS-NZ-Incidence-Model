"""Description:
Create headline figures for distributional incidence.

This script takes inputs produced by `results_tables.py` in a results pack folder (by default`outputs/results_pack/`) and writes figures to the corresponding `figures/` subfolder.

Inputs (expected CSVs under `<pack>/tables/`):
- table_decile_levels_mean.csv
- table_decile_share_eqinc_mean.csv
- table_tenure_4cat_levels_mean.csv
- table_tenure_4cat_share_eqinc_mean.csv
- table_pensioner_household_levels_mean.csv
- table_pensioner_household_share_eqinc_mean.csv

Outputs (PNG + PDF under `<pack>/figures/`):
- fig_decile_levels_mean
- fig_decile_share_eqinc_mean
- fig_tenure_levels_mean
- fig_tenure_share_eqinc_mean
- fig_pensioner_levels_mean
- fig_pensioner_share_eqinc_mean
"""

# region Imports
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# endregion Imports

# region Paths and constants
ROOT = Path(__file__).resolve().parents[1]
# endregion Paths and constants

#-----------------------------
# region Reading and Reshaping functions
#-----------------------------
# Read a decile table and coerce the decile column to sorted integers.
def _read_decile_table(path: Path, decile_col: str = "income_decile_eqincdmp") -> pd.DataFrame:
    df = pd.read_csv(path)
    if decile_col not in df.columns:
        raise ValueError(f"Expected decile column '{decile_col}' in {path.name}")

    # Ensure deciles are numeric and sorted (drop any non numeric rows just in case)
    df[decile_col] = pd.to_numeric(df[decile_col], errors="coerce")
    df = df.dropna(subset=[decile_col]).sort_values(decile_col)
    df[decile_col] = df[decile_col].astype(int)
    return df

# Read a subgroup table and check it contains the expected grouping column.
def _read_group_table(path: Path, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if group_col not in df.columns:
        raise ValueError(f"Expected group column '{group_col}' in {path.name}")
    return df

# Convert a wide scenario table into long format with columns: group, scenario, value.
def _to_long(df: pd.DataFrame, decile_col: str) -> pd.DataFrame:
    scenario_cols: List[str] = [c for c in df.columns if c != decile_col]
    long = df.melt(id_vars=[decile_col], value_vars=scenario_cols, var_name="scenario", value_name="value")
    return long

# Weighted Lorenz curve points for a non-negative variable.
# Returns cumulative population shares p (including 0 and 1) and cumulative value shares L (including 0 and 1).
def _weighted_lorenz_points(x: pd.Series, w: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = x.notna() & w.notna() & (w >= 0)
    if not m.any():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    xx = np.clip(x[m].to_numpy(), 0.0, None)
    ww = w[m].to_numpy()

    order = np.argsort(xx)
    xx = xx[order]
    ww = ww[order]

    W = float(ww.sum())
    Y = float((ww * xx).sum())
    if W <= 0 or Y <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum_w = np.cumsum(ww)
    cum_y = np.cumsum(ww * xx)

    p = np.concatenate([[0.0], cum_w / W])
    L = np.concatenate([[0.0], cum_y / Y])

    p[-1] = 1.0
    L[-1] = 1.0
    return p, L

# Weighted concentration curve points for payments y, ranking by rank_x.
def _weighted_concentration_points(rank_x: pd.Series, y: pd.Series, w: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rx = pd.to_numeric(rank_x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = rx.notna() & y.notna() & w.notna() & (w >= 0)
    if not m.any():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    rr = rx[m].to_numpy()
    yy = np.clip(y[m].to_numpy(), 0.0, None)
    ww = w[m].to_numpy()

    order = np.argsort(rr)
    rr = rr[order]
    yy = yy[order]
    ww = ww[order]

    W = float(ww.sum())
    Y = float((ww * yy).sum())
    if W <= 0 or Y <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum_w = np.cumsum(ww)
    cum_y = np.cumsum(ww * yy)

    p = np.concatenate([[0.0], cum_w / W])
    Cc = np.concatenate([[0.0], cum_y / Y])

    p[-1] = 1.0
    Cc[-1] = 1.0
    return p, Cc

# Plot Lorenz curve for income and concentration curves for liabilities.
def _plot_lorenz_and_concentration(
    *,
    impacts_path: Path,
    hh_analysis_path: Path,
    out_base: Path,
    scenario_prefixes: List[str],
    title: str,
) -> None:
    df_imp = pd.read_csv(impacts_path)
    df_hh = pd.read_csv(hh_analysis_path)

    # Income ranking (annualised equivalised disposable income)
    inc_weekly_col = "income_equivalised_weekly_eqincdmp"
    if inc_weekly_col not in df_hh.columns:
        raise ValueError(f"Expected `{inc_weekly_col}` in {hh_analysis_path.name}")
    if "case" not in df_hh.columns:
        raise ValueError("Expected `case` in household analysis file")

    weeks_per_year = 365.0 / 7.0
    df_hh = df_hh[["case", inc_weekly_col]].copy()
    df_hh["eqinc_annual"] = pd.to_numeric(df_hh[inc_weekly_col], errors="coerce") * weeks_per_year

    if "case" not in df_imp.columns:
        raise ValueError("Expected `case` in impacts file")
    if "scenario" not in df_imp.columns:
        raise ValueError("Expected `scenario` in impacts file")
    if "weight_annual" not in df_imp.columns:
        raise ValueError("Expected `weight_annual` in impacts file")
    if "liability_gbp_annual" not in df_imp.columns:
        raise ValueError("Expected `liability_gbp_annual` in impacts file")

    dfm = df_imp.merge(df_hh[["case", "eqinc_annual"]], on="case", how="left")

    # Lorenz curve for income: take a unique household slice to avoid duplicated cases across scenarios
    df_base = dfm[dfm["scenario"].astype(str).str.startswith("A_")].copy()
    if df_base.empty:
        df_base = dfm.drop_duplicates(subset=["case"]).copy()

    pL, LL = _weighted_lorenz_points(df_base["eqinc_annual"], df_base["weight_annual"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # 45-degree equality line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Equality (45°)")
    ax.plot(pL, LL, marker=None, linewidth=2.0, label="Lorenz: equivalised income")

    # Concentration curves for selected scenarios
    for pref in scenario_prefixes:
        sub = dfm[dfm["scenario"].astype(str).str.startswith(f"{pref}_")].copy()
        if sub.empty:
            continue
        pC, CC = _weighted_concentration_points(sub["eqinc_annual"], sub["liability_gbp_annual"], sub["weight_annual"])
        ax.plot(pC, CC, linewidth=1.8, label=f"Concentration: liabilities ({pref})")

    ax.set_title(title)
    ax.set_xlabel("Cumulative population share (ranked by equivalised income)")
    ax.set_ylabel("Cumulative share")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
# endregion Helper functions: reading and reshaping tables

#-----------------------------
# region Plotting functions
#-----------------------------
# Plot grouped bar charts from long-data and save as PNG and PDF. Grouping by distributional variable.
def _plot_grouped_bars(
    long: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    group_order: List[str] | None = None,
) -> None:
    # Tweak distributional variable order (alphabetical by default)
    if group_order is not None:
        long[group_col] = pd.Categorical(long[group_col].astype(str), categories=group_order, ordered=True)
        long = long.sort_values([group_col, "scenario"])
    else:
        long[group_col] = long[group_col].astype(str)

    groups = long[group_col].unique().tolist()
    scenarios = long["scenario"].unique().tolist()

    x = list(range(len(groups)))
    n_s = max(len(scenarios), 1)
    width = 0.8 / n_s

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, scen in enumerate(scenarios):
        sub = long[long["scenario"] == scen]
        # Align values to group order
        vals = (
            sub.set_index(group_col)["value"]
            .reindex(groups)
            .astype(float)
            .to_numpy()
        )
        xpos = [xx + (i - (n_s - 1) / 2) * width for xx in x]
        ax.bar(xpos, vals, width=width, label=scen)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Scenario", fontsize=8, title_fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

# Plot line charts from long-data and save as PNG and PDF. Grouping by scenario.
def _plot_lines(long: pd.DataFrame, *, x: str, y: str, title: str, xlabel: str, ylabel: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for scen, sub in long.groupby("scenario"):
        ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, label=scen)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(long[x].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Scenario", fontsize=8, title_fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

# endregion Plotting functions

#-----------------------------
# region Headline Figures
#-----------------------------
'''
Main function to generate all headline figures for a given results pack (optionally a sensitivity tag).
Headline figure 1: mean £ impact by decile
Headline figure 2: mean impact as % of equivalised income by decile
'''

def main(*, tag: Optional[str] = None) -> None:
    if tag:
        pack_dir = ROOT / "outputs" / "results_pack_sensitivities" / tag
    else:
        pack_dir = ROOT / "outputs" / "results_pack"

    table_dir = pack_dir / "tables"
    fig_dir = pack_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Headline figure 1: mean £ impact by decile
    levels_path = table_dir / "table_decile_levels_mean.csv"
    levels = _read_decile_table(levels_path)
    levels_long = _to_long(levels, "income_decile_eqincdmp")

    _plot_lines(
        levels_long,
        x="income_decile_eqincdmp",
        y="value",
        title="Mean annual liability by equivalised income decile",
        xlabel="Equivalised income decile (1 = lowest)",
        ylabel="Mean annual liability (£)",
        out_base=fig_dir / "fig_decile_levels_mean",
    )

    # Headline figure 2: mean impact as % of equivalised income by decile
    share_path = table_dir / "table_decile_share_eqinc_mean.csv"
    share = _read_decile_table(share_path)
    share_long = _to_long(share, "income_decile_eqincdmp")
    share_long["value"] = 100.0 * share_long["value"]

# endregion Headline Figures

    _plot_lines(
        share_long,
        x="income_decile_eqincdmp",
        y="value",
        title="Mean liability as a share of equivalised income by decile",
        xlabel="Equivalised income decile (1 = lowest)",
        ylabel="Mean liability (% of equivalised income)",
        out_base=fig_dir / "fig_decile_share_eqinc_mean",
    )

    # Progressivity figure: Lorenz curve (income) and concentration curves (payments)
    impacts_path = ROOT / "outputs" / "impacts_scenarios_smoketest.csv"
    hh_analysis_path = ROOT / "outputs" / "lcfs_2022_23_HH_analysis.csv"

    _plot_lorenz_and_concentration(
        impacts_path=impacts_path,
        hh_analysis_path=hh_analysis_path,
        out_base=fig_dir / "fig_lorenz_concentration",
        scenario_prefixes=["A", "C", "E3", "E2"],
        title="Lorenz curve of income and concentration curves of liabilities",
    )
# endregion Headline Figures

    # ------------------------------------------------------------------
    # region Subgroup headline figures (headline): tenure and pensioner households
    # ------------------------------------------------------------------

    # Tenure (mean £ levels)
    tenure_levels_path = table_dir / "table_tenure_4cat_levels_mean.csv"
    tenure_levels = _read_group_table(tenure_levels_path, "tenure_4cat")
    tenure_levels_long = _to_long(tenure_levels, "tenure_4cat")

    tenure_order = ["Owner occupier", "Private rent", "Social rent", "Rent free"]
    _plot_grouped_bars(
        tenure_levels_long,
        group_col="tenure_4cat",
        title="Mean annual liability by tenure",
        xlabel="Tenure",
        ylabel="Mean annual liability (£)",
        out_base=fig_dir / "fig_tenure_levels_mean",
        group_order=tenure_order,
    )

    # Tenure (mean % of equivalised income)
    tenure_share_path = table_dir / "table_tenure_4cat_share_eqinc_mean.csv"
    tenure_share = _read_group_table(tenure_share_path, "tenure_4cat")
    tenure_share_long = _to_long(tenure_share, "tenure_4cat")
    tenure_share_long["value"] = 100.0 * tenure_share_long["value"]

    _plot_grouped_bars(
        tenure_share_long,
        group_col="tenure_4cat",
        title="Mean liability as a share of equivalised income by tenure",
        xlabel="Tenure",
        ylabel="Mean liability (% of equivalised income)",
        out_base=fig_dir / "fig_tenure_share_eqinc_mean",
        group_order=tenure_order,
    )

    # Pensioner household (mean £ levels)
    pens_levels_path = table_dir / "table_pensioner_household_levels_mean.csv"
    pens_levels = _read_group_table(pens_levels_path, "pensioner_household")
    pens_levels_long = _to_long(pens_levels, "pensioner_household")

    _plot_grouped_bars(
        pens_levels_long,
        group_col="pensioner_household",
        title="Mean annual liability by pensioner household status",
        xlabel="Pensioner household (0 = no, 1 = yes)",
        ylabel="Mean annual liability (£)",
        out_base=fig_dir / "fig_pensioner_levels_mean",
    )

    # Pensioner household (mean % of equivalised income)
    pens_share_path = table_dir / "table_pensioner_household_share_eqinc_mean.csv"
    pens_share = _read_group_table(pens_share_path, "pensioner_household")
    pens_share_long = _to_long(pens_share, "pensioner_household")
    pens_share_long["value"] = 100.0 * pens_share_long["value"]

    _plot_grouped_bars(
        pens_share_long,
        group_col="pensioner_household",
        title="Mean liability as a share of equivalised income by pensioner household status",
        xlabel="Pensioner household (0 = no, 1 = yes)",
        ylabel="Mean liability (% of equivalised income)",
        out_base=fig_dir / "fig_pensioner_share_eqinc_mean",
    )

    print("Wrote figures to:", fig_dir)
    for fname in [
        "fig_decile_levels_mean.png",
        "fig_decile_share_eqinc_mean.png",
        "fig_lorenz_concentration.png",
        "fig_tenure_levels_mean.png",
        "fig_tenure_share_eqinc_mean.png",
        "fig_pensioner_levels_mean.png",
        "fig_pensioner_share_eqinc_mean.png",
    ]:
        print("-", (fig_dir / fname))

    # endregion Subgroup headline figures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate headline figures from results tables.")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional sensitivity tag; reads/writes under outputs/results_pack_sensitivities/<tag>/",
    )
    args = parser.parse_args()

    main(tag=args.tag)