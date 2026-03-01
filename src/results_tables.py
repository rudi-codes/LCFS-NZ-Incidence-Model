"""Description:
Produce core distributional results tables from scenario impacts.
This script takes inputs from the datasets produced by `run_incidence.py` (`outputs/impacts_scenarios_smoketest.csv`) and writes CSV tables for inspection.
Outputs are written to `outputs/results_pack/`.
Notes
- All summaries are weighted using `weight_annual`.
- Deciles are based on `income_decile_eqincdmp`.
- Medians are computed as weighted quantiles (p50) using the survey weights.
"""
#region Package Imports
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import argparse
import numpy as np
import pandas as pd
#endregion Package Imports

# region Paths and constants
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPACTS_PATH = ROOT / "outputs" / "impacts_scenarios_smoketest.csv"
# Central location for LaTeX tables that are actually included in the report
TEX_TABLES_DIR = ROOT / "outputs" / "tex_tables"
BENCHMARKS_ENERGY_BY_DECILE = ROOT / "outputs" / "benchmarks_energy_by_decile.csv"
SCENARIO_CHECKS_JSON = ROOT / "outputs" / "checks_scenarios_smoketest.json"
IMPLIED_BILL_PARAMS_TEX = "tab_implied_bill_params.tex"
TAU_TO_PKWH_TEX = "tab_tau_to_pkwh.tex"
ATKINSON_WELFARE_TEX = "tab_welfare_atkinson.tex"
SENS_S1_IMPACTS = ROOT / "outputs" / "sensitivities" / "impacts_sensitivity_S1_passthrough80_A_D.csv"
SENS_S2_IMPACTS = ROOT / "outputs" / "sensitivities" / "impacts_sensitivity_S2_eqincfloor5000_A_D_E1_E2.csv"
SENS_S3_IMPACTS = ROOT / "outputs" / "sensitivities" / "impacts_sensitivity_S3_standingcharge_allhouseholds_B_D.csv"
SENS_S4_IMPACTS = ROOT / "outputs" / "sensitivities" / "impacts_sensitivity_S4_revenuescale_R100m_A_D_E1_E2.csv"
SENS_S5_IMPACTS = ROOT / "outputs" / "sensitivities" / "impacts_sensitivity_S5_standingcharge_pm20_A_C.csv"
HH_ANALYSIS_PATH = ROOT / "outputs" / "lcfs_2022_23_HH_analysis.csv"
# region Paths and constants
HH_DIAGNOSTICS_PATH = ROOT / "outputs" / "lcfs_2022_23_HH_diagnostics.csv"
RURALITY_EQINC_CSV = ROOT / "outputs" / "results_pack" / "tables" / "secondary" / "table_rurality_label_share_eqinc_mean.csv"
# endregion Paths and constants

# -----------------------------------------------------------------------------
# region Weighted stats helper functions
# -----------------------------------------------------------------------------

# Convert to numeric
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# Weighted mean and quantiles using survey weights to represent population
def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = _to_num(x)
    w = _to_num(w)
    m = x.notna() & w.notna()
    if not m.any():
        return float("nan")
    denom = float(w[m].sum())
    if denom == 0.0:
        return float("nan")
    return float((x[m] * w[m]).sum() / denom)

"""Weighted quantile using cumulative weights, needed to represent 'true' population.
Parameters
----------
x, w: Values and non negative weights.
q: Quantile in [0, 1].
interpolate: If True, linearly interpolate between adjacent points when the cumulative weight crosses the target.
"""
def weighted_quantile(
    x: pd.Series,
    w: pd.Series,
    q: float,
    *,
    interpolate: bool = True,
) -> float:

    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")

    x = _to_num(x)
    w = _to_num(w)
    m = x.notna() & w.notna() & (w >= 0)
    if not m.any():
        return float("nan")

    xx = x[m].to_numpy()
    ww = w[m].to_numpy()

    order = np.argsort(xx)
    xx = xx[order]
    ww = ww[order]

    cum_w = np.cumsum(ww)
    total_w = float(cum_w[-1])
    if total_w == 0.0:
        return float("nan")

    target = q * total_w

    idx = int(np.searchsorted(cum_w, target, side="left"))
    idx = min(max(idx, 0), len(xx) - 1)

    if not interpolate or idx == 0:
        return float(xx[idx])

    w_lo = float(cum_w[idx - 1])
    w_hi = float(cum_w[idx])
    x_lo = float(xx[idx - 1])
    x_hi = float(xx[idx])

    if w_hi == w_lo:
        return float(x_hi)

    alpha = (target - w_lo) / (w_hi - w_lo)
    return float(x_lo + alpha * (x_hi - x_lo))
# endregion Weighted stats helper functions

# Weighted Lorenz curve points for a non-negative variable.
# Returns cumulative population shares p (including 0 and 1) and cumulative value shares L (including 0 and 1).
def weighted_lorenz_points(x: pd.Series, w: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = _to_num(x)
    w = _to_num(w)
    m = x.notna() & w.notna() & (w >= 0)
    if not m.any():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    xx = x[m].to_numpy()
    ww = w[m].to_numpy()

    # Enforce non-negativity for Lorenz objects
    xx = np.clip(xx, 0.0, None)

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

    # Guard numerical drift
    p[-1] = 1.0
    L[-1] = 1.0
    return p, L

# Weighted Gini coefficient computed from Lorenz points.
def weighted_gini(x: pd.Series, w: pd.Series) -> float:
    p, L = weighted_lorenz_points(x, w)
    # Trapezoid area under Lorenz curve
    A = float(np.sum((p[1:] - p[:-1]) * (L[1:] + L[:-1]) / 2.0))
    G = 1.0 - 2.0 * A
    return float(max(min(G, 1.0), 0.0))

# Weighted concentration coefficient for payments y, ranking by income rank_x.
# Computes concentration curve C(p) and returns the concentration coefficient.
def weighted_concentration(rank_x: pd.Series, y: pd.Series, w: pd.Series) -> tuple[float, np.ndarray, np.ndarray]:
    rx = _to_num(rank_x)
    y = _to_num(y)
    w = _to_num(w)
    m = rx.notna() & y.notna() & w.notna() & (w >= 0)
    if not m.any():
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0])

    rr = rx[m].to_numpy()
    yy = y[m].to_numpy()
    ww = w[m].to_numpy()

    # Concentration objects require non-negative payments
    yy = np.clip(yy, 0.0, None)

    order = np.argsort(rr)
    rr = rr[order]
    yy = yy[order]
    ww = ww[order]

    W = float(ww.sum())
    Y = float((ww * yy).sum())
    if W <= 0 or Y <= 0:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum_w = np.cumsum(ww)
    cum_y = np.cumsum(ww * yy)

    p = np.concatenate([[0.0], cum_w / W])
    Cc = np.concatenate([[0.0], cum_y / Y])

    p[-1] = 1.0
    Cc[-1] = 1.0

    A = float(np.sum((p[1:] - p[:-1]) * (Cc[1:] + Cc[:-1]) / 2.0))
    conc = 1.0 - 2.0 * A
    return float(conc), p, Cc

# Weighted mean for strictly positive variables with an enforced floor
def _enforce_floor(x: pd.Series, floor: float) -> pd.Series:
    xx = _to_num(x)
    return xx.clip(lower=float(floor))

# Weighted equally-distributed equivalent income (EDE) for Atkinson welfare.
# eps = inequality aversion parameter. Requires strictly positive incomes.
def weighted_ede_atkinson(y: pd.Series, w: pd.Series, eps: float, *, floor: float) -> float:
    yy = _enforce_floor(y, floor)
    ww = _to_num(w)
    m = yy.notna() & ww.notna() & (ww >= 0)
    if not m.any():
        return float("nan")

    yy = yy[m].to_numpy(dtype=float)
    ww = ww[m].to_numpy(dtype=float)
    W = float(ww.sum())
    if W <= 0:
        return float("nan")

    eps = float(eps)
    if eps == 0.0:
        # No inequality aversion: EDE equals the mean
        return float((ww * yy).sum() / W)

    if eps == 1.0:
        # Log utility case
        return float(np.exp((ww * np.log(yy)).sum() / W))

    # General case
    p = 1.0 - eps
    return float(((ww * (yy ** p)).sum() / W) ** (1.0 / p))

# Weighted Atkinson index A(eps) = 1 - EDE/mean
def weighted_atkinson_index(y: pd.Series, w: pd.Series, eps: float, *, floor: float) -> float:
    mu = weighted_mean(_enforce_floor(y, floor), w)
    ede = weighted_ede_atkinson(y, w, eps, floor=floor)
    if not (mu == mu and ede == ede and mu != 0.0):
        return float("nan")
    return float(1.0 - (ede / mu))

# -----------------------------------------------------------------------------
# region Table build functions
# -----------------------------------------------------------------------------

# Summarise by scenario and group, computing weighted means and quantiles
def summarise_by_scenario_and_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    scenario_col: str = "scenario",
    weight_col: str = "weight_annual",
    quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:

    if quantiles is None:
        quantiles = [0.5] # Default to weighted median unless specified

    required = {scenario_col, group_col, value_col, weight_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_rows = []
    for (scen, grp), sub in df.groupby([scenario_col, group_col], dropna=False):
        w = sub[weight_col]
        x = sub[value_col]
        row = {
            "scenario": scen,
            group_col: grp,
            "mean": weighted_mean(x, w),
        }
        for q in quantiles:
            key = f"p{int(round(q * 100))}"
            row[key] = weighted_quantile(x, w, q)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    return out.sort_values(["scenario", group_col]).reset_index(drop=True)

# Pivot a tidy scenario x group table into group rows with scenario columns
def pivot_scenario_table(
    tidy: pd.DataFrame,
    group_col: str,
    metric_col: str,
    *,
    scenario_col: str = "scenario",
) -> pd.DataFrame:
    pt = tidy.pivot(index=group_col, columns=scenario_col, values=metric_col)
    pt = pt.reset_index()
    pt.columns.name = None
    return pt
# endregion Table build functions


# -----------------------------------------------------------------------------
# region LaTeX export helpers
# -----------------------------------------------------------------------------

def _fmt_int(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):,}"

def _fmt_3(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"

def write_baseline_energy_benchmarks_tex(
    *,
    csv_path: Path,
    out_dir: Path,
    out_name: str = "tab_benchmarks_energy_by_decile.tex",
) -> Path:
    """Write a compact LaTeX table for baseline energy benchmarks by equivalised income decile.

    Reads `outputs/benchmarks_energy_by_decile.csv` and produces a self-contained LaTeX table
    (with caption/label) suitable for `\input{...}` in the report.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dfb = pd.read_csv(csv_path)

    required_cols = [
        "income_decile_eqincdmp",
        "mean_gas_gbp_annual",
        "mean_elec_gbp_annual",
        "mean_energy_gbp_annual",
        "mean_energy_share_totalexp",
        "share_zero_gas",
        "share_zero_elec",
        "n_unweighted",
    ]
    missing = [c for c in required_cols if c not in dfb.columns]
    if missing:
        raise ValueError(f"Benchmark file missing required columns: {missing}")

    # Keep deciles 1-10 plus ALL (at end)
    df_num = dfb[dfb["income_decile_eqincdmp"].astype(str).str.fullmatch(r"\d+")].copy()
    df_num["income_decile_eqincdmp"] = pd.to_numeric(df_num["income_decile_eqincdmp"], errors="coerce")
    df_num = df_num.sort_values("income_decile_eqincdmp")

    df_all = dfb[dfb["income_decile_eqincdmp"].astype(str).str.upper().eq("ALL")].copy()

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Decile} & \textbf{Gas (\pounds)} & \textbf{Elec (\pounds)} & \textbf{Total (\pounds)} & \textbf{Share of exp} & \textbf{Zero gas} & \textbf{Zero elec} & \textbf{N (unw.)} \\"
    )
    lines.append(r"\midrule")

    for _, row in df_num.iterrows():
        d = int(row["income_decile_eqincdmp"]) if not pd.isna(row["income_decile_eqincdmp"]) else None
        gas = _fmt_int(row["mean_gas_gbp_annual"])
        elec = _fmt_int(row["mean_elec_gbp_annual"])
        tot = _fmt_int(row["mean_energy_gbp_annual"])
        sh = _fmt_3(row["mean_energy_share_totalexp"])
        zg = _fmt_3(row["share_zero_gas"])
        ze = _fmt_3(row["share_zero_elec"])
        n = _fmt_int(row["n_unweighted"])
        lines.append(f"{d} & {gas} & {elec} & {tot} & {sh} & {zg} & {ze} & {n} \\\\ ")

    if not df_all.empty:
        row = df_all.iloc[0]
        gas = _fmt_int(row["mean_gas_gbp_annual"])
        elec = _fmt_int(row["mean_elec_gbp_annual"])
        tot = _fmt_int(row["mean_energy_gbp_annual"])
        sh = _fmt_3(row["mean_energy_share_totalexp"])
        zg = _fmt_3(row["share_zero_gas"])
        ze = _fmt_3(row["share_zero_elec"])
        n = _fmt_int(row["n_unweighted"])
        lines.append(r"\midrule")
        lines.append(f"\\textbf{{All}} & {gas} & {elec} & {tot} & {sh} & {zg} & {ze} & {n} \\\\ ")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Baseline annual energy spending benchmarks by equivalised income decile. Means are weighted; N is unweighted. Zero shares report the weighted share with zero recorded expenditure for the relevant fuel.}"
    )
    lines.append(r"\label{tab:baseline-energy-benchmarks-decile}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# New function: write_scenario_calibration_diagnostics_tex
def write_scenario_calibration_diagnostics_tex(
    *,
    json_path: Path,
    out_dir: Path,
    out_name: str = "tab_scenario_calibration_diagnostics.tex",
) -> Path:
    """Write a compact LaTeX table summarising scenario calibration diagnostics (A--E3).

    Expects a JSON mapping scenario -> diagnostics as produced by the incidence QA checks.
    Produces a self-contained LaTeX table (with caption/label) suitable for `\input{...}`.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario checks JSON not found: {json_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    d = json.loads(json_path.read_text(encoding="utf-8"))
    scenarios = ["A", "B", "C", "D", "E1", "E2", "E3"]

    def _get_param(scen: str, rec: dict) -> tuple[str, float | None]:
        if "tau" in rec:
            return (r"$\tau$", float(rec["tau"]))
        if scen in ("B", "D") and "s" in rec:
            return (r"$s$", float(rec["s"]))
        if scen == "E1" and "u" in rec:
            return (r"$u$", float(rec["u"]))
        if scen == "E2" and "kappa" in rec:
            return (r"$\kappa$", float(rec["kappa"]))
        if scen == "E3" and "rho" in rec:
            return (r"$\rho$", float(rec["rho"]))
        return ("", None)

    def _fmt_money(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{float(x):,.0f}"

    def _fmt_pct(x: float, ndp: int = 2) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.{ndp}f}\\%"

    def _payer_cell(x: float) -> str:
        # If payer share is not provided (unrestricted base), treat as 100%.
        return _fmt_pct(1.0, 2) if not (x == x) else _fmt_pct(x, 2)

    def _fmt_float(x: float, ndp: int) -> str:
        if pd.isna(x):
            return ""
        return f"{float(x):.{ndp}f}"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Scenario} & \textbf{Parameter} & \textbf{Revenue (\pounds)} & \textbf{Gap (\%)} & \textbf{p99 (\pounds)} & \textbf{Payer share} \\"
    )
    lines.append(r"\midrule")

    for scen in scenarios:
        rec = d.get(scen, {})
        label, val = _get_param(scen, rec)

        if val is None:
            param = ""
        else:
            if label in (r"$\tau$", r"$\kappa$", r"$\rho$"):
                # report as a percentage rate
                param = f"{label} = {_fmt_pct(val, 3)}"
            else:
                # lump-sum amounts in £ per year
                param = f"{label} = \pounds\,{_fmt_float(val, 2)}"

        rev_t = float(rec.get("revenue_target", float("nan")))
        rev_i = float(rec.get("revenue_implied", float("nan")))
        gap = float(rec.get("revenue_gap", float("nan")))
        gap_pct = gap / rev_t if (rev_t == rev_t and rev_t != 0.0) else float("nan")
        p99 = float(rec.get("liability_p99", float("nan")))

        payer_share = float("nan")
        if "payer_share_weighted" in rec:
            payer_share = float(rec["payer_share_weighted"])
        elif "tax_payer_share_weighted" in rec:
            payer_share = float(rec["tax_payer_share_weighted"])

        payer_cell = _payer_cell(payer_share)

        lines.append(
            f"{scen} & {param} & {_fmt_money(rev_i)} & {_fmt_pct(gap_pct, 3)} & {_fmt_money(p99)} & {payer_cell} \\\\ "
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Scenario calibration diagnostics (A--E3). The parameter is the calibrated unit levy rate ($\tau$), annual standing charge ($s$), uniform rebate ($u$), income tax rate ($\kappa$), or proportional income levy rate ($\rho$), depending on scenario. Revenues are computed using survey weights; the gap is the percentage deviation from the common revenue requirement $R$; p99 reports the 99th percentile of annual household liabilities.}"
    )
    lines.append(r"\label{tab:scenario-calibration-diagnostics}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

def write_sensitivity_summary_tex(
    *,
    impacts_paths: Dict[str, Path],
    out_dir: Path,
    out_name: str = "tab_sensitivity_summary.tex",
) -> Path:
    """Write a compact LaTeX summary table for sensitivity runs and baseline scenarios.

    Reports two headline metrics (weighted mean liability share of equivalised income) for
    decile 1 and decile 10, using one representative scenario per sensitivity run, and baseline headline rows for scenarios A, B, C, D.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "S1": {
            "change": "Partial pass-through (80\\%)",
            "expect": "Cash liabilities scale down; profiles invariant",
            "scenario_prefixes": ["A_", "C_"],
        },
        "S2": {
            "change": "Higher income floor (\\pounds 5,000)",
            "expect": "Normalised impacts less sensitive to near-zero incomes",
            "scenario_prefixes": ["A_", "C_"],
        },
        "S3": {
            "change": "Standing charge base expanded to all households",
            "expect": "Per-household charge falls; payer share becomes 100\\%",
            "scenario_prefixes": ["B_", "D_"],
        },
        "S4": {
            "change": "Revenue scaled to \\pounds 100m",
            "expect": "Impacts scale approximately linearly with R",
            "scenario_prefixes": ["A_", "C_"],
        },
        "S5": {
            "change": "Standing charge proxy +20\\% (unit levy base netting)",
            "expect": "Higher SC reduces the net base; $\\tau$ rises; distributional profiles should be stable",
            "scenario_prefixes": ["A_", "C_"],
        },
    }

    def _pick_scenario(df: pd.DataFrame, prefix: str) -> str:
        opts = sorted(df.loc[df["scenario"].astype(str).str.startswith(prefix), "scenario"].unique())
        if not opts:
            raise ValueError(f"No scenario found with prefix {prefix}")
        return opts[0]

    def _fmt_pct_points(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.3f}\\%"

    def _fmt_ratio(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{float(x):.2f}"

    def _pick_baseline_scenario(df: pd.DataFrame, prefix: str) -> str:
        opts = sorted(
            df.loc[
                df["scenario"].astype(str).str.startswith(prefix)
                & (~df["scenario"].astype(str).str.contains(r"_S\\d+_", regex=True)),
                "scenario",
            ].unique()
        )
        if not opts:
            # Fallback: if no unsuffixed scenario found, fall back to the first matching scenario
            return _pick_scenario(df, prefix)
        return opts[0]

    rows = []

    # Baseline headline rows (A--D) for context
    baseline_path = impacts_paths.get("BASELINE", DEFAULT_IMPACTS_PATH)
    if baseline_path is not None and Path(baseline_path).exists():
        df_base = pd.read_csv(baseline_path)
        for pref in ["A_", "B_", "C_", "D_"]:
            scen_name = _pick_baseline_scenario(df_base, pref)
            dfi = df_base[df_base["scenario"].astype(str).eq(scen_name)].copy()

            d1 = dfi[dfi["income_decile_eqincdmp"].astype(int).eq(1)]
            d10 = dfi[dfi["income_decile_eqincdmp"].astype(int).eq(10)]

            m1 = weighted_mean(d1["liability_share_eqinc"], d1["weight_annual"]) if len(d1) else float("nan")
            m10 = weighted_mean(d10["liability_share_eqinc"], d10["weight_annual"]) if len(d10) else float("nan")
            ratio = (m1 / m10) if (m10 == m10 and m10 != 0.0) else float("nan")

            scen_short = scen_name.split("_")[0]

            rows.append({
                "sens": "Central",
                "change": "N/A",
                "expect": "Reference",
                "scenario": scen_short,
                "d1": m1,
                "d10": m10,
                "ratio": ratio,
            })

    # Sensitivity runs
    for key in ["S1", "S2", "S3", "S4", "S5"]:
        p = impacts_paths.get(key)
        if p is None or not p.exists():
            continue

        dfi_all = pd.read_csv(p)

        for pref in meta[key]["scenario_prefixes"]:
            if key == "S5":
                # For S5 we report the +20% (high standing charge) case for transparency
                opts = sorted(
                    dfi_all.loc[
                        dfi_all["scenario"].astype(str).str.startswith(pref)
                        & dfi_all["scenario"].astype(str).str.contains(r"_S5_SC120"),
                        "scenario",
                    ].unique()
                )
                if not opts:
                    scen_name = _pick_scenario(dfi_all, pref)
                else:
                    scen_name = opts[0]
            else:
                scen_name = _pick_scenario(dfi_all, pref)
            dfi = dfi_all[dfi_all["scenario"].astype(str).eq(scen_name)].copy()

            d1 = dfi[dfi["income_decile_eqincdmp"].astype(int).eq(1)]
            d10 = dfi[dfi["income_decile_eqincdmp"].astype(int).eq(10)]

            m1 = weighted_mean(d1["liability_share_eqinc"], d1["weight_annual"]) if len(d1) else float("nan")
            m10 = weighted_mean(d10["liability_share_eqinc"], d10["weight_annual"]) if len(d10) else float("nan")
            ratio = (m1 / m10) if (m10 == m10 and m10 != 0.0) else float("nan")

            scen_short = scen_name.split("_")[0]

            rows.append({
                "sens": key,
                "change": meta[key]["change"],
                "expect": meta[key]["expect"],
                "scenario": scen_short,
                "d1": m1,
                "d10": m10,
                "ratio": ratio,
            })

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l p{3.4cm} p{3.8cm} >{\centering\arraybackslash}p{1.6cm} r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Run} & \textbf{What changes} & \textbf{Expected direction} & \textbf{Scenario} & \textbf{\shortstack{D1\\(\% eqinc)}} & \textbf{\shortstack{D10\\(\% eqinc)}} & \textbf{D1/D10} \\\\"
    )
    lines.append(r"\midrule")

    def _tex_escape_text(s: str) -> str:
        # Escape only characters that will break tabular alignment.
        return s.replace("&", r"\&")

    for r in rows:
        # `change` and `expect` strings intentionally include LaTeX commands (e.g. \pounds, \%), so only escape '&'.
        change = _tex_escape_text(str(r["change"]))
        expect = _tex_escape_text(str(r["expect"]))
        scen = str(r["scenario"])  # already short
        lines.append(
            f"{r['sens']} & {change} & {expect} & {scen} & {_fmt_pct_points(r['d1'])} & {_fmt_pct_points(r['d10'])} & {_fmt_ratio(r['ratio'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Sensitivity summary (Baseline, S1--S5). Headline metrics report weighted mean liabilities as shares of equivalised income for decile 1 and decile 10. Baseline rows provide context for the sensitivity runs. Two affected scenarios are shown for each sensitivity run (A and C for unit levies; B and D for standing charges).}")
    lines.append(r"\label{tab:sensitivity-summary}")
    lines.append(r"\end{table}")


    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# New function: write_effective_n_tex
def write_effective_n_tex(
    *,
    hh_path: Path,
    out_dir: Path,
    out_name: str = "tab_effective_n.tex",
    eqinc_floor_annual: float = 1000.0,
) -> Path:
    """Write a compact LaTeX table reporting effective sample sizes and exclusions.

    Uses the household-level analysis dataset to compute (i) unweighted N and (ii) weighted
    shares retained/excluded for the denominators used in headline normalisations.
    """
    if not hh_path.exists():
        raise FileNotFoundError(f"Household analysis file not found: {hh_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfh = pd.read_csv(hh_path)

    # Required columns for the diagnostics we want.
    # Fallbacks: some builds may use slightly different naming.
    if "weight_annual" not in dfh.columns:
        raise ValueError("Expected `weight_annual` in household analysis file")

    # Equivalised income weekly (disposable, McClements) is used in the project.
    eqinc_weekly_col = "income_equivalised_weekly_eqincdmp"
    if eqinc_weekly_col not in dfh.columns:
        raise ValueError(f"Expected `{eqinc_weekly_col}` in household analysis file")

    totalexp_col = "total_expenditure_annualised"
    if totalexp_col not in dfh.columns:
        raise ValueError(f"Expected `{totalexp_col}` in household analysis file")

    w = pd.to_numeric(dfh["weight_annual"], errors="coerce")
    w_ok = w.notna() & (w >= 0)
    w_sum = float(w[w_ok].sum())
    n_unw = int(len(dfh))

    # Annualise weekly eqinc consistently (365/7).
    weeks_per_year = 365.0 / 7.0
    eqinc_weekly = pd.to_numeric(dfh[eqinc_weekly_col], errors="coerce")
    eqinc_annual = eqinc_weekly * weeks_per_year
    m_eqinc = w_ok & eqinc_weekly.notna() & (eqinc_annual >= float(eqinc_floor_annual))

    tot = pd.to_numeric(dfh[totalexp_col], errors="coerce")
    m_totexp = w_ok & tot.notna() & (tot > 0)

    # Levels: no denominator restrictions beyond having a weight
    share_levels_ret = float(w[m_eqinc | (~m_eqinc)].sum() / w_sum) if w_sum else float("nan")
    # but for clarity compute levels retained as weight_ok
    share_levels_ret = float(w[w_ok].sum() / w_sum) if w_sum else float("nan")

    share_eqinc_ret = float(w[m_eqinc].sum() / w_sum) if w_sum else float("nan")
    share_eqinc_exc = 1.0 - share_eqinc_ret if share_eqinc_ret == share_eqinc_ret else float("nan")

    share_totexp_ret = float(w[m_totexp].sum() / w_sum) if w_sum else float("nan")
    share_totexp_exc = 1.0 - share_totexp_ret if share_totexp_ret == share_totexp_ret else float("nan")

    def _fmt_pct(x: float, ndp: int = 2) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.{ndp}f}\\%"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Measure} & \textbf{N (unw.)} & \textbf{Retained (wtd.)} & \textbf{Excluded (wtd.)} \\"
    )
    lines.append(r"\midrule")
    lines.append(f"Levels (\\pounds) & {n_unw:,} & {_fmt_pct(share_levels_ret)} & {_fmt_pct(1.0 - share_levels_ret)} \\\\ ")
    lines.append(f"Share of eqinc (floor \\pounds {int(eqinc_floor_annual):,}) & {n_unw:,} & {_fmt_pct(share_eqinc_ret)} & {_fmt_pct(share_eqinc_exc)} \\\\ ")
    lines.append(f"Share of total expenditure & {n_unw:,} & {_fmt_pct(share_totexp_ret)} & {_fmt_pct(share_totexp_exc)} \\\\ ")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Effective sample sizes and exclusions for headline normalisations. Retained and excluded shares are weighted using survey weights. The equivalised income normalisation applies an annual floor of \pounds 1,000.}")
    lines.append(r"\label{tab:effective-n}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
# endregion LaTeX export helpers


def write_implied_bill_params_tex(
    *,
    checks_json_path: Path,
    benchmarks_csv_path: Path,
    out_dir: Path,
    out_name: str = IMPLIED_BILL_PARAMS_TEX,
) -> Path:
    """Write an illustrative table of implied bill impact parameters.

    This table is purely descriptive: it reports calibrated parameters (tau, s, u, kappa, rho)
    alongside a simple mapping to annualised cash amounts using baseline mean bills.

    Notes
    - For unit levies, the illustrative annual amount is tau times the baseline mean fuel spend.
    - For standing charges and the uniform rebate, the parameter is already an annual cash amount.
    - For the income tax proxy, we report kappa only (mapping to cash depends on the tax base definition).
    """
    if not checks_json_path.exists():
        raise FileNotFoundError(f"Scenario checks JSON not found: {checks_json_path}")
    if not benchmarks_csv_path.exists():
        raise FileNotFoundError(f"Benchmarks CSV not found: {benchmarks_csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    checks = json.loads(checks_json_path.read_text(encoding="utf-8"))
    b = pd.read_csv(benchmarks_csv_path)
    b_all = b[b["income_decile_eqincdmp"].astype(str).str.upper().eq("ALL")]
    if b_all.empty:
        raise ValueError("Benchmarks CSV missing ALL row")
    b_all = b_all.iloc[0]

    mean_gas = float(b_all.get("mean_gas_gbp_annual", float("nan")))
    mean_elec = float(b_all.get("mean_elec_gbp_annual", float("nan")))

    def _fmt_money(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"\\pounds\\,{float(x):,.0f}"

    def _fmt_money_2dp(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"\\pounds\\,{float(x):,.2f}"

    def _fmt_pct(x: float, ndp: int = 3) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.{ndp}f}\\%"

    rows = []

    # A: Gas unit levy
    rec = checks.get("A", {})
    tau_a = float(rec.get("tau", float("nan")))
    rows.append({
        "scenario": "A",
        "instrument": "Gas unit levy",
        "parameter": f"$\\tau$ = {_fmt_pct(tau_a, 3)}",
        "illustrative": _fmt_money(tau_a * mean_gas) if (tau_a == tau_a and mean_gas == mean_gas) else "",
    })

    # B: Electricity standing charge
    rec = checks.get("B", {})
    s_b = float(rec.get("s", float("nan")))
    rows.append({
        "scenario": "B",
        "instrument": "Electricity standing charge",
        "parameter": f"$s$ = {_fmt_money_2dp(s_b)} per year",
        "illustrative": _fmt_money(s_b) if s_b == s_b else "",
    })

    # C: Electricity unit levy
    rec = checks.get("C", {})
    tau_c = float(rec.get("tau", float("nan")))
    rows.append({
        "scenario": "C",
        "instrument": "Electricity unit levy",
        "parameter": f"$\\tau$ = {_fmt_pct(tau_c, 3)}",
        "illustrative": _fmt_money(tau_c * mean_elec) if (tau_c == tau_c and mean_elec == mean_elec) else "",
    })

    # D: Gas standing charge
    rec = checks.get("D", {})
    s_d = float(rec.get("s", float("nan")))
    rows.append({
        "scenario": "D",
        "instrument": "Gas standing charge",
        "parameter": f"$s$ = {_fmt_money_2dp(s_d)} per year",
        "illustrative": _fmt_money(s_d) if s_d == s_d else "",
    })

    # E1: Uniform rebate
    rec = checks.get("E1", {})
    u = float(rec.get("u", float("nan")))
    rows.append({
        "scenario": "E1",
        "instrument": "Uniform rebate",
        "parameter": f"$u$ = {_fmt_money_2dp(u)} per year",
        "illustrative": _fmt_money(u) if u == u else "",
    })

    # E2: Income tax proxy
    rec = checks.get("E2", {})
    kappa = float(rec.get("kappa", float("nan")))
    rows.append({
        "scenario": "E2",
        "instrument": "Income tax proxy",
        "parameter": f"$\\kappa$ = {_fmt_pct(kappa, 3)}",
        "illustrative": "",
    })

    # E3: Proportional income levy (NI-style proxy)
    rec = checks.get("E3", {})
    rho = float(rec.get("rho", float("nan")))
    rows.append({
        "scenario": "E3",
        "instrument": "Proportional income levy (NI proxy)",
        "parameter": f"$\\rho$ = {_fmt_pct(rho, 3)}",
        "illustrative": "",
    })


    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l p{4.4cm} p{4.4cm} r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Scenario} & \textbf{Instrument} & \textbf{Calibrated parameter} & \textbf{Annual amount (\pounds)} \\"
    )
    lines.append(r"\midrule")

    for r in rows:
        lines.append(f"{r['scenario']} & {r['instrument']} & {r['parameter']} & {r['illustrative']} \\\\ ")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Implied bill impact parameters (illustrative). For unit levies, the annual amount is computed as the calibrated levy rate times the baseline mean annual fuel expenditure (ALL row of Table~\ref{tab:baseline-energy-benchmarks-decile}). Standing charges and the uniform rebate are annual cash amounts by construction.}"
    )
    lines.append(r"\label{tab:implied-bill-params}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# New function: write_tau_to_pkwh_tex
def write_tau_to_pkwh_tex(
    *,
    checks_json_path: Path,
    out_dir: Path,
    out_name: str = TAU_TO_PKWH_TEX,
    revenue_targets_gbp: Optional[List[float]] = None,
    assumed_gas_p_per_kwh: float = 8.65,
    assumed_elec_p_per_kwh: float = 31.00,
) -> Path:
    """Write an illustrative table mapping calibrated unit levy rates (tau) to p/kWh uplifts.

    The incidence engine calibrates an ad valorem levy rate tau such that revenue equals a target R,
    given the model’s levy base (net of the standing charge proxy for unit levy scenarios).

    For a range of revenue targets R, we compute:
        tau_A(R) = R / base_A
        tau_C(R) = R / base_C
    and translate to an approximate unit-rate uplift using assumed average FY 2022--23 unit rates:
        Δp/kWh ≈ tau × \bar{p/kWh}

    Notes
    - This is an illustrative translation using a single average tariff; it is not a billing forecast.
    - It is static (no behavioural response) and ignores regional and within-year tariff variation.
    - Because the levy is applied to a net spend base (not directly to kWh), the p/kWh mapping is approximate.
    """

    if not checks_json_path.exists():
        raise FileNotFoundError(f"Scenario checks JSON not found: {checks_json_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    checks = json.loads(checks_json_path.read_text(encoding="utf-8"))

    # Revenue targets to display
    if revenue_targets_gbp is None:
        revenue_targets_gbp = [
            100_000_000,
            250_000_000,
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            5_000_000_000,
        ]

    # Use the calibrated tau at the baseline revenue requirement R0, and scale linearly with R.
    # This avoids relying on base_weighted_sum, which may be stored on a normalised (non-national) scale.
    try:
        rec_A = checks.get("A", {})
        rec_C = checks.get("C", {})
        tau0_A = float(rec_A.get("tau"))
        tau0_C = float(rec_C.get("tau"))
        R0_A = float(rec_A.get("revenue_target"))
        R0_C = float(rec_C.get("revenue_target"))
    except Exception as e:
        raise ValueError("Could not read tau and revenue_target for scenarios A and C from checks JSON") from e

    if not (tau0_A == tau0_A and tau0_C == tau0_C and R0_A > 0 and R0_C > 0):
        raise ValueError(f"Expected finite tau and positive revenue_target for A and C, got tau0_A={tau0_A}, R0_A={R0_A}, tau0_C={tau0_C}, R0_C={R0_C}")

    def _fmt_money(x: float) -> str:
        return f"\\pounds\\,{float(x):,.0f}"

    def _fmt_tau(x: float) -> str:
        return f"{float(x):.2f}\\%"

    def _fmt_pkwh(x: float) -> str:
        return f"{float(x):.2f}"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Revenue target $R$} & \textbf{$\tau_A$ (gas)} & \textbf{$\Delta p/kWh$ (gas)} & \textbf{$\tau_C$ (elec)} & \textbf{$\Delta p/kWh$ (elec)} \\"
    )
    lines.append(r"\midrule")

    for R in revenue_targets_gbp:
        tau_A = tau0_A * (float(R) / R0_A)
        tau_C = tau0_C * (float(R) / R0_C)
        # tau_A and tau_C are in percentage points; convert to fraction for uplift calculation
        uplift_gas = (tau_A / 100.0) * float(assumed_gas_p_per_kwh)
        uplift_elec = (tau_C / 100.0) * float(assumed_elec_p_per_kwh)
        lines.append(
            f"{_fmt_money(R)} & {_fmt_tau(tau_A)} & {_fmt_pkwh(uplift_gas)} & {_fmt_tau(tau_C)} & {_fmt_pkwh(uplift_elec)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Illustrative translation from calibrated unit levy rates ($\tau$) to an approximate unit-rate uplift (p/kWh) for a range of annual revenue targets.}"
    )
    lines.append(r"\label{tab:tau-to-pkwh}")
    lines.append(r"\par\vspace{0.25em}\footnotesize\textit{Note:} Assumptions and limitations for the p/kWh mapping are stated in the main text.")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_welfare_atkinson_tex(
    *,
    impacts_df: pd.DataFrame,
    hh_path: Path,
    out_dir: Path,
    out_name: str = ATKINSON_WELFARE_TEX,
    scenarios: Optional[List[str]] = None,
    eps_list: Optional[List[float]] = None,
    income_floor_annual: float = 1000.0,
) -> Path:
    """Write a compact LaTeX table ranking scenarios by Atkinson welfare (EDE income).

    We compute post-policy equivalised disposable income as:
        y_post = y_base - liability
    where y_base is equivalised disposable income (weekly) annualised using 365/7.

    For inequality aversion eps, we report the equally distributed equivalent income (EDE).
    Rates are weighted using the survey weight `weight_annual` from the impacts file.

    Notes
    - EDE requires strictly positive income; we impose an annual floor for welfare calculations only.
    - This is a normative ranking; results can vary with eps.
    """

    if scenarios is None:
        scenarios = ["A", "B", "C", "D", "E1", "E2", "E3"]

    if eps_list is None:
        eps_list = [0.0, 0.5, 1.0, 2.0]

    if not hh_path.exists():
        raise FileNotFoundError(f"Household analysis file not found: {hh_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Income from household analysis
    dfh = pd.read_csv(hh_path)
    if "case" not in dfh.columns:
        raise ValueError("Expected `case` in household analysis file")

    inc_weekly_col = "income_equivalised_weekly_eqincdmp"
    if inc_weekly_col not in dfh.columns:
        raise ValueError(f"Expected `{inc_weekly_col}` in household analysis file")

    weeks_per_year = 365.0 / 7.0
    dfh = dfh[["case", inc_weekly_col]].copy()
    dfh["eqinc_annual"] = pd.to_numeric(dfh[inc_weekly_col], errors="coerce") * weeks_per_year

    if "case" not in impacts_df.columns:
        raise ValueError("Expected `case` in impacts dataframe")
    if "scenario" not in impacts_df.columns:
        raise ValueError("Expected `scenario` in impacts dataframe")
    if "weight_annual" not in impacts_df.columns:
        raise ValueError("Expected `weight_annual` in impacts dataframe")
    if "liability_gbp_annual" not in impacts_df.columns:
        raise ValueError("Expected `liability_gbp_annual` in impacts dataframe")

    dfm = impacts_df.merge(dfh[["case", "eqinc_annual"]], on="case", how="left")
    dfm["eqinc_post"] = pd.to_numeric(dfm["eqinc_annual"], errors="coerce") - pd.to_numeric(
        dfm["liability_gbp_annual"], errors="coerce"
    )

    rows = []
    for scen in scenarios:
        sub = dfm[dfm["scenario"].astype(str).str.startswith(f"{scen}_")].copy()
        if sub.empty:
            continue

        w = sub["weight_annual"]
        y_post = sub["eqinc_post"]

        mu = weighted_mean(_enforce_floor(y_post, income_floor_annual), w)

        # Share floored (weighted)
        y_num = _to_num(y_post)
        w_num = _to_num(w)
        m = y_num.notna() & w_num.notna() & (w_num >= 0)
        floored_share = float("nan")
        if m.any():
            W = float(w_num[m].sum())
            if W > 0:
                floored_share = float(w_num[m & (y_num < float(income_floor_annual))].sum() / W)

        rec = {
            "scenario": scen,
            "mean_post": mu,
            "floored_share": floored_share,
        }

        for eps in eps_list:
            ede = weighted_ede_atkinson(y_post, w, float(eps), floor=income_floor_annual)
            rec[f"ede_eps{str(eps).replace('.', '_')}"] = ede

        rows.append(rec)

    dfo = pd.DataFrame(rows)

    # Build LaTeX
    def _fmt_money0(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"\\pounds\\,{float(x):,.0f}"

    def _fmt_pct(x: float, ndp: int = 1) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.{ndp}f}\\%"

    # Column headers for eps list
    eps_labels = []
    for eps in eps_list:
        if float(eps).is_integer():
            eps_labels.append(str(int(eps)))
        else:
            eps_labels.append(str(eps))

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")

    # tabular spec: scenario + mean + EDEs + floored share
    n_ede = len(eps_list)
    spec = "l r " + ("r " * n_ede) + "r"
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")

    # Header: shortened as per instructions
    header = r"\textbf{Scenario} & \textbf{Mean post}"
    for lab in eps_labels:
        header += rf" & \textbf{{$\varepsilon={lab}$}}"
    header += r" & \textbf{Floored} \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, r in dfo.iterrows():
        row = f"{r['scenario']} & {_fmt_money0(r['mean_post'])}"
        for eps in eps_list:
            key = f"ede_eps{str(eps).replace('.', '_')}"
            row += f" & {_fmt_money0(r.get(key, float('nan')))}"
        row += f" & {_fmt_pct(r['floored_share'], 1)} \\\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Social welfare ranking using Atkinson equally distributed equivalent (EDE) income computed on post-policy equivalised disposable income. Columns labelled by $\varepsilon$ report EDE income in \pounds. EDE is reported for inequality aversion parameters $\varepsilon\in\{0,0.5,1,2\}$. An annual income floor of \pounds "
        + f"{int(income_floor_annual):,}"
        + r" is applied for welfare calculations only.}"
    )
    lines.append(r"\label{tab:welfare-atkinson}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

# New function: write_progressivity_kakwani_tex
def write_progressivity_kakwani_tex(
    *,
    impacts_df: pd.DataFrame,
    hh_path: Path,
    out_dir: Path,
    out_name: str = "tab_progressivity_kakwani.tex",
    scenarios: Optional[List[str]] = None,
) -> Path:
    """Write a compact LaTeX table of progressivity metrics (Gini, concentration, Kakwani).

    We compute the weighted Gini coefficient of equivalised income (baseline distribution),
    then compute concentration coefficients of liabilities for each scenario using the
    same income ranking. The Kakwani index is K = C - G.

    Notes
    - Income is equivalised disposable income (weekly) annualised using 365/7.
    - Liabilities are treated as non-negative payments for this progressivity summary.
    """

    if scenarios is None:
        scenarios = ["A", "B", "C", "D", "E1", "E2", "E3"]

    if not hh_path.exists():
        raise FileNotFoundError(f"Household analysis file not found: {hh_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Income ranking from household analysis
    dfh = pd.read_csv(hh_path)
    if "case" not in dfh.columns:
        raise ValueError("Expected `case` in household analysis file")

    inc_weekly_col = "income_equivalised_weekly_eqincdmp"
    if inc_weekly_col not in dfh.columns:
        raise ValueError(f"Expected `{inc_weekly_col}` in household analysis file")

    weeks_per_year = 365.0 / 7.0
    dfh = dfh[["case", inc_weekly_col]].copy()
    dfh["eqinc_annual"] = pd.to_numeric(dfh[inc_weekly_col], errors="coerce") * weeks_per_year

    # Merge income into impacts
    if "case" not in impacts_df.columns:
        raise ValueError("Expected `case` in impacts dataframe")

    dfm = impacts_df.merge(dfh[["case", "eqinc_annual"]], on="case", how="left")

    if "weight_annual" not in dfm.columns:
        raise ValueError("Expected `weight_annual` in impacts dataframe")
    if "liability_gbp_annual" not in dfm.columns:
        raise ValueError("Expected `liability_gbp_annual` in impacts dataframe")

    # Compute baseline Gini of equivalised income using the weights in the impacts file.
    # Use any one scenario slice to avoid duplicated cases.
    df_any = dfm[dfm["scenario"].astype(str).str.startswith("A_")].copy()
    if df_any.empty:
        df_any = dfm.drop_duplicates(subset=["case"]).copy()

    G = weighted_gini(df_any["eqinc_annual"], df_any["weight_annual"])

    rows = []
    for scen_prefix in scenarios:
        sub = dfm[dfm["scenario"].astype(str).str.startswith(f"{scen_prefix}_")].copy()
        if sub.empty:
            continue

        # Use concentration coefficient of liabilities ranked by income
        Cc, _, _ = weighted_concentration(sub["eqinc_annual"], sub["liability_gbp_annual"], sub["weight_annual"])
        K = Cc - G if (Cc == Cc and G == G) else float("nan")

        rows.append({"scenario": scen_prefix, "gini": G, "concentration": Cc, "kakwani": K})

    dfo = pd.DataFrame(rows)

    def _fmt(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{float(x):.3f}"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Scenario} & \textbf{Gini (income)} & \textbf{Concentration (payments)} & \textbf{Kakwani (C--G)} \\\\")
    lines.append(r"\midrule")

    for _, r in dfo.iterrows():
        lines.append(f"{r['scenario']} & {_fmt(r['gini'])} & {_fmt(r['concentration'])} & {_fmt(r['kakwani'])} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Progressivity summary using Kakwani indices. The Gini coefficient is computed for equivalised disposable income (annualised). The concentration coefficient is computed for annual household liabilities ranked by equivalised income. Kakwani is the difference $K=C-G$: positive values indicate progressive payment incidence relative to income; negative values indicate regressivity.}")
    lines.append(r"\label{tab:progressivity-kakwani}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# New function: write_rurality_share_eqinc_tex
def write_rurality_share_eqinc_tex(
    *,
    csv_path: Path,
    out_dir: Path,
    out_name: str = "tab_rurality_share_eqinc.tex",
) -> Path:
    """Write a compact rurality subgroup table (mean liability as share of eqinc).

    Uses the precomputed secondary results-pack CSV and formats values as percentages of
    equivalised income (not percentage points).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Rurality subgroup CSV not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfr = pd.read_csv(csv_path)

    required = {
        "rurality_label",
        "A_gas_unit_levy",
        "C_elec_unit_levy",
        "E2_income_tax_proxy",
    }
    missing = sorted(required - set(dfr.columns))
    if missing:
        raise ValueError(f"Rurality CSV missing required columns: {missing}")

    # Normalise labels
    def _lab(x: str) -> str:
        s = "" if pd.isna(x) else str(x).strip()
        if s == "":
            return "Missing (NI)"
        return s

    dfr = dfr.copy()
    dfr["rurality_label"] = dfr["rurality_label"].apply(_lab)

    # Order rows sensibly
    order = ["urban", "rural", "Missing (NI)"]
    dfr["_ord"] = dfr["rurality_label"].map({k: i for i, k in enumerate(order)}).fillna(99)
    dfr = dfr.sort_values(["_ord", "rurality_label"]).drop(columns=["_ord"]).reset_index(drop=True)

    def _fmt_pct(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.3f}\\%"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Rurality} & \textbf{A: Gas levy} & \textbf{C: Elec levy} & \textbf{E2: Income tax} \\"
    )
    lines.append(r"\midrule")

    for _, row in dfr.iterrows():
        lines.append(
            f"{row['rurality_label']} & {_fmt_pct(row['A_gas_unit_levy'])} & {_fmt_pct(row['C_elec_unit_levy'])} & {_fmt_pct(row['E2_income_tax_proxy'])} \\\\ "
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Additional subgroup results by rurality. Entries report mean liabilities as shares of equivalised income for selected scenarios. The Missing (NI) category reflects Northern Ireland households where rurality is unavailable by construction.}"
    )
    lines.append(r"\label{tab:rurality-share-eqinc}")
    lines.append(r"\end{table}")

    out_path = out_dir / out_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

# -----------------------------------------------------------------------------
# region Main
# -----------------------------------------------------------------------------


def main(impacts_path: Path = DEFAULT_IMPACTS_PATH, *, tag: Optional[str] = None) -> None:
    #Route outputs to specific folders for file management
    if tag:
        PACK_DIR = ROOT / "outputs" / "results_pack_sensitivities" / tag
    else:
        PACK_DIR = ROOT / "outputs" / "results_pack"

    TABLE_DIR = PACK_DIR / "tables"
    HEADLINE_DIR = TABLE_DIR / "headline"
    SECONDARY_DIR = TABLE_DIR / "secondary"

    PACK_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    HEADLINE_DIR.mkdir(parents=True, exist_ok=True)
    SECONDARY_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure the report-facing LaTeX tables directory exists
    TEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(impacts_path)

    # Check necessary columns are present
    required = {
        "case",
        "scenario",
        "liability_gbp_annual",
        "liability_share_eqinc",
        "liability_share_totalexp",
        "weight_annual",
        "income_decile_eqincdmp",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Impacts file missing required columns: {missing}")

    # Coerce deciles to an ordered numeric type where possible
    df["income_decile_eqincdmp"] = pd.to_numeric(df["income_decile_eqincdmp"], errors="coerce")

    # 1) Mean and median impacts in pounds by decile
    decile_levels = summarise_by_scenario_and_group(
        df,
        value_col="liability_gbp_annual",
        group_col="income_decile_eqincdmp",
        quantiles=[0.5],
    )
    decile_levels.to_csv(TABLE_DIR / "table_decile_levels_tidy.csv", index=False)

    decile_levels_mean_wide = pivot_scenario_table(decile_levels, "income_decile_eqincdmp", "mean")
    decile_levels_p50_wide = pivot_scenario_table(decile_levels, "income_decile_eqincdmp", "p50")

    decile_levels_mean_wide.to_csv(TABLE_DIR / "table_decile_levels_mean.csv", index=False)
    decile_levels_p50_wide.to_csv(TABLE_DIR / "table_decile_levels_median.csv", index=False)

    # 2) Impacts as shares by decile (mean only by default)
    decile_share_eqinc = summarise_by_scenario_and_group(
        df,
        value_col="liability_share_eqinc",
        group_col="income_decile_eqincdmp",
        quantiles=[0.5],
    )
    decile_share_totalexp = summarise_by_scenario_and_group(
        df,
        value_col="liability_share_totalexp",
        group_col="income_decile_eqincdmp",
        quantiles=[0.5],
    )

    decile_share_eqinc.to_csv(TABLE_DIR / "table_decile_share_eqinc_tidy.csv", index=False)
    decile_share_totalexp.to_csv(TABLE_DIR / "table_decile_share_totalexp_tidy.csv", index=False)

    pivot_scenario_table(decile_share_eqinc, "income_decile_eqincdmp", "mean").to_csv(
        TABLE_DIR / "table_decile_share_eqinc_mean.csv", index=False
    )
    pivot_scenario_table(decile_share_totalexp, "income_decile_eqincdmp", "mean").to_csv(
        TABLE_DIR / "table_decile_share_totalexp_mean.csv", index=False
    )

    # 3) Subgroup tables (levels and shares; mean and weighted median)
    subgroup_cols: List[str] = [
        "pensioner_household",
        "has_children",
        "tenure_4cat",
        "region_label",
        "rurality_label",
    ]

    # Loop through distributional subgroup columns, producing tidy and pivoted tables
    subgroup_outputs = {}
    for gcol in subgroup_cols:
        if gcol not in df.columns:
            continue
        tidy = summarise_by_scenario_and_group(
            df,
            value_col="liability_gbp_annual",
            group_col=gcol,
            quantiles=[0.5],
        )
        tidy.to_csv(TABLE_DIR / f"table_{gcol}_levels_tidy.csv", index=False)
        pivot_scenario_table(tidy, gcol, "mean").to_csv(TABLE_DIR / f"table_{gcol}_levels_mean.csv", index=False)
        pivot_scenario_table(tidy, gcol, "p50").to_csv(TABLE_DIR / f"table_{gcol}_levels_median.csv", index=False)

        # Shares of equivalised income
        tidy_eqinc = summarise_by_scenario_and_group(
            df,
            value_col="liability_share_eqinc",
            group_col=gcol,
            quantiles=[0.5],
        )
        tidy_eqinc.to_csv(TABLE_DIR / f"table_{gcol}_share_eqinc_tidy.csv", index=False)
        pivot_scenario_table(tidy_eqinc, gcol, "mean").to_csv(
            TABLE_DIR / f"table_{gcol}_share_eqinc_mean.csv", index=False
        )
        pivot_scenario_table(tidy_eqinc, gcol, "p50").to_csv(
            TABLE_DIR / f"table_{gcol}_share_eqinc_median.csv", index=False
        )

        # Shares of total expenditure
        tidy_totalexp = summarise_by_scenario_and_group(
            df,
            value_col="liability_share_totalexp",
            group_col=gcol,
            quantiles=[0.5],
        )
        tidy_totalexp.to_csv(TABLE_DIR / f"table_{gcol}_share_totalexp_tidy.csv", index=False)
        pivot_scenario_table(tidy_totalexp, gcol, "mean").to_csv(
            TABLE_DIR / f"table_{gcol}_share_totalexp_mean.csv", index=False
        )
        pivot_scenario_table(tidy_totalexp, gcol, "p50").to_csv(
            TABLE_DIR / f"table_{gcol}_share_totalexp_median.csv", index=False
        )

        subgroup_outputs[gcol] = {
            "n_groups": int(tidy[gcol].nunique(dropna=False)),
            "tables": [
                "levels",
                "share_eqinc",
                "share_totalexp",
            ],
        }

    # Create file for tracking the contents of the results pack, including metadata on the source impacts file and the tables included
    manifest = {
        "impacts_path": str(impacts_path),
        "tag": tag,
        "pack_dir": str(PACK_DIR),
        "table_dir": str(TABLE_DIR),
        "scenarios": sorted(df["scenario"].dropna().unique().tolist()),
        "n_rows": int(len(df)),
        "subgroups_written": subgroup_outputs,
    }
    (PACK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Core results for report
    headline_results = [
        # Decile headline tables
        "table_decile_levels_mean.csv",
        "table_decile_levels_median.csv",
        "table_decile_share_eqinc_mean.csv",
        "table_decile_share_totalexp_mean.csv",

        # Subgroup headline tables (supporting the headline figures)
        "table_tenure_4cat_levels_mean.csv",
        "table_tenure_4cat_share_eqinc_mean.csv",
        "table_pensioner_household_levels_mean.csv",
        "table_pensioner_household_share_eqinc_mean.csv",
    ]

    # Copy core results into dedicated folder for ease
    for fname in headline_results:
        src = TABLE_DIR / fname
        if src.exists():
            (HEADLINE_DIR / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Copy the remainder into a secondary folder
    for p in TABLE_DIR.glob("*.csv"):
        if p.name in set(headline_results):
            continue
        (SECONDARY_DIR / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    # Put everything else into secondary folder
    secondary_outputs = sorted([p.name for p in SECONDARY_DIR.glob("*.csv")])

    headline_manifest = {
        "headline_results": headline_results,
        "secondary_outputs": secondary_outputs,
    }
    (PACK_DIR / "headline_manifest.json").write_text(
        json.dumps(headline_manifest, indent=2), encoding="utf-8"
    )

    # Results pack index: mapping figures and tables to their metadata for report reference and traceability back to source files.
    index_rows = [
        # Headline figures (generated by results_figures.py)
        {
            "id": "F1",
            "type": "figure",
            "title": "Mean annual liability by equivalised income decile (£ per year)",
            "filename": str((PACK_DIR / "figures" / "fig_decile_levels_mean.pdf").relative_to(ROOT)),
            "report_location": "Results: Headline figures",
        },
        {
            "id": "F2",
            "type": "figure",
            "title": "Mean liability as share of equivalised income by decile (%)",
            "filename": str((PACK_DIR / "figures" / "fig_decile_share_eqinc_mean.pdf").relative_to(ROOT)),
            "report_location": "Results: Headline figures",
        },
        {
            "id": "F3",
            "type": "figure",
            "title": "Mean liability as share of equivalised income by tenure (%)",
            "filename": str((PACK_DIR / "figures" / "fig_tenure_share_eqinc_mean.pdf").relative_to(ROOT)),
            "report_location": "Results: Headline figures",
        },
        {
            "id": "F4",
            "type": "figure",
            "title": "Mean liability as share of equivalised income by pensioner status (%)",
            "filename": str((PACK_DIR / "figures" / "fig_pensioner_share_eqinc_mean.pdf").relative_to(ROOT)),
            "report_location": "Results: Headline figures",
        },

        # Headline tables (generated by results_tables.py)
        {
            "id": "T1",
            "type": "table",
            "title": "Decile incidence in levels: weighted mean (£ per year)",
            "filename": str((HEADLINE_DIR / "table_decile_levels_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Baseline distributional impacts",
        },
        {
            "id": "T2",
            "type": "table",
            "title": "Decile incidence in levels: weighted median (£ per year)",
            "filename": str((HEADLINE_DIR / "table_decile_levels_median.csv").relative_to(ROOT)),
            "report_location": "Results: Baseline distributional impacts",
        },
        {
            "id": "T3",
            "type": "table",
            "title": "Decile incidence as share of equivalised income: weighted mean",
            "filename": str((HEADLINE_DIR / "table_decile_share_eqinc_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Baseline distributional impacts",
        },
        {
            "id": "T4",
            "type": "table",
            "title": "Decile incidence as share of total expenditure: weighted mean",
            "filename": str((HEADLINE_DIR / "table_decile_share_totalexp_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Baseline distributional impacts",
        },
        {
            "id": "T5",
            "type": "table",
            "title": "Incidence by tenure: weighted mean (£ per year)",
            "filename": str((HEADLINE_DIR / "table_tenure_4cat_levels_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Headline figures (supporting table)",
        },
        {
            "id": "T6",
            "type": "table",
            "title": "Incidence by tenure as share of equivalised income: weighted mean",
            "filename": str((HEADLINE_DIR / "table_tenure_4cat_share_eqinc_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Headline figures (supporting table)",
        },
        {
            "id": "T7",
            "type": "table",
            "title": "Incidence by pensioner status: weighted mean (£ per year)",
            "filename": str((HEADLINE_DIR / "table_pensioner_household_levels_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Headline figures (supporting table)",
        },
        {
            "id": "T8",
            "type": "table",
            "title": "Incidence by pensioner status as share of equivalised income: weighted mean",
            "filename": str((HEADLINE_DIR / "table_pensioner_household_share_eqinc_mean.csv").relative_to(ROOT)),
            "report_location": "Results: Headline figures (supporting table)",
        },
    ]

    index_df = pd.DataFrame(index_rows)
    index_path = PACK_DIR / "results_pack_index.csv"
    index_df.to_csv(index_path, index=False)

    # Write LaTeX tables used directly in the report
    try:
        out_tex = write_baseline_energy_benchmarks_tex(
            csv_path=BENCHMARKS_ENERGY_BY_DECILE,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except FileNotFoundError:
        # Benchmarks are optional for some runs; do not hard-fail the results pack.
        print("Note: benchmark CSV not found; skipped LaTeX benchmark table.")

    try:
        out_tex = write_scenario_calibration_diagnostics_tex(
            json_path=SCENARIO_CHECKS_JSON,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except FileNotFoundError:
        print("Note: scenario checks JSON not found; skipped LaTeX scenario diagnostics table.")

    try:
        out_tex = write_sensitivity_summary_tex(
            impacts_paths={
                "S1": SENS_S1_IMPACTS,
                "S2": SENS_S2_IMPACTS,
                "S3": SENS_S3_IMPACTS,
                "S4": SENS_S4_IMPACTS,
                "S5": SENS_S5_IMPACTS,
            },
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write sensitivity summary LaTeX table:", e)

    try:
        out_tex = write_effective_n_tex(
            hh_path=HH_ANALYSIS_PATH,
            out_dir=TEX_TABLES_DIR,
            eqinc_floor_annual=1000.0,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write effective N LaTeX table:", e)

    try:
        out_tex = write_implied_bill_params_tex(
            checks_json_path=SCENARIO_CHECKS_JSON,
            benchmarks_csv_path=BENCHMARKS_ENERGY_BY_DECILE,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write implied bill params LaTeX table:", e)

    try:
        out_tex = write_tau_to_pkwh_tex(
            checks_json_path=SCENARIO_CHECKS_JSON,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write tau-to-p/kWh LaTeX table:", e)

    try:
        out_tex = write_rurality_share_eqinc_tex(
            csv_path=RURALITY_EQINC_CSV,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write rurality subgroup LaTeX table:", e)

    try:
        out_tex = write_progressivity_kakwani_tex(
            impacts_df=df,
            hh_path=HH_ANALYSIS_PATH,
            out_dir=TEX_TABLES_DIR,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write progressivity (Kakwani) LaTeX table:", e)

    try:
        out_tex = write_welfare_atkinson_tex(
            impacts_df=df,
            hh_path=HH_ANALYSIS_PATH,
            out_dir=TEX_TABLES_DIR,
            income_floor_annual=1000.0,
        )
        print("Wrote LaTeX table:", out_tex)
    except Exception as e:
        print("Note: failed to write welfare (Atkinson) LaTeX table:", e)

    print("Wrote results pack to:", PACK_DIR)
    print("Headline tables:")
    for fname in headline_results:
        print("-", HEADLINE_DIR / fname)
    print("Secondary tables folder:")
    print("-", SECONDARY_DIR)
    print("Table list manifest:")
    print("-", PACK_DIR / "headline_manifest.json")
    print("-", PACK_DIR / "results_pack_index.csv")
# endregion Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distributional results tables from an impacts file.")
    parser.add_argument("--impacts", type=str, default=str(DEFAULT_IMPACTS_PATH), help="Path to impacts CSV")
    parser.add_argument("--tag", type=str, default=None, help="Optional sensitivity tag; writes to outputs/results_pack_sensitivities/<tag>/")
    args = parser.parse_args()

    main(Path(args.impacts), tag=args.tag)