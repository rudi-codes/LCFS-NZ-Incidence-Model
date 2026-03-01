"""Description:
Incidence engine for LCFS household microdata.

This module contains reusable functions for:
- loading the analysis dataset built by lcfs_load.py
- defining and running funding scenarios
- calibrating scenario parameters to hit a fixed annual revenue target
- producing household-level impacts and basic verification checks
"""

#region Package Imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
#endregion Package Imports

# region Constants and assumptions
ANNUAL_FACTOR: float = 365.0 / 7.0
# Annual gas standing charge proxy (GBP, per household-year) for FY 2022/23.
# Used to net off the fixed component from the unit-levy base in Scenario A.
GAS_STANDING_CHARGE_ANNUAL_GBP: float = 104.05
# Annual electricity standing charge proxy (GBP, per household-year) for FY 2022/23.
# Used to net off the fixed component from the unit-levy base in Scenario C.
ELEC_STANDING_CHARGE_ANNUAL_GBP: float = 165.31
#endregion Constants and assumptions

# -----------------------------------------------------------------------------
# region Scenario specification
# -----------------------------------------------------------------------------
# Setting up a data object that is passed into run_scenario.
@dataclass(frozen=True)
class ScenarioSpec:
    """Definition of a funding scenario.

    Parameters
    ----------
    name
        Short scenario label (e.g. "A_gas_levy").
    kind
        Either "levy" (proportional uplift on a monetary base), "standing_charge",
        "uniform_charge" (flat per-household amount), "income_tax_proxy"
        (progressive schedule scaled to a revenue target), or "income_proportional"
        (flat % of gross income scaled to a revenue target).
    base_col
        Column containing the monetary base (annual, in pounds). Required for levy and income_tax_proxy.
    payer_col
        Boolean indicator column for payer households. Required for standing charge.
    notes
        free text.
    """

    name: str
    kind: str  # "levy" | "standing_charge" | "uniform_charge" | "income_tax_proxy" | "income_proportional"
    base_col: Optional[str] = None
    payer_col: Optional[str] = None
    notes: str = ""
# endregion Scenario specification

# -----------------------------------------------------------------------------
# region Useful functions
# -----------------------------------------------------------------------------
# Read CSV to Dataframe and check columns
def load_analysis_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "case",
        "weight_annual",
        "income_equivalised_weekly_eqincdmp",
        "total_expenditure_annualised",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df

# Convert weekly to annual
def annualise_weekly(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * ANNUAL_FACTOR

# Ensure all numeric
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# Set up weighted sums
def _weighted_sum(x: pd.Series, w: pd.Series) -> float:
    x_num = _to_num(x)
    w_num = _to_num(w)
    m = x_num.notna() & w_num.notna()
    if not m.any():
        return 0.0
    return float((x_num[m] * w_num[m]).sum())
# endregion Useful functions

# -----------------------------------------------------------------------------
# region Scenario calcs with error handling
# -----------------------------------------------------------------------------
# Calculate a levy rate tau such that sum_i w_i * tau * base_i = R.
def calibrate_levy_rate(base: pd.Series, weight: pd.Series, revenue_target: float) -> float:
    denom = _weighted_sum(base, weight)
    if denom <= 0:
        raise ValueError(
            "Cannot calibrate levy: weighted sum of base is non-positive. "
            "Check base definition and payers."
        )
    return float(revenue_target / denom)

# Calculate rate an annual standing charge s such that sum_i w_i * s * payer_i = R.
def calibrate_standing_charge(payer: pd.Series, weight: pd.Series, revenue_target: float) -> float:
    payer_num = payer.astype(float)
    denom = _weighted_sum(payer_num, weight)
    if denom <= 0:
        raise ValueError(
            "Cannot calibrate standing charge: weighted payer count is non-positive. "
            "Check payer definition."
        )
    return float(revenue_target / denom)

# Calculate a uniform annual charge u such that sum_i w_i * u = R.
def calibrate_uniform_charge(weight: pd.Series, revenue_target: float) -> float:
    denom = _weighted_sum(pd.Series(np.ones(len(weight))), weight)
    if denom <= 0:
        raise ValueError("Cannot calibrate uniform charge: sum of weights is non-positive")
    return float(revenue_target / denom)

"""Notional UK income tax schedule (England/Wales/Northern Ireland) for 2022/23.
Applied as a stylised progressive benchmark to annual gross household income.
Bands (annual):
- 0% up to £12,570
- 20% on £12,571–£50,270 (next £37,700)
- 40% on £50,271–£150,000
- 45% above £150,000
This does not model allowance withdrawal, deductions, or Scottish bands.
    """
def notional_income_tax_2022_23_eewn(y_annual: pd.Series) -> pd.Series:
    y = _to_num(y_annual).fillna(0).clip(lower=0)

    pa = 12_570.0
    basic_upper = 50_270.0
    higher_upper = 150_000.0

    taxable = (y - pa).clip(lower=0)

    basic_band = (basic_upper - pa)
    basic_tax = 0.20 * taxable.clip(upper=basic_band)

    higher_band = (higher_upper - basic_upper)
    higher_tax = 0.40 * (taxable - basic_band).clip(lower=0, upper=higher_band)

    add_tax = 0.45 * (taxable - basic_band - higher_band).clip(lower=0)

    return basic_tax + higher_tax + add_tax

# Calibrate a scaling factor kappa such that sum_i w_i * kappa * T_i = R, where T_i is the notional tax under schedule above.
def calibrate_income_tax_proxy(notional_tax: pd.Series, weight: pd.Series, revenue_target: float) -> float:
    denom = _weighted_sum(notional_tax, weight)
    if denom <= 0:
        raise ValueError(
            "Cannot calibrate income tax proxy: weighted sum of notional tax is non-positive."
        )
    return float(revenue_target / denom)
# endregion Scenario calcs with error handling

# -----------------------------------------------------------------------------
# region Scenario execution with error handling
# -----------------------------------------------------------------------------

# Run a scenario and return household impacts plus summary checks.
def run_scenario(
    df: pd.DataFrame,
    spec: ScenarioSpec,
    revenue_target: float,
    *,
    weight_col: str = "weight_annual",
    eqinc_weekly_col: str = "income_equivalised_weekly_eqincdmp",
    total_exp_annual_col: str = "total_expenditure_annualised",
    clip_negative: bool = True,
    eqinc_floor_annual: float = 1000.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    if spec.kind not in {"levy", "standing_charge", "uniform_charge", "income_tax_proxy", "income_proportional"}:
        raise ValueError(f"Unknown scenario kind: {spec.kind}")

    out = df.copy()
    w = _to_num(out[weight_col])

    # QA metric used for Scenario A net-of-standing-charge base; default to NaN for other scenarios
    share_zero_after_net_weighted: float = float("nan")

    # Denominators
    eqinc_annual = annualise_weekly(out[eqinc_weekly_col])
    total_exp_annual = _to_num(out[total_exp_annual_col])

    if spec.kind == "levy":
        if not spec.base_col:
            raise ValueError("Levy scenario requires base_col")
        base_raw = _to_num(out[spec.base_col])
        if clip_negative:
            base_raw = base_raw.clip(lower=0)

        # Scenario A (gas unit levy) and Scenario C (electricity unit levy): net off an annual
        # standing charge proxy so the levy applies to the variable component of spend.
        if spec.name == "A_gas_unit_levy" and spec.base_col == "gas_expenditure_annualised":
            sc_annual = GAS_STANDING_CHARGE_ANNUAL_GBP
        elif spec.name == "C_elec_unit_levy" and spec.base_col == "electricity_expenditure_annualised":
            sc_annual = ELEC_STANDING_CHARGE_ANNUAL_GBP
        else:
            sc_annual = None

        if sc_annual is not None:
            payer = base_raw > 0
            base = (base_raw - sc_annual).clip(lower=0)

            payer_weighted_count = _weighted_sum(payer.astype(float), w)
            zero_after_net_weighted = _weighted_sum((payer & (base == 0)).astype(float), w)
            share_zero_after_net_weighted = (
                float(zero_after_net_weighted / payer_weighted_count) if payer_weighted_count > 0 else float("nan")
            )
        else:
            base = base_raw

        base_weighted_sum = _weighted_sum(base, w)
        tau = calibrate_levy_rate(base=base, weight=w, revenue_target=revenue_target)
        liability = tau * base

        calibrated_param_name = "tau"
        calibrated_param_value = tau

    elif spec.kind == "uniform_charge":
        u = calibrate_uniform_charge(weight=w, revenue_target=revenue_target)
        liability = pd.Series(np.full(len(out), u))

        calibrated_param_name = "u"
        calibrated_param_value = u

    elif spec.kind == "income_tax_proxy":
        if not spec.base_col:
            raise ValueError("Income tax proxy scenario requires base_col for gross income")

        y = _to_num(out[spec.base_col])
        if clip_negative:
            y = y.clip(lower=0)

        T = notional_income_tax_2022_23_eewn(y)

        # Tax payer indicator: positive notional income tax under the stylised schedule
        tax_payer = T > 0
        tax_payer_weighted_count = _weighted_sum(tax_payer.astype(float), w)
        weight_sum = float(_to_num(w).sum())
        tax_payer_share_unweighted = float(tax_payer.mean())
        tax_payer_share_weighted = (
            float(tax_payer_weighted_count / weight_sum) if weight_sum > 0 else float("nan")
        )

        tax_weighted_sum = _weighted_sum(T, w)
        kappa = calibrate_income_tax_proxy(notional_tax=T, weight=w, revenue_target=revenue_target)
        liability = kappa * T

        calibrated_param_name = "kappa"
        calibrated_param_value = kappa

    elif spec.kind == "income_proportional":
        if not spec.base_col:
            raise ValueError("Income proportional scenario requires base_col for gross income")

        y = _to_num(out[spec.base_col])
        if clip_negative:
            y = y.clip(lower=0)

        # Payer indicator: positive gross income
        payer = y > 0
        payer_weighted_count = _weighted_sum(payer.astype(float), w)
        weight_sum = float(_to_num(w).sum())
        payer_share_unweighted = float(payer.mean())
        payer_share_weighted = float(payer_weighted_count / weight_sum) if weight_sum > 0 else float("nan")

        income_weighted_sum = _weighted_sum(y, w)
        rho = calibrate_levy_rate(base=y, weight=w, revenue_target=revenue_target)
        liability = rho * y

        calibrated_param_name = "rho"
        calibrated_param_value = rho

    elif spec.kind == "standing_charge":
        if not spec.payer_col:
            raise ValueError("Standing charge scenario requires payer_col")
        payer = out[spec.payer_col].astype(bool)

        payer_weighted_count = _weighted_sum(payer.astype(float), w)
        weight_sum = float(_to_num(w).sum())
        payer_share_unweighted = float(payer.mean())
        payer_share_weighted = float(payer_weighted_count / weight_sum) if weight_sum > 0 else float("nan")

        s = calibrate_standing_charge(payer=payer, weight=w, revenue_target=revenue_target)
        liability = s * payer.astype(float)

        calibrated_param_name = "s"
        calibrated_param_value = s

    else:
        raise ValueError(f"Unknown scenario kind (unexpected): {spec.kind}")

    # Ensure HH liability is strictly positive or 0
    if clip_negative:
        liability = pd.Series(liability).clip(lower=0)

    out["scenario"] = spec.name
    out["liability_gbp_annual"] = liability

    # Share of equivalised income: treat near-zero annualised incomes as missing to avoid issues
    eqinc_ok = eqinc_annual >= float(eqinc_floor_annual)
    out["liability_share_eqinc"] = np.where(eqinc_ok, liability / eqinc_annual, np.nan)

    # Share of total expenditure: standard positive-denominator rule
    out["liability_share_totalexp"] = np.where(total_exp_annual > 0, liability / total_exp_annual, np.nan)

    revenue_implied = _weighted_sum(out["liability_gbp_annual"], w)

    liab_num = _to_num(out["liability_gbp_annual"]).fillna(0)
    p50, p90, p99, p999 = (float(liab_num.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999))
    liab_max = float(liab_num.max())

    # Tail check: extremely high incidence relative to total expenditure
    share_liab_over_50pct_totalexp = float((out["liability_share_totalexp"] > 0.5).mean())

    # Automated checks for QA and scenario comaprison.
    checks: Dict[str, float] = {
        "revenue_target": float(revenue_target),
        "revenue_implied": float(revenue_implied),
        "revenue_gap": float(revenue_implied - revenue_target),
        "liability_p50": float(p50),
        "liability_p90": float(p90),
        "liability_p99": float(p99),
        "liability_p99_9": float(p999),
        "liability_max": float(liab_max),
        "share_liab_over_50pct_totalexp": float(share_liab_over_50pct_totalexp),
        calibrated_param_name: float(calibrated_param_value),
        "share_negative_liability": float((out["liability_gbp_annual"] < 0).mean()),
        "share_missing_eqinc": float(eqinc_annual.isna().mean()),
        "share_eqinc_below_floor": float((~eqinc_ok & eqinc_annual.notna()).mean()),
        "eqinc_floor_annual": float(eqinc_floor_annual),
        "share_missing_totalexp": float(total_exp_annual.isna().mean()),
        "share_zero_base_after_net_weighted": float(share_zero_after_net_weighted),
    }

    if spec.kind == "levy":
        checks["base_weighted_sum"] = float(base_weighted_sum)
    elif spec.kind == "standing_charge":
        checks["payer_weighted_count"] = float(payer_weighted_count)
        checks["payer_share_unweighted"] = float(payer_share_unweighted)
        checks["payer_share_weighted"] = float(payer_share_weighted)
    elif spec.kind == "income_tax_proxy":
        checks["tax_weighted_sum"] = float(tax_weighted_sum)
        checks["tax_payer_weighted_count"] = float(tax_payer_weighted_count)
        checks["tax_payer_share_unweighted"] = float(tax_payer_share_unweighted)
        checks["tax_payer_share_weighted"] = float(tax_payer_share_weighted)
    elif spec.kind == "income_proportional":
        checks["income_weighted_sum"] = float(income_weighted_sum)
        checks["income_payer_weighted_count"] = float(payer_weighted_count)
        checks["income_payer_share_unweighted"] = float(payer_share_unweighted)
        checks["income_payer_share_weighted"] = float(payer_share_weighted)
    else:
        checks["weight_sum"] = float(_to_num(w).sum())

    # Specify outputs columns
    keep_cols = [
        "case",
        "scenario",
        "liability_gbp_annual",
        "liability_share_eqinc",
        "liability_share_totalexp",
        weight_col,
        eqinc_weekly_col,
        total_exp_annual_col,
    ]

    # Specift ditributional variables for analysis
    subgroup_cols = [
        "income_decile_eqincdmp",
        "tenure_4cat",
        "region_label",
        "rurality_label",
        "has_children",
        "pensioner_household",
        "ppm",
        "direct_debit",
    ]
    for c in subgroup_cols:
        if c in out.columns and c not in keep_cols:
            keep_cols.append(c)

    impacts = out[keep_cols].copy()

    return impacts, checks
# endregion Scenario execution with error handling

# -----------------------------------------------------------------------------
# region Scernario checks and summary helpers
# -----------------------------------------------------------------------------
# Scenario A: proportional levy on annual gas spend.
def scenario_A_gas_levy(base_col: str = "gas_expenditure_annualised") -> ScenarioSpec:
    return ScenarioSpec(name="A_gas_unit_levy", kind="levy", base_col=base_col)

# Scenario E1: uniform per-household annual charge.
def scenario_E1_uniform_charge() -> ScenarioSpec:
    return ScenarioSpec(name="E1_uniform_charge", kind="uniform_charge")

# Scenario E2: progressive income tax proxy on gross annual household income.
def scenario_E2_income_tax_proxy(base_col: str = "income_gross_annualised") -> ScenarioSpec:
    return ScenarioSpec(name="E2_income_tax_proxy", kind="income_tax_proxy", base_col=base_col)

# Scenario E3: proportional income levy (NI-style proxy) on gross annual household income.
def scenario_E3_income_proportional(base_col: str = "income_gross_annualised") -> ScenarioSpec:
    return ScenarioSpec(name="E3_income_proportional", kind="income_proportional", base_col=base_col)

# Flag on whether HH is a payer under scenario.
def make_payer_indicator(df: pd.DataFrame, base_col: str, new_col: str) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = _to_num(out[base_col]).fillna(0) > 0
    return out
# endregion Scernario checks and summary helpers

# -----------------------------------------------------------------------------
# region Aggregation functions
# -----------------------------------------------------------------------------
# Weighted mean as each row represents different proportion of population.
def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x_num = _to_num(x)
    w_num = _to_num(w)
    m = x_num.notna() & w_num.notna()
    if not m.any():
        return float("nan")
    return float((x_num[m] * w_num[m]).sum() / w_num[m].sum())

# Summarise mean incidence by a single (distributional) variable.
def summarise_by_group(impacts: pd.DataFrame, group_col: str, *, weight_col: str = "weight_annual") -> pd.DataFrame:
    if group_col not in impacts.columns:
        raise ValueError(f"Group column not found: {group_col}")

    rows = []
    for g, sub in impacts.groupby(group_col, dropna=False):
        w = sub[weight_col]
        rows.append(
            {
                group_col: g,
                "mean_gbp_annual": weighted_mean(sub["liability_gbp_annual"], w),
                "mean_share_eqinc": weighted_mean(sub["liability_share_eqinc"], w),
                "mean_share_totalexp": weighted_mean(sub["liability_share_totalexp"], w),
                "n_unweighted": int(len(sub)),
                "weight_sum": float(_to_num(w).sum()),
            }
        )

    return pd.DataFrame(rows)
# endregion Aggregation functions

# -----------------------------------------------------------------------------
# region Benchmarking functions
# -----------------------------------------------------------------------------

# Benchmark energy spend and share of total expenditure by equivalised income decile. Outputs are intended as stylised facts for QA, not as headline results.
def benchmark_energy_by_decile(
    df: pd.DataFrame,
    *,
    decile_col: str = "income_decile_eqincdmp",
    weight_col: str = "weight_annual",
    gas_col: str = "gas_expenditure_annualised",
    elec_col: str = "electricity_expenditure_annualised",
    totalexp_col: str = "total_expenditure_annualised",
) -> pd.DataFrame:
    required = {decile_col, weight_col, gas_col, elec_col, totalexp_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for benchmark: {missing}")

    d = df.copy()

    # Weighted mean function to reuse
    def _wmean(x: pd.Series, w: pd.Series) -> float:
        x_num = _to_num(x)
        w_num = _to_num(w)
        m = x_num.notna() & w_num.notna()
        if not m.any() or float(w_num[m].sum()) == 0.0:
            return float("nan")
        return float((x_num[m] * w_num[m]).sum() / w_num[m].sum())

    w_all = _to_num(d[weight_col])
    gas_all = _to_num(d[gas_col]).fillna(0)
    elec_all = _to_num(d[elec_col]).fillna(0)
    totalexp_all = _to_num(d[totalexp_col]).replace({0: np.nan})

    energy_all = gas_all + elec_all
    energy_share_all = energy_all / totalexp_all

    # Benchmark rows
    rows = [
        {
            decile_col: "ALL",
            "mean_gas_gbp_annual": _wmean(gas_all, w_all),
            "mean_elec_gbp_annual": _wmean(elec_all, w_all),
            "mean_energy_gbp_annual": _wmean(energy_all, w_all),
            "mean_energy_share_totalexp": _wmean(energy_share_all, w_all),
            "share_zero_gas": float((gas_all == 0).mean()),
            "share_zero_elec": float((elec_all == 0).mean()),
            "share_energy_share_gt_50pct": float((energy_share_all > 0.5).mean()),
            "n_unweighted": int(len(d)),
            "weight_sum": float(w_all.sum()),
        }
    ]

    for dec, sub in d.groupby(decile_col, dropna=False):
        ws = _to_num(sub[weight_col])
        gas_s = _to_num(sub[gas_col]).fillna(0)
        elec_s = _to_num(sub[elec_col]).fillna(0)
        totalexp_s = _to_num(sub[totalexp_col]).replace({0: np.nan})

        energy_s = gas_s + elec_s
        share_s = energy_s / totalexp_s

        rows.append(
            {
                decile_col: dec,
                "mean_gas_gbp_annual": _wmean(gas_s, ws),
                "mean_elec_gbp_annual": _wmean(elec_s, ws),
                "mean_energy_gbp_annual": _wmean(energy_s, ws),
                "mean_energy_share_totalexp": _wmean(share_s, ws),
                "share_zero_gas": float((gas_s == 0).mean()),
                "share_zero_elec": float((elec_s == 0).mean()),
                "share_energy_share_gt_50pct": float((share_s > 0.5).mean()),
                "n_unweighted": int(len(sub)),
                "weight_sum": float(ws.sum()),
            }
        )

    return pd.DataFrame(rows)
# endregion Benchmarking functions