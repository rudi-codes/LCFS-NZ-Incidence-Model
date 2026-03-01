"""Description:
Run incidence scenarios on the LCFS household analysis dataset.

This uses functions and set up from `incidence_engine.py`.
It loads the household dataset built by `lcfs_load.py`, executes scenarios, writes household-level impacts, and creates a QA JSON file.
Can be used to generate sensitivity runs (e.g. partial pass-through assumptions) as separate outputs.
"""

# region Package Imports
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import incidence_engine as ie
# endregion Package Imports

# region Paths and constants
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "outputs" / "lcfs_2022_23_HH_analysis.csv"
OUTDIR = ROOT / "outputs"
SENS_DIR = OUTDIR / "sensitivities"
# endregion Paths and constants

# --------------------------------------------------------------
# region Helper functions for running scenarios
# ---------------------------------------------------------------
"""Convert statutory liabilities into consumer incidence, to mechanically run sensitivities.

This helper implements the following interpretation of partial pass-through.

- Let R be the target amount recovered from domestic consumers (households).
- Let tau in (0, 1] be the assumed pass-through rate from statutory liabilities to consumers.

We therefore calibrate the statutory instrument to raise R / tau, then scale household-level
liabilities by tau so that the weighted sum of consumer liabilities still equals R.

This is a calibration device, not a model of firm incidence. Any residual (1 - tau) share is
not attributed to businesses within the LCFS household framework.
"""
def _apply_pass_through(
    impacts: pd.DataFrame,
    checks: dict,
    *,
    pass_through: float,
    scenario_suffix: str,
) -> tuple[pd.DataFrame, dict]:

    out = impacts.copy()
    out["scenario"] = out["scenario"].astype(str) + scenario_suffix

    scale_cols = [
        "liability_gbp_annual",
        "liability_share_eqinc",
        "liability_share_totalexp",
    ]
    for c in scale_cols:
        if c in out.columns:
            out[c] = pass_through * out[c]

    checks_out = dict(checks)
    checks_out["pass_through"] = float(pass_through)
    checks_out["revenue_target_consumer"] = float(checks.get("revenue_target", float("nan")) * pass_through)
    return out, checks_out

# Function adds suffix to scenario name
def _suffix_scenario(impacts: pd.DataFrame, checks: dict, *, suffix: str) -> tuple[pd.DataFrame, dict]:
    out = impacts.copy()
    out["scenario"] = out["scenario"].astype(str) + suffix
    checks_out = dict(checks)
    checks_out["scenario_suffix"] = suffix
    return out, checks_out
# endregion Helper functions for running scenarios

# --------------------------------------------------------------
# region Main function to run scenarios and write outputs
# --------------------------------------------------------------
def main() -> None:
    revenue_target = 1_000_000.0  # £1m

    df = ie.load_analysis_dataset(str(DATASET_PATH))

    # Payer indicators for standing charge scenarios
    df = ie.make_payer_indicator(
        df, base_col="electricity_expenditure_annualised", new_col="payer_electricity"
    )
    df = ie.make_payer_indicator(df, base_col="gas_expenditure_annualised", new_col="payer_gas")

    # Scenario A: gas unit levy (parameters taken from incidence_engine)
    impacts_A, checks_A = ie.run_scenario(df, ie.scenario_A_gas_levy(), revenue_target=revenue_target)

    # Scenario B: electricity standing charge (explicity defined)
    scen_B = ie.ScenarioSpec(
        name="B_elec_standing_charge",
        kind="standing_charge",
        payer_col="payer_electricity",
        notes="Payer defined as electricity_expenditure_annualised > 0",
    )
    impacts_B, checks_B = ie.run_scenario(df, scen_B, revenue_target=revenue_target)

    # Scenario C: electricity unit levy (explicity defined)
    scen_C = ie.ScenarioSpec(
        name="C_elec_unit_levy",
        kind="levy",
        base_col="electricity_expenditure_annualised",
        notes="Proportional levy on electricity_expenditure_annualised",
    )
    impacts_C, checks_C = ie.run_scenario(df, scen_C, revenue_target=revenue_target)

    # Scenario D: gas standing charge (explicity defined)
    scen_D = ie.ScenarioSpec(
        name="D_gas_standing_charge",
        kind="standing_charge",
        payer_col="payer_gas",
        notes="Payer defined as gas_expenditure_annualised > 0",
    )
    impacts_D, checks_D = ie.run_scenario(df, scen_D, revenue_target=revenue_target)

    # Scenario E1: uniform charge (parameters taken from incidence_engine)
    impacts_E1, checks_E1 = ie.run_scenario(df, ie.scenario_E1_uniform_charge(), revenue_target=revenue_target)

    # Scenario E2: income tax proxy (requires income_gross_annualised in dataset)
    impacts_E2, checks_E2 = ie.run_scenario(df, ie.scenario_E2_income_tax_proxy(), revenue_target=revenue_target)

    # Scenario E3: proportional income levy (NI-style proxy)
    impacts_E3, checks_E3 = ie.run_scenario(df, ie.scenario_E3_income_proportional(), revenue_target=revenue_target)

    impacts = pd.concat([impacts_A, impacts_B, impacts_C, impacts_D, impacts_E1, impacts_E2, impacts_E3], ignore_index=True)
    checks = {"A": checks_A, "B": checks_B, "C": checks_C, "D": checks_D, "E1": checks_E1, "E2": checks_E2, "E3": checks_E3}

    # Smoke test files to check outputs reasonable
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    impacts_path = OUTDIR / "impacts_scenarios_smoketest.csv"
    checks_path = OUTDIR / "checks_scenarios_smoketest.json"

    impacts.to_csv(impacts_path, index=False)
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    # Baseline benchmarks for QA: energy spend and energy budget shares by income decile
    bench = ie.benchmark_energy_by_decile(df)
    bench_path = OUTDIR / "benchmarks_energy_by_decile.csv"
    bench.to_csv(bench_path, index=False)

    print("Wrote:", impacts_path)
    print("Wrote:", checks_path)
    print("Wrote:", bench_path)
    print("Checks (A-D, E1, E2, E3):\n", json.dumps(checks, indent=2))
    # endregion Main function to run scenarios and write outputs

    # ------------------------------------------------------------------
    # region Sensitivity analysis (run as part of main, as relies on outputs)
    # ------------------------------------------------------------------
    '''We run a range of sensitivity scenarios to test the robustness of results to key assumptions. These include:
    - S1: Partial pass-through calibration for bill-based instruments (Scenarios A to D). We hold the target amount recovered from consumers fixed at R by (i) calibrating the statutory instrument to raise R / tau, then (ii) scaling household liabilities by tau (for example tau = 0.8).
    - S2: Higher equivalised-income floor for % income metric (e.g. £5,000 annual floor instead of £1,000).
    - S3: Standing charges applied to all households (i.e. all households treated as payers, instead of only those with positive energy expenditure).
    - S4: Revenue scale invariance (e.g. run all scenarios with a higher revenue target of £100m instead of £1m, to test whether patterns are consistent at different scales of intervention).
    '''

    # ------------------------------------------------------------------
    # region Sensitivity S1: Partial pass-through for bill-based instruments (Scenarios A-D)
    # ------------------------------------------------------------------
    pass_through = 0.8

    # We hold the target amount recovered from consumers fixed at R.
    # With pass-through tau < 1, we calibrate statutory liabilities to R / tau, then scale household liabilities by tau.
    revenue_target_statutory = revenue_target / pass_through

    impacts_A_s, checks_A_s = ie.run_scenario(df, ie.scenario_A_gas_levy(), revenue_target=revenue_target_statutory)

    scen_B_s = ie.ScenarioSpec(
        name="B_elec_standing_charge",
        kind="standing_charge",
        payer_col="payer_electricity",
        notes="Payer defined as electricity_expenditure_annualised > 0",
    )
    impacts_B_s, checks_B_s = ie.run_scenario(df, scen_B_s, revenue_target=revenue_target_statutory)

    scen_C_s = ie.ScenarioSpec(
        name="C_elec_unit_levy",
        kind="levy",
        base_col="electricity_expenditure_annualised",
        notes="Proportional levy on electricity_expenditure_annualised",
    )
    impacts_C_s, checks_C_s = ie.run_scenario(df, scen_C_s, revenue_target=revenue_target_statutory)

    scen_D_s = ie.ScenarioSpec(
        name="D_gas_standing_charge",
        kind="standing_charge",
        payer_col="payer_gas",
        notes="Payer defined as gas_expenditure_annualised > 0",
    )
    impacts_D_s, checks_D_s = ie.run_scenario(df, scen_D_s, revenue_target=revenue_target_statutory)

    # Convert statutory liabilities into consumer incidence under pass-through tau < 1 (keeping consumer revenue at R).
    impacts_A_s, checks_A_s = _apply_pass_through(impacts_A_s, checks_A_s, pass_through=pass_through, scenario_suffix="_S1_pt80")
    impacts_B_s, checks_B_s = _apply_pass_through(impacts_B_s, checks_B_s, pass_through=pass_through, scenario_suffix="_S1_pt80")
    impacts_C_s, checks_C_s = _apply_pass_through(impacts_C_s, checks_C_s, pass_through=pass_through, scenario_suffix="_S1_pt80")
    impacts_D_s, checks_D_s = _apply_pass_through(impacts_D_s, checks_D_s, pass_through=pass_through, scenario_suffix="_S1_pt80")

    impacts_s1 = pd.concat([impacts_A_s, impacts_B_s, impacts_C_s, impacts_D_s], ignore_index=True)
    checks_s1 = {"A": checks_A_s, "B": checks_B_s, "C": checks_C_s, "D": checks_D_s}

    impacts_s1_path = SENS_DIR / "impacts_sensitivity_S1_passthrough80_A_D.csv"
    checks_s1_path = SENS_DIR / "checks_sensitivity_S1_passthrough80_A_D.json"

    impacts_s1.to_csv(impacts_s1_path, index=False)
    checks_s1_path.write_text(json.dumps(checks_s1, indent=2), encoding="utf-8")

    print("Wrote:", impacts_s1_path)
    print("Wrote:", checks_s1_path)
    # endregion Sensitivity S1: Partial pass-through for bill-based instruments

    # ------------------------------------------------------------------
    # region Sensitivity S2: Higher equivalised-income floor for % income metric
    # ------------------------------------------------------------------
    eqinc_floor_annual = 5_000.0

    # Re-run scenarios with a higher denominator used for liability_share_eqinc
    impacts_A2, checks_A2 = ie.run_scenario(
        df,
        ie.scenario_A_gas_levy(),
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    impacts_B2, checks_B2 = ie.run_scenario(
        df,
        scen_B,
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    impacts_C2, checks_C2 = ie.run_scenario(
        df,
        scen_C,
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    impacts_D2, checks_D2 = ie.run_scenario(
        df,
        scen_D,
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    impacts_E12, checks_E12 = ie.run_scenario(
        df,
        ie.scenario_E1_uniform_charge(),
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    impacts_E22, checks_E22 = ie.run_scenario(
        df,
        ie.scenario_E2_income_tax_proxy(),
        revenue_target=revenue_target,
        eqinc_floor_annual=eqinc_floor_annual,
    )

    # Suffix scenarios
    impacts_A2, checks_A2 = _suffix_scenario(impacts_A2, checks_A2, suffix="_S2_floor5000")
    impacts_B2, checks_B2 = _suffix_scenario(impacts_B2, checks_B2, suffix="_S2_floor5000")
    impacts_C2, checks_C2 = _suffix_scenario(impacts_C2, checks_C2, suffix="_S2_floor5000")
    impacts_D2, checks_D2 = _suffix_scenario(impacts_D2, checks_D2, suffix="_S2_floor5000")
    impacts_E12, checks_E12 = _suffix_scenario(impacts_E12, checks_E12, suffix="_S2_floor5000")
    impacts_E22, checks_E22 = _suffix_scenario(impacts_E22, checks_E22, suffix="_S2_floor5000")

    impacts_s2 = pd.concat(
        [impacts_A2, impacts_B2, impacts_C2, impacts_D2, impacts_E12, impacts_E22],
        ignore_index=True,
    )
    checks_s2 = {"A": checks_A2, "B": checks_B2, "C": checks_C2, "D": checks_D2, "E1": checks_E12, "E2": checks_E22}

    impacts_s2_path = SENS_DIR / "impacts_sensitivity_S2_eqincfloor5000_A_D_E1_E2.csv"
    checks_s2_path = SENS_DIR / "checks_sensitivity_S2_eqincfloor5000_A_D_E1_E2.json"

    impacts_s2.to_csv(impacts_s2_path, index=False)
    checks_s2_path.write_text(json.dumps(checks_s2, indent=2), encoding="utf-8")

    print("Wrote:", impacts_s2_path)
    print("Wrote:", checks_s2_path)
    # endregion Sensitivity S2: Higher equivalised-income floor for % income metric

    # ------------------------------------------------------------------
    # region Sensitivity S3: Standing charges applied to all households (Scenarios B and D)
    # ------------------------------------------------------------------
    # Payer base alternative: treat all households as payers for standing charges.
    df_all = df.copy()
    df_all["payer_all"] = 1

    scen_B_all = ie.ScenarioSpec(
        name="B_elec_standing_charge",
        kind="standing_charge",
        payer_col="payer_all",
        notes="All households treated as payers (payer_all = 1)",
    )
    impacts_B3, checks_B3 = ie.run_scenario(df_all, scen_B_all, revenue_target=revenue_target)

    scen_D_all = ie.ScenarioSpec(
        name="D_gas_standing_charge",
        kind="standing_charge",
        payer_col="payer_all",
        notes="All households treated as payers (payer_all = 1)",
    )
    impacts_D3, checks_D3 = ie.run_scenario(df_all, scen_D_all, revenue_target=revenue_target)

    impacts_B3, checks_B3 = _suffix_scenario(impacts_B3, checks_B3, suffix="_S3_allhh")
    impacts_D3, checks_D3 = _suffix_scenario(impacts_D3, checks_D3, suffix="_S3_allhh")

    impacts_s3 = pd.concat([impacts_B3, impacts_D3], ignore_index=True)
    checks_s3 = {"B": checks_B3, "D": checks_D3}

    impacts_s3_path = SENS_DIR / "impacts_sensitivity_S3_standingcharge_allhouseholds_B_D.csv"
    checks_s3_path = SENS_DIR / "checks_sensitivity_S3_standingcharge_allhouseholds_B_D.json"

    impacts_s3.to_csv(impacts_s3_path, index=False)
    checks_s3_path.write_text(json.dumps(checks_s3, indent=2), encoding="utf-8")

    print("Wrote:", impacts_s3_path)
    print("Wrote:", checks_s3_path)
    # endregion Sensitivity S3: Standing charges applied to all households

    # ------------------------------------------------------------------
    # region Sensitivity S4: Revenue scale invariance (R = £100m)
    # ------------------------------------------------------------------

    revenue_target_big = 100_000_000.0  # £100m

    impacts_A4, checks_A4 = ie.run_scenario(df, ie.scenario_A_gas_levy(), revenue_target=revenue_target_big)
    impacts_B4, checks_B4 = ie.run_scenario(df, scen_B, revenue_target=revenue_target_big)
    impacts_C4, checks_C4 = ie.run_scenario(df, scen_C, revenue_target=revenue_target_big)
    impacts_D4, checks_D4 = ie.run_scenario(df, scen_D, revenue_target=revenue_target_big)
    impacts_E14, checks_E14 = ie.run_scenario(df, ie.scenario_E1_uniform_charge(), revenue_target=revenue_target_big)
    impacts_E24, checks_E24 = ie.run_scenario(df, ie.scenario_E2_income_tax_proxy(), revenue_target=revenue_target_big)

    impacts_A4, checks_A4 = _suffix_scenario(impacts_A4, checks_A4, suffix="_S4_R100m")
    impacts_B4, checks_B4 = _suffix_scenario(impacts_B4, checks_B4, suffix="_S4_R100m")
    impacts_C4, checks_C4 = _suffix_scenario(impacts_C4, checks_C4, suffix="_S4_R100m")
    impacts_D4, checks_D4 = _suffix_scenario(impacts_D4, checks_D4, suffix="_S4_R100m")
    impacts_E14, checks_E14 = _suffix_scenario(impacts_E14, checks_E14, suffix="_S4_R100m")
    impacts_E24, checks_E24 = _suffix_scenario(impacts_E24, checks_E24, suffix="_S4_R100m")

    impacts_s4 = pd.concat([impacts_A4, impacts_B4, impacts_C4, impacts_D4, impacts_E14, impacts_E24], ignore_index=True)
    checks_s4 = {"A": checks_A4, "B": checks_B4, "C": checks_C4, "D": checks_D4, "E1": checks_E14, "E2": checks_E24}

    impacts_s4_path = SENS_DIR / "impacts_sensitivity_S4_revenuescale_R100m_A_D_E1_E2.csv"
    checks_s4_path = SENS_DIR / "checks_sensitivity_S4_revenuescale_R100m_A_D_E1_E2.json"

    impacts_s4.to_csv(impacts_s4_path, index=False)
    checks_s4_path.write_text(json.dumps(checks_s4, indent=2), encoding="utf-8")

    print("Wrote:", impacts_s4_path)
    print("Wrote:", checks_s4_path)
    #endregion Sensitivity S4: Revenue scale invariance

    # ------------------------------------------------------------------
    # region Sensitivity S5: Standing charge proxy values (±20%) for unit-levy netting (Scenarios A and C)
    # ------------------------------------------------------------------
    # Motivation: standing charge proxies are externally sourced and (for electricity) vary materially by Ofgem region.
    # We test robustness to plausible uncertainty by scaling both fuel proxies by ±20%.

    gas_sc_base = float(getattr(ie, "GAS_STANDING_CHARGE_ANNUAL_GBP"))
    elec_sc_base = float(getattr(ie, "ELEC_STANDING_CHARGE_ANNUAL_GBP"))

    impacts_s5_parts = []
    checks_s5 = {}

    for factor, tag in [(0.8, "SC80"), (1.2, "SC120")]:
        # Override standing charge proxies in the incidence engine for this sensitivity run
        ie.GAS_STANDING_CHARGE_ANNUAL_GBP = gas_sc_base * factor
        ie.ELEC_STANDING_CHARGE_ANNUAL_GBP = elec_sc_base * factor

        # Re-run A and C only
        impacts_A5, checks_A5 = ie.run_scenario(df, ie.scenario_A_gas_levy(), revenue_target=revenue_target)
        impacts_C5, checks_C5 = ie.run_scenario(df, scen_C, revenue_target=revenue_target)

        # Add metadata to checks
        checks_A5 = dict(checks_A5)
        checks_C5 = dict(checks_C5)
        checks_A5["standing_charge_factor"] = float(factor)
        checks_C5["standing_charge_factor"] = float(factor)
        checks_A5["gas_standing_charge_annual_gbp"] = float(ie.GAS_STANDING_CHARGE_ANNUAL_GBP)
        checks_C5["elec_standing_charge_annual_gbp"] = float(ie.ELEC_STANDING_CHARGE_ANNUAL_GBP)

        # Suffix scenario names
        impacts_A5, checks_A5 = _suffix_scenario(impacts_A5, checks_A5, suffix=f"_S5_{tag}")
        impacts_C5, checks_C5 = _suffix_scenario(impacts_C5, checks_C5, suffix=f"_S5_{tag}")

        impacts_s5_parts.extend([impacts_A5, impacts_C5])
        checks_s5[f"A_{tag}"] = checks_A5
        checks_s5[f"C_{tag}"] = checks_C5

    # Restore baseline standing charge proxies
    ie.GAS_STANDING_CHARGE_ANNUAL_GBP = gas_sc_base
    ie.ELEC_STANDING_CHARGE_ANNUAL_GBP = elec_sc_base

    impacts_s5 = pd.concat(impacts_s5_parts, ignore_index=True)
    impacts_s5_path = SENS_DIR / "impacts_sensitivity_S5_standingcharge_pm20_A_C.csv"
    checks_s5_path = SENS_DIR / "checks_sensitivity_S5_standingcharge_pm20_A_C.json"

    impacts_s5.to_csv(impacts_s5_path, index=False)
    checks_s5_path.write_text(json.dumps(checks_s5, indent=2), encoding="utf-8")

    print("Wrote:", impacts_s5_path)
    print("Wrote:", checks_s5_path)
    # endregion Sensitivity S5: Standing charge proxy values (±20%)

    # endregion Sensitivity analysis

if __name__ == "__main__":
    main()