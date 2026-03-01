""" Decription:
LCFS 2022 to 2023 (UKDS 9335).
From UK Data Service: https://datacatalogue.ukdataservice.ac.uk/studies/study/9335 (stata files)

Builds a household level analysis dataset for distributional incidence modelling.
2022-23 chosen for consistency with internal Ofgem modelling. Ambition to pool further years in future if data and resources allow.

Outputs:
- Household analysis dataset (1 row per household) with identifiers, weights, income measures,
  deciles, region, urban rural flag, payment method indicators, and gas and electricity
  expenditure bases (weekly and annualised).
- Wider diagnostics dataset retaining intermediate components used to construct expenditure bases.
- Plain text build log with QA summaries.

Annualisation:
- Weekly values are annualised using 365/7.

Notes:
- This script does not perform validation against any external processed sample.
- EPC variables are intentionally excluded, due to significant data issues. English housing survey would be needed to accurately extract EPC data.
"""

#region Package Imports
from __future__ import annotations # Handles expected data types
from pathlib import Path
import numpy as np
import pandas as pd
#endregion Package Imports

# -----------------------------------------------------------------------------
# region Paths and constants
# -----------------------------------------------------------------------------
DATA_DIR = Path("data/lcfs_2022_23")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DVHH_PATH = DATA_DIR / "dvhh_ukanon_2022.dta"
URBAN_PATH = DATA_DIR / "dvhh_urbanrural_ukanon_2022.dta"

# Annualisation factor for weekly to annual values (365/7).
ANNUAL_FACTOR = 365.0 / 7.0

# Urban rural indicators (from dvhh_urbanrural_ukanon_2022):
# - URGridEWp: Urban/Rural Indicator England/Wales (1=Urban, 2=Rural)
# - URGridSCp: Urban/Rural Indicator Scotland (1=Urban, 2=Rural)
URGRID_EW_COL = "URGridEWp"
URGRID_SC_COL = "URGridSCp"

# GOR code used in this script (region_gor):
GOR_SCOTLAND = 11
GOR_NI = 12
#endregion Paths and constants

# -----------------------------------------------------------------------------
# region Build Output Log
# -----------------------------------------------------------------------------
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(str(msg))


def write_log() -> None:
    log_path = OUT_DIR / "lcfs_log_checks.txt"
    log_path.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"Wrote build log: {log_path}")
#endregion Build Output Log

# -----------------------------------------------------------------------------
# region Helper Functions
# -----------------------------------------------------------------------------

# Candidate column names and returns ones that exist in dataframe (case-insensitive)
# Used as many columns and complex naming across LCFS datasets, so majorly helpful during data exploration phase, retained here for ease.
def present(df: pd.DataFrame, names: list[str]) -> list[str]:
    """Return candidate names present in df (case-insensitive), preserving order."""
    lower_map = {c.lower(): c for c in df.columns}
    out: list[str] = []
    for n in names:
        if n in df.columns:
            out.append(n)
        elif n.lower() in lower_map:
            out.append(lower_map[n.lower()])
    seen = set()
    dedup: list[str] = []
    for c in out:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup

# Converts a pandas Series to numeric, coercing non-numeric values to NaN
# Errors coerced to numeric are handled later in QA checks, to avoid silent data issues from unexpected non-numeric values.
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# Sums across columns, but only for those that exist and treats missing values as zero.
def safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return block.sum(axis=1)

# Computes weighted deciles of a variable using survey weights
def weighted_deciles(x: pd.Series, w: pd.Series, *, n: int = 10) -> pd.Series:
    """Return 1..n weighted deciles of x using weights w (NA where x or w missing)."""
    x_num = to_num(x)
    w_num = to_num(w)
    mask = x_num.notna() & w_num.notna() & (w_num > 0)
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    if mask.sum() == 0:
        return out

    df = pd.DataFrame({"x": x_num[mask], "w": w_num[mask]}).sort_values("x")
    cumw = df["w"].cumsum()
    total = float(df["w"].sum())
    if total <= 0: #total should be strictly positive.
        return out

    cuts = [(k / n) * total for k in range(1, n)]
    df["decile"] = (np.searchsorted(cuts, cumw, side="right") + 1).astype(int)
    out.loc[df.index] = pd.Series(df["decile"].values, index=df.index, dtype="Int64")
    return out
#endregion Helper Functions

# -----------------------------------------------------------------------------
# region Data loading and processing
# -----------------------------------------------------------------------------
def main() -> None:
    log("Reading LCFS files")

    if not DVHH_PATH.exists():
        raise FileNotFoundError(f"DVHH file not found: {DVHH_PATH}")
    if not URBAN_PATH.exists():
        raise FileNotFoundError(f"Urban rural file not found: {URBAN_PATH}")

    # convert_categoricals=False avoids issues from non-unique labels
    dvhh = pd.read_stata(DVHH_PATH, convert_categoricals=False)
    urban = pd.read_stata(URBAN_PATH, convert_categoricals=False)

    #Ensure ID variable present in datasets.
    if "case" not in dvhh.columns:
        raise ValueError("Expected household identifier 'case' not found in DVHH.")
    if "case" not in urban.columns:
        raise ValueError("Expected household identifier 'case' not found in urban rural file.")

    #Check data has been read in correctly, no missing rows.
    log(f"dvhh shape: {dvhh.shape}")
    log(f"dvhh unique case: {dvhh['case'].is_unique}")
    log(f"urban shape: {urban.shape}")
    log(f"urban unique case: {urban['case'].is_unique}")

    # Core DVHH variables at household level.
    # p550tp is total household expenditure (weekly) as provided in the DVHH derived variables.
    core_cols = [
        "case",
        "weighta",
        "anon_income",
        # "p389p",  # normal weekly disposable household income (after tax and NI)
        "EqIncDMp",
        "EqIncDOp",
        "gorx",
        "p550tp",
        "p344p",  # gross normal weekly household income (top-coded)
        # tenure
        "a121",
        # household composition (LCFS DVHH A-codes)
        "a040",  # children under 2
        "a041",  # children 2-4
        "a042",  # children 5-17
        "a043",  # adults 18-44
        "a044",  # adults 45-59
        "a045",  # adults 60-64
        "a046",  # adults 65-69
        "a047",  # adults 70+
        "a049",  # household size
        "a062",  # household composition code
    ]

    # Disposable household income (normal weekly) is P389. In this DVHH extract it is stored as `P389p`.
    DISP_COL = "P389p"
    core_cols.append(DISP_COL)
    missing_core = [c for c in core_cols if c not in dvhh.columns]
    if missing_core:
        raise ValueError(f"Missing expected core variables in DVHH: {missing_core}")

    hh = dvhh[core_cols].copy()
    #rename columns for ease
    hh = hh.rename(
        columns={
            "weighta": "weight_annual",
            "anon_income": "income_anon_weekly",  # anonymised income measure (not treated as disposable)
            DISP_COL: "income_disposable_weekly",
            "EqIncDMp": "income_equivalised_weekly_eqincdmp",
            "EqIncDOp": "income_equivalised_weekly_eqincdop",
            "gorx": "region_gor",
            "p550tp": "total_expenditure_weekly",
            "p344p": "income_gross_weekly",
            "a121": "tenure_raw",
            "a040": "n_children_u2",
            "a041": "n_children_2_4",
            "a042": "n_children_5_17",
            "a043": "n_adults_18_44",
            "a044": "n_adults_45_59",
            "a045": "n_adults_60_64",
            "a046": "n_adults_65_69",
            "a047": "n_adults_70_plus",
            "a049": "hh_size",
            "a062": "hh_composition_raw",
        }
    )

    # Region labels
    GOR_LABELS = {
        1: "north east",
        2: "north west and merseyside",
        3: "yorkshire and the humber",
        4: "east midlands",
        5: "west midlands",
        6: "east of england",
        7: "london",
        8: "south east",
        9: "south west",
        10: "wales",
        11: "scotland",
        12: "northern ireland",
    }
    hh["region_label"] = to_num(hh["region_gor"]).map(GOR_LABELS)

    # Total household expenditure (weekly -> annualised) for expenditure-share metrics
    hh["total_expenditure_annualised"] = to_num(hh["total_expenditure_weekly"]) * ANNUAL_FACTOR

    # Gross normal household income (weekly -> annualised). Note: p344p is top-coded in the released data.
    hh["income_gross_annualised"] = to_num(hh["income_gross_weekly"]) * ANNUAL_FACTOR

    # Disposable household income (weekly -> annualised). Derived as P389 in LCFS documentation.
    hh["income_disposable_annualised"] = to_num(hh["income_disposable_weekly"]) * ANNUAL_FACTOR

    # Household composition
    hh["n_children"] = (
        to_num(hh["n_children_u2"]).fillna(0)
        + to_num(hh["n_children_2_4"]).fillna(0)
        + to_num(hh["n_children_5_17"]).fillna(0)
    )

    hh["n_adults"] = (
        to_num(hh["n_adults_18_44"]).fillna(0)
        + to_num(hh["n_adults_45_59"]).fillna(0)
        + to_num(hh["n_adults_60_64"]).fillna(0)
        + to_num(hh["n_adults_65_69"]).fillna(0)
        + to_num(hh["n_adults_70_plus"]).fillna(0)
    )

    # Check: DVHH household size should match adults+children, hh_size_calc is checked later against hh_size in QA.
    hh["hh_size_calc"] = hh["n_adults"] + hh["n_children"]
    hh["has_children"] = (hh["n_children"] > 0).astype(int)

    # Pensioner household flag: at least one adult aged 65+ (assumption)
    hh["pensioner_household"] = (
        (to_num(hh["n_adults_65_69"]).fillna(0) + to_num(hh["n_adults_70_plus"]).fillna(0)) > 0
    ).astype(int)

    # Tenure (DVHH a121: Tenure type)
    # Mapping is based on LCFS DVHH conventions.
    tenure_map = {
        0: "Not recorded",
        1: "Social rent (local authority)",
        2: "Social rent (housing association)",
        3: "Private rent (unfurnished)",
        4: "Private rent (furnished)",
        5: "Owner occupier (mortgage)",
        6: "Owner occupier (rental purchase)",
        7: "Owner occupier (outright)",
        8: "Rent free",
    }

    hh["tenure_label"] = to_num(hh["tenure_raw"]).map(tenure_map)

    #Simpliy tenure categories to 4 groups: owner occupier, social rent, private rent, rent free, with unknown/other category for missing and uncategorisable cases.
    def collapse_tenure(v) -> str:
        if pd.isna(v) or v == 0:
            return "Unknown"
        v = int(v)
        if v in (5, 6, 7):
            return "Owner occupier"
        if v in (1, 2):
            return "Social rent"
        if v in (3, 4):
            return "Private rent"
        if v == 8:
            return "Rent free"
        return "Other"

    hh["tenure_4cat"] = to_num(hh["tenure_raw"]).apply(collapse_tenure)

    # Household composition (DVHH a062: Composition of household)
    # Categories taken from the DVHH data dictionary and derived variable flowcharts.
    hh_comp_map = {
        1: "1 man",
        2: "1 woman",
        3: "1 man and 1 child",
        4: "1 woman and 1 child",
        5: "1 man and 2+ children",
        6: "1 woman and 2+ children",
        7: "1 man and 1 woman",
        8: "2 men or 2 women",
        9: "1 man, 1 woman and 1 child",
        10: "2 men or 2 women and 1 child",
        11: "1 man, 1 woman and 2 children",
        12: "2 men or 2 women and 2 children",
        13: "1 man, 1 woman and 3 children",
        14: "2 men or 2 women and 3 children",
        15: "2 adults and 4 children",
        16: "2 adults and 5 children",
        17: "2 adults and 6+ children",
        18: "3 adults",
        19: "3 adults and 1 child",
        20: "3 adults and 2 children",
        21: "3 adults and 3 children",
        22: "3 adults and 4+ children",
        23: "4 adults",
        24: "4 adults and 1 child",
        25: "4 adults and 2+ children",
        26: "5 adults",
        27: "5 adults and 1+ children",
        28: "6+ adults",
        29: "children only",
        30: "other households with children",
    }

    hh["hh_composition_label"] = to_num(hh["hh_composition_raw"]).map(hh_comp_map)
    #endregion Data loading and processing

    # -------------------------------------------------------------------------
    # region Urban rural merge
    # -------------------------------------------------------------------------
    missing_ur = [c for c in [URGRID_EW_COL, URGRID_SC_COL] if c not in urban.columns]
    if missing_ur:
        raise ValueError(
            "Expected urban/rural indicator columns not found in the urban-rural file: "
            f"{missing_ur}. Available columns (excluding 'case'): {[c for c in urban.columns if c != 'case']}"
        )

    # Merge both columns onto the household spine
    urban_small = urban[["case", URGRID_EW_COL, URGRID_SC_COL]].copy()
    hh = hh.merge(urban_small, on="case", how="left", validate="1:1")

    # Construct a single urban/rural indicator:
    # - For Scotland (GOR=11), use URGridSCp
    # - Otherwise, use URGridEWp (England/Wales)
    # - For Northern Ireland (GOR=12), set to missing
    reg = to_num(hh["region_gor"]).astype("Int64")
    ur_ew = to_num(hh[URGRID_EW_COL]).astype("Int64")
    ur_sc = to_num(hh[URGRID_SC_COL]).astype("Int64")

    hh["rurality_raw"] = pd.Series(pd.NA, index=hh.index, dtype="Int64")
    hh.loc[reg == GOR_SCOTLAND, "rurality_raw"] = ur_sc[reg == GOR_SCOTLAND]
    hh.loc[(reg != GOR_SCOTLAND) & (reg != GOR_NI), "rurality_raw"] = ur_ew[(reg != GOR_SCOTLAND) & (reg != GOR_NI)]

    hh["rurality"] = to_num(hh["rurality_raw"]).astype("Int64")

    # Coding per UKDA dictionary: 1=Urban, 2=Rural (for both EW and Scotland indicators)
    RURALITY_LABELS = {1: "urban", 2: "rural"}
    hh["rurality_label"] = to_num(hh["rurality"]).map(RURALITY_LABELS)

    log("\nQA: rurality coverage by region (share missing rurality_raw)")
    tmp = hh[["region_label", "rurality_raw"]].copy()
    miss_by_region = tmp.groupby("region_label")["rurality_raw"].apply(lambda s: s.isna().mean()).sort_index()
    log(str(miss_by_region))
    #endregion Urban rural merge

    # -------------------------------------------------------------------------
    # region Energy expenditure variables (weekly and annualised)
    # -------------------------------------------------------------------------
    """Some households have energy spend recorded via a direct route (bill or account amount), while others only have component items recorded. Prefer the direct route when observed, otherwise fall back to the component sum (expanded route), to maximise coverage. The chosen route is stored in `energy_method` for QA."""
    # Direct route variables
    GAS_DIRECT = present(dvhh, ["b170", "b1701"])
    ELEC_DIRECT = present(dvhh, ["b175", "b1751"])

    # Expanded route variables
    GAS_EXPANDED_COMPONENTS = present(dvhh, ["b490", "b226", "b235", "b255"])
    ELEC_EXPANDED_COMPONENTS = present(dvhh, ["b489", "b227", "b234", "b254"])

    # Pull required variables and payment method variables
    energy_cols = present(
        dvhh,
        [
            # direct and expanded components
            "b170", "b1701", "b175", "b1751",
            "b490", "b489", "b226", "b227", "b234", "b235", "b254", "b255",
            # payment method codes
            "a128", "a129", "a130",
        ],
    )

    if energy_cols:
        hh = hh.merge(dvhh[["case"] + energy_cols], on="case", how="left", validate="1:1")

    hh["gas_expenditure_weekly_direct"] = safe_sum(hh, GAS_DIRECT)
    hh["electricity_expenditure_weekly_direct"] = safe_sum(hh, ELEC_DIRECT)

    hh["gas_expenditure_weekly_expanded"] = safe_sum(hh, GAS_EXPANDED_COMPONENTS)
    hh["electricity_expenditure_weekly_expanded"] = safe_sum(hh, ELEC_EXPANDED_COMPONENTS)

    # Route selection
    hh["energy_method"] = np.where(
        (hh["gas_expenditure_weekly_direct"] > 0) | (hh["electricity_expenditure_weekly_direct"] > 0),
        "direct",
        "expanded",
    )

    hh["gas_expenditure_weekly"] = np.where(
        hh["energy_method"] == "direct",
        hh["gas_expenditure_weekly_direct"],
        hh["gas_expenditure_weekly_expanded"],
    )

    hh["electricity_expenditure_weekly"] = np.where(
        hh["energy_method"] == "direct",
        hh["electricity_expenditure_weekly_direct"],
        hh["electricity_expenditure_weekly_expanded"],
    )

    hh["gas_expenditure_annualised"] = hh["gas_expenditure_weekly"] * ANNUAL_FACTOR
    hh["electricity_expenditure_annualised"] = hh["electricity_expenditure_weekly"] * ANNUAL_FACTOR

    # Payment method (Raw codes) - NA if missing
    hh["gas_mop_raw"] = to_num(hh["a128"]).astype("Int64") if "a128" in hh.columns else pd.Series(pd.NA, index=hh.index, dtype="Int64")
    hh["elec_mop_raw"] = to_num(hh["a130"]).astype("Int64") if "a130" in hh.columns else pd.Series(pd.NA, index=hh.index, dtype="Int64")
    hh["combined_mop"] = to_num(hh["a129"]).astype("Int64") if "a129" in hh.columns else pd.Series(pd.NA, index=hh.index, dtype="Int64")

    # Use fuel-specific method of payment where available; only fall back to the combined code when the fuel-specific value is missing. This avoids overwriting PPM codes present in a128/a130.
    hh["gas_mop_eff"] = hh["gas_mop_raw"].where(hh["gas_mop_raw"].notna(), hh["combined_mop"]).astype("Int64")
    hh["elec_mop_eff"] = hh["elec_mop_raw"].where(hh["elec_mop_raw"].notna(), hh["combined_mop"]).astype("Int64")

    # Simplified coding: 3 = prepayment, 1 = otherwise, NA if missing
    hh["gas_mop"] = np.where(
        hh["gas_mop_eff"].fillna(0) == 3,
        3,
        np.where(hh["gas_mop_eff"].notna(), 1, pd.NA),
    )
    hh["elec_mop"] = np.where(
        hh["elec_mop_eff"].fillna(0) == 3,
        3,
        np.where(hh["elec_mop_eff"].notna(), 1, pd.NA),
    )
    hh["gas_mop"] = pd.Series(hh["gas_mop"], index=hh.index).astype("Int64")
    hh["elec_mop"] = pd.Series(hh["elec_mop"], index=hh.index).astype("Int64")

    # Flags based on the simplified coding
    hh["ppm"] = ((hh["gas_mop_eff"] == 3) | (hh["elec_mop_eff"] == 3)).astype(int)
    hh["direct_debit"] = ((hh["gas_mop_eff"] == 1) | (hh["elec_mop_eff"] == 1)).astype(int)
    #endregion Energy expenditure variables (weekly and annualised)

    # -------------------------------------------------------------------------
    # region Deciles
    # -------------------------------------------------------------------------
    # Construct income deciles (could change code so you can change n)
    hh["income_decile_eqincdmp"] = weighted_deciles(
        hh["income_equivalised_weekly_eqincdmp"], hh["weight_annual"], n=10
    )

    bad_deciles = hh["income_decile_eqincdmp"].notna() & ~hh["income_decile_eqincdmp"].between(1, 10)
    if bad_deciles.any():
        n_bad = int(bad_deciles.sum())
        raise ValueError(f"Found {n_bad} households with income_decile_eqincdmp outside 1..10.")
    #endregion Deciles

    # -------------------------------------------------------------------------
    # region QA checks
    # -------------------------------------------------------------------------
    # Check ID is unique (no duplicate households)
    if not hh["case"].is_unique:
        dup_n = int(hh["case"].duplicated().sum())
        raise ValueError(f"Household identifier 'case' is not unique in final dataset (duplicates={dup_n}).")

    # Compute data completesness for key variables
    log("\nQA: Missing data rates (share missing)")
    for c in [
        "weight_annual",
        "income_equivalised_weekly_eqincdmp",
        "income_decile_eqincdmp",
        "tenure_raw",
        "tenure_4cat",
        "hh_size",
        "n_adults",
        "n_children",
        "has_children",
        "pensioner_household",
        "hh_composition_raw",
        "total_expenditure_weekly",
        "total_expenditure_annualised",
        "income_gross_weekly",
        "income_gross_annualised",
        "income_disposable_weekly",
        "income_disposable_annualised",
        "gas_expenditure_annualised",
        "electricity_expenditure_annualised",
    ]:
        log(f"- {c}: {hh[c].isna().mean():.3f}")

    # Check sum of weights is in expected range (should be close to number of households in population, around 28 million for UK in this year, weight give 28,000 so multiply by 1000 for population)
    w_sum = to_num(hh["weight_annual"]).sum()
    log(f"QA: sum of household weights (weight_annual) = {w_sum:,.2f}")

    # Income check: disposable income should not systematically exceed gross income
    both = to_num(hh["income_disposable_weekly"]).notna() & to_num(hh["income_gross_weekly"]).notna()
    if both.any():
        share_disp_gt_gross = (to_num(hh.loc[both, "income_disposable_weekly"]) > to_num(hh.loc[both, "income_gross_weekly"])).mean()
        log(f"QA: share with disposable_weekly > gross_weekly: {share_disp_gt_gross:.3f}")

    # Tenure check: unweighted counts for categories
    log("QA: tenure_4cat value counts (unweighted)")
    log(str(hh["tenure_4cat"].value_counts(dropna=False)))

    log("QA: hh_composition_raw value counts (unweighted, with labels)")
    comp_counts = (
        hh[["hh_composition_raw", "hh_composition_label"]]
        .assign(hh_composition_raw=lambda d: to_num(d["hh_composition_raw"]).astype("Int64"))
        .value_counts(dropna=False)
        .reset_index(name="count")
        .sort_values(["hh_composition_raw", "count"], ascending=[True, False])
    )
    log(str(comp_counts))

    # Household size consistency check: share where DVHH hh_size differs from adults+children
    size_diff = (to_num(hh["hh_size"]) != to_num(hh["hh_size_calc"])) & to_num(hh["hh_size"]).notna()
    log(f"QA: share with hh_size != (n_adults + n_children): {size_diff.mean():.3f}")

    # Expenditure check: Share with no income recorded
    share_zero_totexp = (to_num(hh["total_expenditure_annualised"]).fillna(0) == 0).mean()
    log(f"QA: share of households with total_expenditure_annualised equal to 0: {share_zero_totexp:.3f}")

    # Elec & Gas Expenditure check: share with no expenditure recorded
    share_zero_gas = (to_num(hh["gas_expenditure_annualised"]).fillna(0) == 0).mean()
    share_zero_elec = (to_num(hh["electricity_expenditure_annualised"]).fillna(0) == 0).mean()
    log(f"QA: share of households with gas_expenditure_annualised equal to 0: {share_zero_gas:.3f}")
    log(f"QA: share of households with electricity_expenditure_annualised equal to 0: {share_zero_elec:.3f}")

    # Compure perctiles for expenditure to check outliers
    for c in ["gas_expenditure_annualised", "electricity_expenditure_annualised", "total_expenditure_annualised"]:
        s = to_num(hh[c])
        p50, p90, p99 = s.quantile([0.5, 0.9, 0.99]).tolist()
        log(f"QA: {c} percentiles (p50/p90/p99) = {p50:.2f} / {p90:.2f} / {p99:.2f}")
    #endregion QA checks

    # -------------------------------------------------------------------------
    # region Outputs
    # -------------------------------------------------------------------------
    # Final dataframe
    contract_cols = [
        "case",
        "weight_annual",
        "income_anon_weekly",
        "income_gross_weekly",
        "income_gross_annualised",
        "income_disposable_weekly",
        "income_disposable_annualised",
        "income_equivalised_weekly_eqincdmp",
        "income_equivalised_weekly_eqincdop",
        "income_decile_eqincdmp",
        "tenure_raw",
        "tenure_label",
        "tenure_4cat",
        "hh_size",
        "n_adults",
        "n_children",
        "has_children",
        "pensioner_household",
        "hh_composition_raw",
        "hh_composition_label",
        "total_expenditure_weekly",
        "total_expenditure_annualised",
        "region_gor",
        "region_label",
        "rurality",
        "rurality_raw",
        "rurality_label",
        "energy_method",
        "gas_mop",
        "elec_mop",
        "direct_debit",
        "ppm",
        "gas_expenditure_weekly",
        "gas_expenditure_annualised",
        "electricity_expenditure_weekly",
        "electricity_expenditure_annualised",
    ]

    missing_contract = [c for c in contract_cols if c not in hh.columns]
    if missing_contract:
        raise ValueError(f"Missing expected contract columns prior to export: {missing_contract}")

    clean_path = OUT_DIR / "lcfs_2022_23_HH_analysis.csv"
    hh[contract_cols].to_csv(clean_path, index=False)
    log(f"\nWrote analysis output: {clean_path} (rows={len(hh):,}, cols={len(contract_cols)})")

    full_path = OUT_DIR / "lcfs_2022_23_HH_full.csv"
    hh.to_csv(full_path, index=False)
    log(f"Wrote full output: {full_path} (rows={len(hh):,}, cols={hh.shape[1]})")

    write_log()
     #endregion Outputs

if __name__ == "__main__":
    main()