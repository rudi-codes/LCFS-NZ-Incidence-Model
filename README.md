# Net Zero funding incidence model (GB)

This repository contains Python code for a static, partial-equilibrium incidence model that compares how a fixed annual revenue requirement could be recovered from Great Britain households under alternative funding mechanisms. The focus is on distributional exposure (who pays) rather than forecasting realised bill impacts.

## What this repository includes

- Core model code in `src/` (scenario construction, calibration to a revenue target, and household-level incidence calculations)
- Helper scripts in `scripts/` (a lightweight pipeline runner)

## What this repository does not include

- The written report and LaTeX sources
- Any microdata, derived datasets, or model outputs

The original analysis used the Living Costs and Food Survey (LCFS) 2022–2023, which is licensed microdata. To reproduce results, users must obtain access independently via the UK Data Service and comply with the relevant licence terms.

## Running the code

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline script:

```bash
bash scripts/pipeline.sh
```

## Notes on interpretation

- This is a comparative accounting model calibrated to a fixed revenue requirement.
- It does not estimate behavioural responses, supplier pricing strategies, or general equilibrium effects.
- Outputs should be interpreted as first-order distributional comparisons under transparent assumptions.

## Licence

MIT (see `LICENSE`).