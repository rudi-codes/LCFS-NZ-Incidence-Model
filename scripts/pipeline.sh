#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline: data build -> incidence -> tables -> figures.
# Optional: build report PDF (set BUILD_PDF=1).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_PDF="${BUILD_PDF:-0}"

mkdir -p build

# 1) Build household dataset
"$PYTHON_BIN" src/lcfs_load.py

# 2) Run incidence scenarios + sensitivities
"$PYTHON_BIN" src/run_incidence.py

# 3) Generate report-facing LaTeX tables + results pack
"$PYTHON_BIN" src/results_tables.py

# 4) Generate figures (writes into outputs/results_pack/figures)
"$PYTHON_BIN" src/results_figures.py

# 5) Optional: build the PDF into ./build and copy to repo root
# Set BUILD_PDF=1 to enable. Default is off to keep the pipeline focused on outputs.
if [[ "$BUILD_PDF" == "1" ]]; then
  # Clean only LaTeX intermediates to avoid stale/malformed .bcf/.bbl
  rm -f \
    build/main.aux build/main.bcf build/main.blg build/main.bbl build/main.bbl-SAVE-ERROR \
    build/main.log build/main.out build/main.run.xml build/main.toc build/main.synctex.gz

  # First LaTeX pass must generate build/main.bcf for Biber.
  pdflatex -halt-on-error -interaction=nonstopmode -synctex=1 -output-directory=build main.tex

  if [[ ! -s build/main.bcf ]]; then
    echo "ERROR: build/main.bcf was not produced by pdflatex; cannot run biber." >&2
    echo "--- tail of build/main.log ---" >&2
    tail -n 120 build/main.log >&2 || true
    exit 1
  fi

  biber --input-directory=build --output-directory=build main

  # Two more LaTeX passes to resolve citations/cross-refs.
  pdflatex -halt-on-error -interaction=nonstopmode -synctex=1 -output-directory=build main.tex
  pdflatex -halt-on-error -interaction=nonstopmode -synctex=1 -output-directory=build main.tex

  echo "Copying PDF into repo root: $(pwd)/main.pdf"
  cp -f build/main.pdf main.pdf
else
  echo "BUILD_PDF=0: skipping LaTeX compilation. Outputs are up to date in ./outputs."
fi