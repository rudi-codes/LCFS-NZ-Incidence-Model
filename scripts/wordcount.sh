#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

# Main body sections (exclude appendix by enumerating sections explicitly)
SECTIONS=(
  sections/00_exec_summary.tex
  sections/01_introduction.tex
  sections/02_scope.tex
  sections/03_literature.tex
  sections/04_data.tex
  sections/05_methodology.tex
  sections/06_results.tex
  sections/07_discussion.tex
  sections/08_conclusion.tex
)

# Extract the *final* totals for a given key from texcount output.
# We rely on texcount's stable verbose lines:
#   Words in text: <n>
#   Words in headers: <n>
#   Words outside text (captions, etc.): <n>
extract_last_total() {
  local out key
  out="$1"
  key="$2"
  # Take the last matching line and strip everything but digits
  printf '%s\n' "$out" | grep -E "$key" | tail -n 1 | tr -dc '0-9'
}

# Run texcount and compute total words = text + headers + captions
texcount_total_words() {
  local out text headers outside
  # shellcheck disable=SC2124
  out="$*"
  text=$(extract_last_total "$out" "^Words in text:")
  headers=$(extract_last_total "$out" "^Words in headers:")
  outside=$(extract_last_total "$out" "^Words outside text")

  # Fallback to 0 if any field is empty
  text=${text:-0}
  headers=${headers:-0}
  outside=${outside:-0}

  echo $((text + headers + outside))
}

# Count main body including \input'ed tables within those section files
BODY_OUT=$(texcount -inc "${SECTIONS[@]}")
BODY_WC=$(texcount_total_words "$BODY_OUT")

# Count any words in main.tex itself (title page, headings etc.) without following includes.
# Do NOT use -inc here.
MAIN_OUT=$(texcount main.tex)
MAIN_WC=$(texcount_total_words "$MAIN_OUT")

TOTAL_WC=$((BODY_WC + MAIN_WC))
export TOTAL_WC
# Format with thousands separator for display
TOTAL_WC_FMT=$(python3 - <<'PY'
import os
n = int(os.environ['TOTAL_WC'])
print(f"{n:,}")
PY
)

cat > outputs/wordcount_main.tex <<EOF
\\newcommand{\\MainWordCount}{$TOTAL_WC_FMT}
EOF

echo "Main body word count (excluding appendix): $TOTAL_WC_FMT"
echo "Wrote outputs/wordcount_main.tex with MainWordCount=$TOTAL_WC_FMT"