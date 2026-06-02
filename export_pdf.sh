#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="${ROOT_DIR}/poster.html"
OUT_FILE="${1:-${ROOT_DIR}/poster-69x104cm.pdf}"

CHROME_BIN="${CHROME_BIN:-}"
if [[ -z "${CHROME_BIN}" ]]; then
  for candidate in \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"; do
    if [[ -x "${candidate}" ]]; then
      CHROME_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CHROME_BIN}" || ! -x "${CHROME_BIN}" ]]; then
  echo "No Chromium/Chrome binary found. Set CHROME_BIN=/path/to/chrome." >&2
  exit 1
fi

"${CHROME_BIN}" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="${OUT_FILE}" \
  "file://${HTML_FILE}"

echo "Wrote ${OUT_FILE}"
echo "PDF page size is defined in poster.html: 69 cm x 104 cm, portrait."
