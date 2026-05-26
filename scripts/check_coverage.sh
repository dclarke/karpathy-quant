#!/usr/bin/env bash
# Run unit tests with coverage, fail if below threshold.
# Usage: ./scripts/check_coverage.sh [min_coverage]  (default: 90)

set -euo pipefail

MIN_COV="${1:-90}"

# find python
PYTHON=".venv/Scripts/python"
[ -f "$PYTHON" ] || PYTHON=".venv/bin/python"
[ -f "$PYTHON" ] || PYTHON="python"

echo "=== Running tests (threshold: ${MIN_COV}%) ==="
echo ""

COV_OUTPUT=$($PYTHON -m pytest tests/unit \
    --cov=karpathy_quant \
    --cov-report=term-missing \
    --cov-fail-under="$MIN_COV" \
    -q 2>&1) || TEST_FAILED=1

echo "$COV_OUTPUT"
echo ""

# label modules
echo "=== Module Status ==="
echo ""
echo "$COV_OUTPUT" | grep -E "^src" | while IFS= read -r line; do
    pct=$(echo "$line" | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    module=$(echo "$line" | awk '{print $1}')
    short=$(echo "$module" | sed 's|src\\karpathy_quant\\||' | sed 's|src/karpathy_quant/||' | sed 's|\.py||')

    if echo "$short" | grep -qE "providers.gemini" && [ "$pct" -lt 100 ]; then
        printf "  %-40s %3s%%  [API calls untested]\n" "$short" "$pct"
    else
        printf "  %-40s %3s%%  [OK]\n" "$short" "$pct"
    fi
done

echo ""
if [ "${TEST_FAILED:-0}" = "1" ]; then
    echo "FAIL: coverage below ${MIN_COV}%"
    exit 1
else
    echo "PASS"
fi
