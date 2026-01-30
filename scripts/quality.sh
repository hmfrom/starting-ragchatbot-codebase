#!/bin/bash
# Run all quality checks

set -e

echo "=== Running Code Quality Checks ==="
echo ""

echo "1. Checking code formatting (black)..."
uv run black --check .
echo "   ✓ Formatting check passed"
echo ""

echo "=== All Quality Checks Passed ==="
