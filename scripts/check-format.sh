#!/bin/bash
# Check formatting without making changes

set -e

echo "Checking Python formatting with black..."
uv run black --check .

echo "All files are properly formatted!"
