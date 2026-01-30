#!/bin/bash
# Format all Python files using black

set -e

echo "Formatting Python files with black..."
uv run black .

echo "Done! All files formatted."
