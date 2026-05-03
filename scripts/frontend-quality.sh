#!/usr/bin/env bash
# Run frontend code quality checks.
# Usage:
#   ./scripts/frontend-quality.sh          # check only (CI mode)
#   ./scripts/frontend-quality.sh --fix    # auto-fix formatting and linting

set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "$0")/../frontend" && pwd)"

cd "$FRONTEND_DIR"

if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

if [ "${1:-}" = "--fix" ]; then
  echo "==> Formatting with Prettier..."
  npm run format

  echo "==> Linting with ESLint (auto-fix)..."
  npm run lint:fix

  echo "All frontend quality fixes applied."
else
  echo "==> Checking formatting with Prettier..."
  npm run format:check

  echo "==> Linting with ESLint..."
  npm run lint

  echo "All frontend quality checks passed."
fi
