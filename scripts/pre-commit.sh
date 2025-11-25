#!/usr/bin/env bash
# Run lightweight quality gates on staged Python files (or all files with --all).
# Checks: Ruff lint, docstring enforcement, mypy (ignore missing imports).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    USE_ALL=true
fi

# Ensure git context exists.
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Not a git repository; skipping checks." >&2
    exit 0
fi

# Prefer uvx; fallback to python -m.
if command -v uvx >/dev/null 2>&1; then
    PYRUNNER=(uvx python)
else
    PYRUNNER=(python)
fi

# Collect target Python files.
PY_FILES=()
if [[ "$USE_ALL" == true ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && PY_FILES+=("$line")
    done < <(git ls-files "*.py")
else
    while IFS= read -r line; do
        [[ -n "$line" ]] && PY_FILES+=("$line")
    done < <(git diff --cached --name-only --diff-filter=AM | grep -E '\.py$' || true)
fi

if [[ ${#PY_FILES[@]} -eq 0 ]]; then
    echo "No Python files to check."
    exit 0
fi

echo "Checking ${#PY_FILES[@]} Python files..."

# Ruff lint
"${PYRUNNER[@]}" -m ruff check "${PY_FILES[@]}"

# Docstring enforcement
"${PYRUNNER[@]}" scripts/check_docstrings.py "${PY_FILES[@]}"

# Mypy (ignore missing imports to accommodate external deps)
"${PYRUNNER[@]}" -m mypy --config-file mypy.ini "${PY_FILES[@]}"

echo "All checks passed."
