# Coding Steps and Working Agreements

Goal: keep work in small, reviewable slices so git becomes a tree of "saved games" we can time travel through.

## Workflow Loop
1. **Branch & Tag**: from `main`/HEAD create a feature branch named for the concept (e.g., `feature/data-grid-generation`). Create a matching tag that points to HEAD of the branch (e.g., `git tag -f feature/data-grid-generation`). Update the tag to point to HEAD after each commit.
2. **Small move**: pick the smallest atomic change that produces forward progress (code, config, asset, or test update).
3. **Implement with safety**:
   - Follow `CODE_QUALITY.md` (typing, low complexity, idiomatic Python).
   - Add/adjust docstrings and inline docs when behavior is non-obvious.
   - Keep functions pure when possible; isolate I/O.
4. **Tests first/with change**: write or update unit tests to cover the change.
5. **Run quality gates**:
   - Use `uvx` for Python invocations (e.g., `uvx python -m ruff check src tests`, `uvx python -m mypy src`, `uvx python -m unittest ...`).
   - `./scripts/pre-commit.sh` (staged) or `./scripts/pre-commit.sh --all` / `make build` (all tracked) to run Ruff + MyPy and enforce `CODE_QUALITY.md` presence.
   - Pre-commit also enforces module and public function/class docstrings via `scripts/check_docstrings.py`.
   - Add any project-specific checks relevant to the change.
6. **Verify outputs**: inspect generated artifacts or screenshots and ensure they match intent.
   - After visual/rendering changes, regenerate a test output (e.g., `./scripts/generate.sh --dataset-name sanity --test`) so humans can visually inspect updated results.
7. **Commit & Tag**: commit the atomic change with a focused message. Ensure the tree compiles, tests pass, and lint/type checks are clean. Update the feature tag to point to HEAD: `git tag -f <branch-name>`.
8. **Repeat**: pick the next smallest move and loop.

## Additional Notes
- Every step should be shippable: buildable, documented, typed, and lint-clean.
- Keep configs (e.g., dataset sizes, grid definitions) under version control so changes are traceable.
- Prefer reproducible scripts for any manual extraction (column geometry, data sampling) to avoid drift.
- When adding assets, note their dimensions/usage in relevant plans or docs.
- If a step grows, split it; smaller commits make code review and debugging easier.

## Repo-local tooling
- `scripts/pre-commit.sh` runs Ruff + docstring checks + mypy on staged files (use `--all` to scan the repo).
- `mypy.ini` enforces typed defs; missing imports are ignored for external deps, so add stubs where possible.
