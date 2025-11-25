"""Docstring enforcement for module and public symbols.

Checks that each Python file has a module docstring and that top-level
public classes and functions (names not starting with "_") include docstrings.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable


def has_docstring(node: ast.AST) -> bool:
    """Return True if the AST node has a docstring."""
    return ast.get_docstring(node) is not None


def iter_public_toplevel(nodes: Iterable[ast.stmt]) -> Iterable[ast.AST]:
    """Yield public top-level classes and functions."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            yield node


def check_file(path: Path) -> list[str]:
    """Return a list of error messages for docstring violations in the file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error ({exc})"]

    errors: list[str] = []
    if not has_docstring(tree):
        errors.append(f"{path}: missing module docstring")

    for node in iter_public_toplevel(tree.body):
        if not has_docstring(node):
            kind = type(node).__name__.replace("Def", "").lower()
            errors.append(f"{path}: missing docstring on {kind} '{node.name}'")
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check for required docstrings.")
    parser.add_argument("files", nargs="+", help="Python files to check")
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for file_path in args.files:
        path = Path(file_path)
        if not path.is_file():
            continue
        all_errors.extend(check_file(path))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
