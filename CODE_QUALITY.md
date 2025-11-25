# Python Code Quality Guidelines

This document defines **non-negotiable guardrails** for Python code in this project. The goal is to make the codebase:

* Easy to understand on first read
* Easy to change without fear
* Hard to break accidentally

Assume many engineers will cycle through this project. Your job when writing code is to be a **good ancestor**.

---

## 1. Core Philosophy

1. **Correctness first**

   * A small correct solution beats a clever incomplete one.
   * We prefer obvious, boring code over “smart” tricks.

2. **Functional at the core**

   * Data in → data out, with minimal side effects.
   * Side effects are isolated at the edges (I/O, network, UI).

3. **Strict typing**

   * All public and most internal interfaces are fully typed.
   * Type errors are treated as build failures.

4. **Low lexical and structural complexity**

   * Small functions, shallow nesting, short modules.
   * No “god functions”, no “god objects”.

5. **Idiomatic Python**

   * Use the common patterns the Python community expects.
   * Avoid re-implementing standard library features.

6. **Consistency over preference**

   * Follow project conventions even if you disagree with them.
   * If you want to change a convention, propose it; don’t fork it.

---

## 2. Language Subset and Style

### 2.1 Allowed language features

* Python 3 only, pinned version as per project.
* Use dataclasses or simple classes for data structures.
* Use enums for discrete sets of related values.
* Use comprehensions where they are obviously readable.
* Use context managers for resources (files, locks, DB sessions).

### 2.2 Discouraged / forbidden patterns

* Deep inheritance hierarchies (prefer composition).
* Metaclasses, implicit magic, runtime monkey-patching.
* Global mutable state (singletons, module-level caches) unless explicitly approved.
* Overloaded operator abuse that obscures intent.
* Highly nested comprehensions; if it’s not trivial, use a loop.
* Catch-all exception handlers (bare `except` in your head, even if not literally written).

---

## 3. Functional Paradigms

We are not writing pure functional Python, but we lean that way to keep code predictable.

### 3.1 Design

* Prefer:

  * Small, pure functions that transform data.
  * Functions that receive all their inputs via parameters.
  * Functions that return new data instead of mutating arguments.

* Avoid:

  * Functions that implicitly read from or write to global state.
  * Functions that both compute and perform I/O; separate orchestration from logic.

### 3.2 Side-effect boundaries

* Centralize side effects in clearly named layers/modules:

  * `infrastructure` or `adapters` for network, filesystem, databases, etc.
  * `services` for domain logic that orchestrates pure functions + side-effecting adapters.

* Rule of thumb:

  * Most unit tests should not require network, filesystem, or environment variables.
  * If a function is hard to test without heavy mocking, it likely mixes concerns.

### 3.3 Immutability bias

* Prefer:

  * Immutable data structures (tuples, frozensets) when appropriate.
  * Avoid in-place mutation unless it is clearly local and performance-critical.
* Shared data should be treated as immutable; copy before modifying.

---

## 4. Strict Typing

Static typing is mandatory, not optional.

### 4.1 Requirements

* Every function and method:

  * Has explicit type hints for all parameters and return values.
  * Uses typed collections (list[int], dict[str, Any], etc.).
* All class attributes are typed, either via:

  * Dataclasses with annotated fields.
  * Explicit type annotations on attributes in regular classes.

### 4.2 Type checking

* The project uses a strict static type checker (e.g., mypy or pyright):

  * The type checker must pass with zero errors for:

    * All source directories.
    * All tests.
  * No “ignore” directives unless:

    * There is a linked issue explaining why.
    * There is a comment describing a path to remove the ignore.

### 4.3 Type design rules

* Prefer:

  * Narrow, precise types over `Any`.
  * Typed aliases for complex types.
* Avoid:

  * `Any`, `object`, or massive union types unless unavoidable.
  * Generic over-engineering; introduce generics only when actually reused.

---

## 5. Low Lexical and Structural Complexity

### 5.1 Function complexity limits

Per function/method:

* Maximum cyclomatic complexity: 10
* Maximum nesting depth (if/for/while/try): 3 levels
* Maximum function length: 50–60 lines (including blank lines)
* Maximum number of parameters: 5 (excluding `self`/`cls`)

If you hit these limits:

* Split the function into smaller helpers.
* Extract logic into separate, named functions or objects.

### 5.2 Module / file complexity limits

Per module/file:

* Maximum file length: ~400 lines (soft limit).
* Prefer small modules with cohesive responsibilities.
* If a file feels like a “grab bag”, split it.

### 5.3 Lexical complexity rules

* Keep the number of local variables in a function low; prefer introducing small helpers.
* Avoid:

  * Long, compound boolean expressions without intermediate naming.
  * Nested lambdas; use named functions instead.
* Use meaningful, full-word names (no “x1”, “tmp2”, etc.) unless in tiny, obvious scopes.

---

## 6. Idiomatic Python

### 6.1 Naming and structure

* Modules, packages: lowercase_with_underscores.
* Classes: PascalCase.
* Functions and methods: lowercase_with_underscores.
* Constants: UPPERCASE_WITH_UNDERSCORES.

### 6.2 Control flow

* Prefer early returns to reduce nesting.
* Avoid clever one-liners when they obscure logic.
* Use standard idioms:

  * `for` loops instead of manual index counters when possible.
  * `enumerate` and `zip` when they clarify intent.

### 6.3 Data handling

* Use built-in types and the standard library before introducing third-party helpers.
* Prefer explicit over implicit:

  * Explicit conversions instead of relying on coercion.
  * Explicit keyword arguments for critical parameters.

---

## 7. Project Structure and Organization

### 7.1 Layers

Organize code by responsibility, not by technology:

* Domain / core logic (pure, business rules).
* Application / services (use cases, orchestration).
* Infrastructure / adapters (DB, HTTP, message queues, etc.).
* Interfaces (CLI, HTTP APIs, UI integration).

### 7.2 Files and directories

* Avoid giant “utils” modules:

  * Utility functions must live near their domain.
  * If you think it’s “generic”, ask: generic for which domain?

---

## 8. Error Handling and Logging

### 8.1 Error handling

* Fail fast on invalid state:

  * Validate inputs at boundaries.
  * Prefer explicit exceptions with clear messages.

* Do not:

  * Silently pass on exceptions; at minimum, log with context.
  * Use exceptions for normal control flow.

### 8.2 Logging

* Log at appropriate levels (debug, info, warning, error).
* Log structured data, not just free-form strings.
* Never log sensitive data (tokens, passwords, PII).

---

## 9. Testing and Documentation

### 9.1 Testing

* Every new feature:

  * Unit tests for core logic.
  * Integration tests at boundaries where appropriate.

* Rules:

  * Test names should describe behavior, not implementation.
  * Tests must be deterministic; no reliance on ordering or timing when avoidable.

### 9.2 Documentation

* Every public function and class:

  * Has a docstring describing what it does, its inputs, and outputs, and notable side effects.
* Non-obvious decisions:

  * Must include a brief comment explaining “why”, not “what”.

---

## 10. Pre-Build Quality Pipeline

Every commit must pass the following checks locally and in CI. The build fails if any step fails.

### 10.1 Code formatting

* Use a single, enforced formatter (e.g., Black).
* No manual style bikeshedding; the formatter is the source of truth.
* All files must be formatted before commit.

### 10.2 Linting

Typical tools: Ruff or Pylint (or both; tune for project).

Lint rules (high level):

* No unused imports, unused variables.
* No shadowing built-ins.
* No wild imports (no `from module import *` in spirit).
* No TODO comments without a linked issue reference.
* No bare except; exceptions must be specific.

Linting is mandatory:

* Lint warnings are treated as errors unless explicitly configured otherwise.
* Any disabled rule must be justified with a comment and ideally a tracking issue.

### 10.3 Type checking

* Run the type checker on all source and test directories.
* Zero allowed type errors.
* No new `type: ignore` lines without:

  * Justification in a comment.
  * A ticket reference if it’s a known limitation.

### 10.4 Complexity analysis

Use a complexity analysis tool (e.g., radon/xenon) to enforce thresholds:

* Cyclomatic complexity:

  * Strict max per function, as defined above (e.g., 10).
* Maintainability index:

  * Establish a minimum threshold for functions and modules.
* Any increase above thresholds is a hard failure:

  * The author must refactor to reduce complexity.

### 10.5 Security and dependency checks

* Static security scan on the codebase (e.g., bandit).
* Dependency vulnerability scanning using the project’s chosen tool.
* The build fails if:

  * New high-severity issues are introduced without an explicit exemption.

### 10.6 Test suite

* All tests must pass with:

  * Coverage above project minimum (e.g., 90% for core modules).
  * New code must not reduce coverage below the project’s threshold.

### 10.7 Pre-commit workflow

* Pre-commit hooks should run, at minimum:

  * Formatter.
  * Linter.
  * Type checker (on staged or changed files).
  * Complexity check (at least for changed files).
* Contributors are expected to commit only after all hooks pass.

### 10.8 Local enforcement (this repo)

* `scripts/pre-commit.sh` (wired to `.git/hooks/pre-commit`) blocks commits if `CODE_QUALITY.md` is missing and runs Ruff + mypy on staged Python files.
* `scripts/pre-commit.sh --all` or `make build` runs the same Ruff + mypy checks against all tracked Python files without committing.
* Install dev tooling before running hooks: `pip install -r requirements-dev.txt`.
* Ruff and mypy are configured to treat both `src/` and `scripts/` as first-class code locations.
* When invoking Python tooling directly, prefer `uvx` (e.g., `uvx python -m ruff check src scripts`, `uvx python -m mypy src scripts`).
* Docstring enforcement: `scripts/check_docstrings.py` runs in the pre-commit hook to require module docstrings and docstrings for public classes/functions.

---

## 11. Code Review Standards

Code review is the last gate against complexity and subtle bugs.

### 11.1 Reviewer checklist (high level)

For each change:

* Correctness:

  * Does the change clearly do what the description claims?
  * Are edge cases covered?

* Simplicity:

  * Could this be split into smaller, clearer functions?
  * Are there any clever shortcuts that would confuse a new engineer?

* Typing and interfaces:

  * Are all new or modified functions fully typed?
  * Are new types/interfaces minimal and precise?

* Complexity:

  * Do any functions feel too long, too nested, or too “busy”?
  * Are there places where extracting helpers would help readability?

* Style and idioms:

  * Does the code follow project conventions and Python idioms?
  * Are names clear and expressive?

* Tests and docs:

  * Are there tests for the new behavior?
  * Are tests well-named and readable?
  * Are docstrings and comments updated?

### 11.2 Reviewer authority

* Reviewers are encouraged to push back on:

  * Unnecessary complexity.
  * Unclear or undocumented logic.
  * Missing tests for non-trivial changes.
* “It works on my machine” is not a valid defense.
* “The formatter/linter/type checker allowed it” is not enough; humans guard readability.

---

## 12. Cultural Rules

* No unreviewed “quick hacks”; if it’s worth merging, it’s worth doing properly.
* Prefer “make it simple first” over “optimize prematurely”.
* The future maintainer is probably not you. Write code that:

  * You could understand after six months away.
  * A new team member could navigate in a day.

If you are unsure whether a pattern is acceptable, **choose the simpler, more explicit option** or ask for guidance before baking in complexity.
