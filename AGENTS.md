# Automated Review Guidance for PyKale

This file guides automated reviewers (Codex) and humans reviewing pull requests. Keep feedback low-noise, friendly, and focused on what blocks a merge. Follow the tone of the contributing guide and the pull request template.

## Priorities
- Assume good intent and keep comments actionable.
- Prefer small, focused pull requests and align with existing PyKale design choices.
- Respect existing CI checks and pre-commit hooks (flake8, black, isort). Do not ask for extra processes beyond the current docs.
- Tests matter: new code should be covered by tests per the test guidelines. If tests are missing or invalid, raise that clearly.
- Protect users: highlight correctness issues, security concerns, breaking changes, and backward-compatibility risks.
- If external code is added, ensure credit and license information are included as required by the contributing guide.

## Severity labels (plain language)
- Must-fix (highest priority, sometimes called P0): Blocking issues that must be resolved before merge. This includes correctness bugs, security concerns, breaking changes or backward-compatibility risks, CI breakage, missing or invalid tests, and dependency or licensing risks.
- Important (medium priority, sometimes called P1): Significant issues that should be fixed soon but are not immediate blockers. Examples include unclear API behavior, missing docstrings needed for documentation, or mismatches with stated design goals.
- Optional (lowest priority, sometimes called P2): Extra polish or minor improvements. Mention Optional items only if the contributor asks for extra polish.

## What to comment on
- CI status: if checks fail, explain the likely cause and how to rerun or fix. If CI issues are due to known platform flakiness, note it without blocking if maintainers typically re-run.
- Tests: call out missing test coverage or failing tests and suggest the smallest reasonable tests to add.
- API and scientific validity: flag unclear semantics, incorrect assumptions, or results that appear inconsistent with the intended method.
- Docs: if new APIs are added, point to the docs source update expectation and Google style docstrings.

## What to avoid
- Style nitpicks already handled by pre-commit or CI.
- Large refactors or formatting-only requests unless they fix a real defect.
- Repeating project policies; link to existing guidance only when a change is required.

## Review output format
1) Summary (2-3 bullets)
2) Issues list grouped by severity (Must-fix, Important), each with a concrete suggested fix
