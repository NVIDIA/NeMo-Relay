---
name: prepare-pr
description: Prepare a NeMo Flow branch for review with the right tests, docs, and contributor hygiene
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Prepare A PR For NeMo Flow

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill at the end of a contributor or maintainer change before opening a
pull request.

## Checklist

- [ ] Branch scope is coherent and reviewable
- [ ] Relevant tests passed under `validate-change`
- [ ] Changed files were formatted with the language-native formatter
- [ ] Any Rust change ran `just test-rust`
- [ ] Any Rust change ran `cargo fmt --all`
- [ ] Any Rust change ran `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `crates/core` or `crates/adaptive` changes ran the full language matrix
- [ ] Targeted `uv run pre-commit run --files <changed files...>` checks were used during iteration where useful
- [ ] `uv run pre-commit run --all-files` passed or issues are understood
- [ ] Docs and examples updated for any public behavior changes
- [ ] Dependent maintainer or consumer skills updated when code changes affected
      their APIs, bindings, commands, paths, packaging guidance, or best
      practices
- [ ] Pull request body follows `.github/pull_request_template.md`
- [ ] Breaking changes or renamed surfaces are called out explicitly

## Opening A Pull Request

Always use `.github/pull_request_template.md` as the source of truth for the PR
body. Before opening a PR, read the current template and preserve its headings,
checkboxes, comments' intent, and related-issue guidance.

When using GitHub CLI, prefer:

```bash
gh pr create --template .github/pull_request_template.md
```

If a tool cannot consume the template directly, create the PR body from the
template content and then fill in every visible section before opening the PR.
Do not replace the template with a freeform summary.

The PR body must include:

- `#### Overview` with a concise summary and both contribution confirmation
  checklist items preserved
- `#### Details` with the concrete changes made
- `#### Where should the reviewer start?` with the most useful file, test, or
  design decision
- `#### Related Issues: (use one of the action keywords Closes / Fixes / Resolves / Relates to)`
  with an issue reference, or a clear `Relates to: none` entry when there is no
  related issue

Only check the contribution confirmation boxes when they are true. If either
confirmation cannot be made, stop before opening the PR and surface the blocker.

## References

- `CONTRIBUTING.md`
- `.github/pull_request_template.md`
- `validate-change`
