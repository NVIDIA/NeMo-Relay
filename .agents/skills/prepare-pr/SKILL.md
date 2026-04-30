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
- [ ] Commit messages and PR summary explain what changed, why, and how it was tested
- [ ] Breaking changes or renamed surfaces are called out explicitly

## PR Description Should Cover

- What changed
- Why the change exists
- Key implementation notes
- Tests run
- Any breaking behavior or migration notes

## References

- `CONTRIBUTING.md`
- `validate-change`
