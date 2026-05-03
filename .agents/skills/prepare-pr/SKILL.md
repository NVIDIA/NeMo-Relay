---
name: prepare-pr
description: Prepare, open, create, publish, update, or edit a NeMo Flow pull request or PR body with the right tests, docs, contributor hygiene, and repository pull request template
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Prepare A PR For NeMo Flow

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill at the end of a contributor or maintainer change before opening a
pull request. Also use it whenever a user asks to create, open, publish, update,
or edit a NeMo Flow pull request, pull request description, or PR body.

If this repo-local guidance conflicts with generic GitHub publishing, connector,
or plugin guidance, this skill wins for PR body format, validation language, and
review handoff details.

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
- [ ] Pull request title follows Conventional Commit style and uses the correct
      type
- [ ] Pull request body follows `.github/pull_request_template.md`
- [ ] Breaking changes or renamed surfaces are called out explicitly

## Pull Request Title

Use Conventional Commit style for PR titles:

```text
<type>: <concise imperative summary>
```

Choose the type from the actual change surface, not from the impact of the
review comment or CI outcome. Use `fix` only for an actual user-facing or
runtime/product code bug fix. Never use `fix` for changes that are not related
to product code behavior, including chores, CI configuration, docs, tests,
packaging metadata, generated-output handling, or agent/skill guidance.

Common examples:

- `ci: update codecov coverage reporting`
- `docs: clarify release workflow`
- `chore: refresh generated attribution data`
- `test: add Python scope regression coverage`
- `fix: preserve scope-local middleware cleanup`

## Opening A Pull Request

Always use `.github/pull_request_template.md` as the source of truth for the PR
body. Before opening a PR, read the current template and preserve its headings,
checkboxes, comments' intent, and related-issue guidance.

This applies both when creating a new PR and when editing an existing PR
description. Do not use a generic `Summary / Why / Validation` body unless the
current repository template uses those headings.

When using GitHub CLI, prefer:

```bash
gh pr create --template .github/pull_request_template.md
```

If a tool cannot consume the template directly, create the PR body from the
template content and then fill in every visible section before opening the PR.
Do not replace the template with a freeform summary.

After creating or editing a PR, fetch the rendered PR body and verify that the
template's visible headings and checklist items are still present.

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
