---
name: contribute-docs
description: Contribute documentation or example changes that stay aligned with NeMo Relay public behavior
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Contribute Docs Or Examples

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill for docs-only or example-heavy changes.

## Rules

- Prefer the documented public API, not internal shortcuts
- Keep package names, repo references, and build commands current
- Update entry-point docs when examples or reading paths change
- Keep release-process and release-notes guidance in repo-maintainer docs such as
  `RELEASING.md`, not as user-facing docs pages or `CHANGELOG.md`
- Keep stable user-facing wrappers at `scripts/` root in docs and examples;
  only point at namespaced helper paths when documenting internal maintenance
  work
- In MDX files, top-of-file comments must use JSX comment delimiters:
  `{/*` to open and `*/}` to close. Do not use HTML comments for MDX SPDX
  headers.

## Checklist

- [ ] `README.md` or `docs/index.md` updated when entry points changed
- [ ] Relevant getting-started or reference docs updated
- [ ] Example commands still match current package names and paths
- [ ] Relevant package or crate `README.md` files updated when examples or binding guidance changed
- [ ] New or regenerated MDX files use `{/* ... */}` for top-of-file SPDX comments
- [ ] Release-policy docs still point to GitHub Releases as the only release-history source of truth
- [ ] Run `just docs` when the docs site changed; `./scripts/build-docs.sh html` remains the compatibility wrapper

## References

- `CONTRIBUTING.md`
- `RELEASING.md`
- `docs/contribute/testing-and-docs.md`
- `review-doc-style`
