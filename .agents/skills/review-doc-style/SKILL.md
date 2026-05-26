---
name: review-doc-style
description: Review documentation, examples, and docs-heavy changes for NVIDIA technical writing style, terminology, and repo accuracy
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Review Documentation Style

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when reviewing docs-only changes, example-heavy changes, or any
public-facing text update that should be checked against NVIDIA style guidance
and NeMo Relay repo conventions.

## Review Priorities

- Prioritize factual accuracy over copy polish
- Flag stale commands, package names, APIs, bindings, repo paths, or support claims before stylistic issues
- Keep docs aligned with current NeMo Relay behavior, repo layout, and entry points
- Apply NVIDIA technical-writing guidance where it improves clarity and consistency without watering down technical precision

## Review Flow

1. Identify the changed docs, examples, or public-facing strings.
2. Confirm the described behavior is still true in the current repo.
3. Check whether entry-point docs also need updates:
   - `README.md`
   - `docs/index.md`
   - Package or crate READMEs
   - Binding-level source READMEs such as `python/nemo_relay/README.md` or `crates/core/README.md`
4. Start with `assets/nvidia-style-guide.md`, then open only the focused support document needed for the issue under review.
5. Scan for high-signal style issues in headings, links, code formatting, terminology, procedures, and plain-English readability.
6. Report findings in severity order with file references and concrete rewrites.

## Must-Fix Findings

Treat these as blocking issues:

- Commands, package names, file paths, or APIs are incorrect or stale
- Public behavior changed but the corresponding entry-point docs were not updated
- A doc claims support for a binding, feature, or workflow that the repo no longer provides
- Examples or procedures are likely to fail as written
- User-facing naming is inconsistent with current repo terminology
- MDX top-of-file SPDX comments use HTML comment delimiters instead of
  `{/* ... */}`
- NVIDIA is not capitalized correctly
- Code, commands, paths, or filenames are not formatted as inline code where needed

## Should-Fix Findings

Flag these when they materially improve clarity or consistency:

- Headings are not in title case for technical documentation
- Code blocks, tables, or lists are introduced with incomplete lead-in sentences
- Raw URLs or generic link text such as "here" appear in prose
- Passive voice, long sentences, or vague wording bury the action
- Terminology changes within the same document for the same concept
- Procedures are not imperative, not parallel, or too long for one sequence
- "once" is used where "after" is clearer
- "may" is used when the meaning is possibility rather than permission and "can" would be clearer

## High-Signal Review Checklist

- **Accuracy**: Commands, paths, package names, APIs, and binding claims match the current repo.
- **Entry points**: Top-level docs changed wherever users would naturally look first.
- **Headings**: Technical docs use title case consistently.
- **Voice**: Prefer active voice, present tense, short sentences, and plain English.
- **Links**: Use descriptive anchor text, not bare URLs or weak labels.
- **Formatting**: Commands, code elements, expressions, file names, and paths are monospace.
- **MDX headers**: Top-of-file MDX SPDX comments use `{/*` and `*/}` delimiters.
- **Procedures**: Steps are easy to scan, imperative, and split into smaller tasks when long.
- **Examples**: Code blocks are introduced by full sentences and match current APIs and build commands.
- **Terminology**: Use consistent terms throughout the document.
- **Dates and time**: Avoid ambiguous numeric dates and ordinal dates in body text.
- **Temporal references**: Prefer "after" over "once".
- **Trademarks**: For learning-oriented docs, do not force trademark symbols unless the source doc explicitly requires them.

## Output Format

When performing a docs review, lead with findings and keep them actionable:

- `Must fix`: incorrect, stale, misleading, or clearly noncompliant issues
- `Should fix`: clarity and consistency issues that materially improve the doc
- `Nice to have`: optional polish only when the review asked for thoroughness

Each finding should include:

- File path and line reference
- What is wrong now
- Why it conflicts with repo or style guidance
- A concrete rewrite or direction

If no issues are found, say so explicitly and mention any residual risk, such as commands or examples that were not executed.

## When To Open Style Support Docs

Start with the checklist above and `assets/nvidia-style-guide.md`. Open support
docs selectively instead of reading every asset for routine reviews.

| Support Doc | Open For |
|---|---|
| `assets/nvidia-style-technical-docs.md` | Headings, links, lists, tables, code examples, procedures, UI references, accessibility, and technical-document formatting. |
| `assets/nvidia-style-language-mechanics.md` | Voice, tone, plain English, active voice, contractions, temporal wording, punctuation, dates, numbers, units, and symbols. |
| `assets/nvidia-style-brand-terminology.md` | NVIDIA capitalization, product names, model names, trademarks, acronyms, titles, legal copy, SEO, and social copy. |

## References

- `CONTRIBUTING.md`
- `assets/nvidia-style-guide.md`
- `assets/nvidia-style-technical-docs.md`
- `assets/nvidia-style-language-mechanics.md`
- `assets/nvidia-style-brand-terminology.md`
