<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Style Guidance for Agents

Use this file as the first-pass reference for NeMo Relay documentation reviews.
It condenses NVIDIA writing guidance into review actions and points to focused
support documents for deeper checks.

This guide is not a substitute for verifying repository facts. For NeMo Relay
docs, factual accuracy and current API behavior are more important than copy
polish.

## Agent Review Order

1. Verify the technical claim against the current repository, public API, or documented command.
2. Run the fast-path checklist in this file.
3. Open only the support document that matches the ambiguity or finding.
4. Report findings by severity and include a concrete rewrite or direction.
5. Avoid style-only findings when the current wording is clear, accurate, and consistent with nearby docs.

## Fast-Path Checklist

Flag these issues before opening the detailed support docs:

- Spell `NVIDIA` in all caps. Do not use `Nvidia`, `nvidia`, or `NV`.
- Format commands, code elements, expressions, package names, file names, and
  paths as inline code.
- Use descriptive link text. Avoid raw URLs and weak anchors such as "here" or "read more."
- Use title case consistently for technical documentation headings.
- Introduce code blocks, lists, tables, and images with complete sentences.
- Write procedures as imperative steps. Keep steps parallel and split long procedures into smaller tasks.
- Prefer active voice, present tense, short sentences, contractions, and plain English.
- Use `can` for possibility and reserve `may` for permission.
- Use `after` for temporal relationships instead of `once`.
- Prefer `refer to` over `see` when the wording points readers to another resource.
- Avoid culture-specific idioms, unnecessary Latinisms, jokes, and marketing exaggeration in technical docs.
- Spell out months in body text, avoid ordinal dates, and use clear time zones.
- Spell out whole numbers from zero through nine unless they are technical values, parameters, versions, or UI values.
- Use numerals for 10 or greater and include commas in thousands.
- Do not add trademark symbols to learning-oriented docs unless the source, platform, or legal guidance explicitly requires them.

## Severity Mapping

Use this table to decide whether a style issue belongs in a review.

| Severity | Use For | Examples |
|---|---|---|
| Must fix | Incorrect, stale, misleading, or clearly noncompliant user-facing content. | Wrong command, stale package name, broken public API example, incorrect support claim, misspelled `NVIDIA`, unformatted command that is hard to read. |
| Should fix | Clear style or readability problems that affect comprehension, scanability, or consistency. | Generic link text, passive sentence hiding the actor, procedure with too many nested steps, inconsistent term for the same concept. |
| Nice to have | Optional polish that does not affect accuracy or reader success. | Slightly shorter wording, minor rhythm improvement, harmless preference difference. |

If a finding is only a preference and the current text is understandable, omit
it unless the user asked for a deep copyedit.

## Support Documents

Open these files only when the fast-path checklist is not enough:

| Support Document | Open When You Need To Check |
|---|---|
| `nvidia-style-technical-docs.md` | Headings, links, lists, tables, code examples, procedures, UI references, accessibility, and technical-document formatting. |
| `nvidia-style-language-mechanics.md` | Voice, tone, PACE, plain English, active voice, contractions, temporal wording, relative pronouns, punctuation, dates, numbers, units, and symbols. |
| `nvidia-style-brand-terminology.md` | NVIDIA capitalization, product names, model names, trademarks, acronyms, professional titles, legal copy, SEO, and social copy. |

Load the smallest document that can answer the question. Do not bulk-load all support files for routine reviews.

## Finding Template

Use this shape for docs-review findings:

```text
Must fix: <short issue>
File: <path>:<line>
Problem: <what is wrong now>
Why it matters: <repo accuracy or NVIDIA style reason>
Rewrite: <concrete replacement or direction>
```

For code review output, use the review tool's inline finding format when
available. Keep the finding body focused on the reader impact and the fix.

## Common Agent Pitfalls

- Do not enforce marketing or social-media rules on technical documentation.
- Do not add trademark symbols to NeMo Relay learning docs by default.
- Do not replace precise technical terms with simpler words when precision would be lost.
- Do not flag passive voice when the actor is unknown or the action is the important part.
- Do not rewrite API names, package names, command flags, or code literals for style.
- Do not report a style issue without a concrete rewrite or remediation path.

## External Style Fallbacks

If the support documents do not answer the question, use these sources in order:

1. Merriam-Webster for spelling and parts of speech.
2. AP Stylebook for enterprise grammar, abbreviations, punctuation, and usage.
3. Chicago Manual of Style for technical-document grammar and punctuation.
4. Microsoft Style Guide for UI text, command text, and technical conventions.
