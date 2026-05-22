<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Technical Documentation Style

Open this file when a review turns on technical-document structure, formatting,
examples, procedures, tables, links, or UI references.

## Review Priorities

1. Verify that commands, examples, paths, APIs, and support claims match the current repository.
2. Make the document easy to scan by fixing headings, lead-in sentences, lists, tables, and procedure shape.
3. Preserve exact code, command, API, package, and UI strings unless they are factually wrong.
4. Prefer focused findings over broad rewrites.

## Headings and Titles

- Use title case consistently in technical documentation headings.
- Avoid quotation marks, ampersands, and exclamation marks in headings.
- Keep product, event, research, and whitepaper names in their official title case.
- Use title case for table headers.
- Do not force social-media sentence case into technical docs.

## Technical Formatting

Use this table for common review calls:

| Item | Format | Review Signal |
|---|---|---|
| Code elements, commands, parameters, package names, expressions | Monospace | Flag prose such as "run just test-rust" and rewrite as `run just test-rust`. |
| Directories, file names, and paths | Monospace | Use backticks around paths such as `python/nemo_relay/README.md`. |
| Variables inside paths | Angle brackets inside monospace | Prefer `/home/<username>/.login` for placeholders. |
| Error messages and strings | Quotation marks | Keep literal code strings in code formatting when that is clearer. |
| UI buttons, menus, fields, and labels | Bold | Example: Select **Save**. |
| Menu paths | Angle brackets between UI labels | Example: Select **File** > **Save As**. |
| New terms | Italics on first use | Use sparingly and only when the term is introduced. |
| Publication titles | Italics | Article and blog titles use title case but are not italicized. |
| Keyboard shortcuts | Plain text | Example: Press Ctrl+Alt+Delete. |
| GitHub repositories | Owner/repo link text | Prefer `[NVIDIA/NeMo](link)` over "the GitHub repo." |

## Code Examples

- Introduce every code block with a complete sentence.
- Do not make a code block complete the grammar of the previous sentence.
- Do not continue a sentence after a code block.
- Use syntax highlighting when the format supports it.
- Avoid the word "snippet" unless the surrounding docs already use it as a term of art.
- Keep inline method, function, and class references consistent with nearby docs. Empty parentheses can be omitted for prose readability when no call is shown.

Correct:

````text
The following command runs the Rust tests:

```bash
just test-rust
```
````

Incorrect:

````text
Run

```bash
just test-rust
```

to test Rust.
````

## Links

- Use descriptive anchor text that matches the destination title when possible.
- Avoid raw URLs in running text.
- Avoid generic anchors such as "here," "this page," and "read more."
- If a linked term includes an acronym, include the acronym in the link text.
- Do not link long sentences or multiple sentences.
- Avoid links that pull readers away from a procedure unless the link is a
  prerequisite or reference required to complete the task.

## Lists

All lists should have:

- A complete lead-in sentence.
- More than one item.
- No more than two levels.
- Parallel sentence construction.
- One idea or action per item.
- End punctuation when list items are complete sentences.

Use bulleted lists when order does not matter. Use numbered lists when order matters or the list is a task sequence.

Definition lists should use a bold term followed by a complete definition. Keep definitions parallel and punctuated.

## Tables

Use tables for reference information, decision support, compatibility matrices, and choices that readers compare.

Flag tables that:

- Have only one row.
- Lack a caption, title, or lead-in sentence.
- Use sentence case in headers when nearby technical docs use title case.
- Leave cells empty without an intentional placeholder.
- Include code samples or links when prose would be easier to read.

## Procedures

- Write steps as imperative sentences.
- Keep one action per step when possible.
- Keep numbered procedures to about five to seven steps. Split longer sequences into smaller tasks.
- Use subheadings to separate tasks or phases.
- Avoid deep nesting. If a step needs several substeps, it probably needs its own procedure.
- Keep explanatory context outside the step list unless the context is needed to complete the step.

## Accessibility

Flag accessibility issues that affect docs readers:

- Missing or vague alt text for images and buttons.
- Link text that does not describe the destination.
- Heading levels that skip hierarchy in rendered documentation.
- Long paragraphs or sentences that make scanning difficult.
- Instructions that rely only on color, position, or visual appearance.
- Low contrast when the change includes rendered images or custom HTML.

## UI References

- Bold UI labels, buttons, menus, and field names.
- Use angle brackets for consecutive UI navigation, such as **File** > **Open**.
- Match UI text exactly, including capitalization.
- Do not rewrite UI labels for prose style.

## Technical Conventions

- Use LaTeX or MathML for equations and formulas when the platform supports it.
- Use a leading period and lowercase for file extensions, such as `.tgz`.
- Use uppercase without a period for file types, such as TGZ.
- Use footnotes sparingly except in research papers or platforms that require them.
- Use the same term for the same concept throughout the document. Introduce synonyms only when connecting an industry-standard term to an NVIDIA-specific term.
