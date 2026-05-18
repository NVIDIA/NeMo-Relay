# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Flow-specific cleanup for generated Fern documentation."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from reference_common import quote_yaml_string

UNSUPPORTED_OPTION_LINE = re.compile(r"^:(?:gutter|sync-group|sync|selected):(?:\s+.*)?$")
GITHUB_BLOB_BASE = "https://github.com/NVIDIA/NeMo-Flow/blob/main"

# The Fern migration runs over pages that may have been produced from several
# intermediate Sphinx layouts. Keep the old variants here so the cleanup remains
# idempotent while the generated output is being refreshed.
API_REFERENCE_LIST = (
    "- [Python Library Reference](/reference/api/python-library-reference)\n"
    "- [Node.js Library Reference](/reference/api/nodejs-library-reference)\n"
    "- [Rust Library Reference](/reference/api/rust-library-reference)"
)
API_REFERENCE_INTRO_PREFIX = "Use these generated library references for symbol-level API documentation:"
API_REFERENCE_INTRO = f"{API_REFERENCE_INTRO_PREFIX}\n\n{API_REFERENCE_LIST}"
SUPPORT_API_REFERENCE_TEXT = (
    "Use the [Python Library Reference](/reference/api/python-library-reference),\n"
    "[Node.js Library Reference](/reference/api/nodejs-library-reference), and\n"
    "[Rust Library Reference](/reference/api/rust-library-reference) for generated\n"
    "symbol-level documentation. The Rust reference includes `nemo-flow`,\n"
    "`nemo-flow-adaptive`, and `nemo-flow-ffi`. For Go and WebAssembly surfaces, use\n"
    "the source directories, tests, and task-focused guides when you need exact\n"
    "behavior."
)
ASSISTANT_SYMBOL_REFERENCE_TEXT = (
    "For symbol-level work, assistants should use the source directories, tests, and\n"
    "the generated Python, Node.js, and Rust library references. For\n"
    "repository-specific automation, use the NeMo Flow agent skills under `skills/`\n"
    "and keep examples aligned with the public docs."
)
REPO_FILE_LINK_REPLACEMENTS = {
    "../../examples/nemoguardrails/example/agent_example.py": (
        f"{GITHUB_BLOB_BASE}/examples/nemoguardrails/example/agent_example.py"
    ),
    "/examples/nemoguardrails/example/agent_example.py": (
        f"{GITHUB_BLOB_BASE}/examples/nemoguardrails/example/agent_example.py"
    ),
    "../../examples/nemoguardrails/example/example_config.yml": (
        f"{GITHUB_BLOB_BASE}/examples/nemoguardrails/example/example_config.yml"
    ),
    "/examples/nemoguardrails/example/example_config.yml": (
        f"{GITHUB_BLOB_BASE}/examples/nemoguardrails/example/example_config.yml"
    ),
    "../../RELEASING.md": f"{GITHUB_BLOB_BASE}/RELEASING.md",
    "/RELEASING": f"{GITHUB_BLOB_BASE}/RELEASING.md",
    "/RELEASING.md": f"{GITHUB_BLOB_BASE}/RELEASING.md",
    "../../../ATTRIBUTIONS-Rust.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Rust.md",
    "/ATTRIBUTIONS-Rust": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Rust.md",
    "/ATTRIBUTIONS-Rust.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Rust.md",
    "../../../ATTRIBUTIONS-Python.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Python.md",
    "/ATTRIBUTIONS-Python": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Python.md",
    "/ATTRIBUTIONS-Python.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Python.md",
    "../../../ATTRIBUTIONS-Node.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Node.md",
    "/ATTRIBUTIONS-Node": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Node.md",
    "/ATTRIBUTIONS-Node.md": f"{GITHUB_BLOB_BASE}/ATTRIBUTIONS-Node.md",
}
TAB_LANGUAGE_BY_TITLE = {
    "Python": "python",
    "Node.js": "node",
    "Rust": "rust",
}
LANGUAGE_TAB_RE = re.compile(
    r'(?P<prefix><Tab\s+title="(?P<title>Python|Node\.js|Rust)")'
    r"(?![^>]*\slanguage=)"
    r"(?P<suffix>[^>]*)>"
)
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
LEADING_H1_RE = re.compile(
    r"\A(?P<prefix>(?:\s|\{/\*.*?\*/\})*)#\s+(?P<title>[^\n]+)\n+",
    re.DOTALL,
)


def _remove_omitted_binding_api_pages(pages_dir: Path) -> int:
    removed = 0

    for path in (
        pages_dir / "reference" / "api" / "python.mdx",
        pages_dir / "reference" / "api" / "python" / "index.mdx",
        pages_dir / "reference" / "api" / "rust.mdx",
        pages_dir / "reference" / "api" / "rust" / "index.mdx",
    ):
        if not path.exists():
            continue
        path.unlink()
        removed += 1

    for directory in (
        pages_dir / "reference" / "api" / "nodejs",
        pages_dir / "reference" / "api" / "rust",
        pages_dir / "reference" / "api" / "python",
    ):
        if not directory.exists():
            continue
        shutil.rmtree(directory)
        removed += 1

    return removed


def _mdx_files(pages_dir: Path) -> list[Path]:
    return sorted(pages_dir.rglob("*.mdx"))


def _write_if_changed(path: Path, original: str, updated: str) -> bool:
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def _dedupe_rust_reference_lines(text: str) -> str:
    duplicate = (
        "- [Rust Library Reference](/reference/api/rust-library-reference)\n"
        "- [Rust Library Reference](/reference/api/rust-library-reference)"
    )
    while duplicate in text:
        text = text.replace(duplicate, "- [Rust Library Reference](/reference/api/rust-library-reference)")
    return text


def _rewrite_omitted_api_mentions(pages_dir: Path) -> int:
    changed = 0
    replacements_by_path = {
        pages_dir / "reference" / "api" / "index.mdx": {
            (
                "Use these pages for generated symbol-level API documentation across the\n"
                "supported public language surfaces."
            ): API_REFERENCE_INTRO,
            (
                "Use the [Python Library Reference](/reference/api/python-library-reference) when\n"
                "you need generated symbol-level documentation for the Python package."
            ): API_REFERENCE_INTRO,
            (
                "Use these generated library references for symbol-level API documentation:\n\n"
                "- [Python Library Reference](/reference/api/python-library-reference)\n"
                "- [Node.js Library Reference](/reference/api/nodejs-library-reference)"
            ): API_REFERENCE_INTRO,
        },
        pages_dir / "reference" / "api" / "python-library-reference" / "index.mdx": {
            "position: 2": "position: 1",
        },
        pages_dir / "getting-started" / "prerequisites.mdx": {
            (
                "| Node.js | 20 or newer | Node.js bindings and generated Node.js API docs |"
            ): "| Node.js | 20 or newer | Node.js bindings and local package builds |",
            ("Install Node.js dependencies when you need Node.js builds or generated Node.js API documentation:"): (
                "Install Node.js dependencies when you need Node.js builds or local package validation:"
            ),
        },
        pages_dir / "resources" / "support-and-faqs.mdx": {
            (
                "Use [API](/reference/api) for generated symbol-level documentation.\n"
                "The primary generated entry points are [Python API](/reference/api/python),\n"
                "[Node.js API](/reference/api/nodejs), and\n"
                "[Rust API](/reference/api/rust).\n\n"
                "Go, WebAssembly, and raw FFI are experimental and source-first; use their source\n"
                "directories and tests when you need exact behavior."
            ): SUPPORT_API_REFERENCE_TEXT,
            (
                "Use the [Python Library Reference](/reference/api/python-library-reference) for\n"
                "generated symbol-level Python documentation. For Rust, Node.js, Go,\n"
                "WebAssembly, and raw FFI surfaces, use the source directories, tests, and\n"
                "task-focused guides when you need exact behavior."
            ): SUPPORT_API_REFERENCE_TEXT,
            (
                "Use the [Python Library Reference](/reference/api/python-library-reference) and\n"
                "[Node.js Library Reference](/reference/api/nodejs-library-reference) for\n"
                "generated symbol-level documentation. For Rust, Go, WebAssembly, and raw FFI\n"
                "surfaces, use the source directories, tests, and task-focused guides when you\n"
                "need exact behavior."
            ): SUPPORT_API_REFERENCE_TEXT,
            (
                "For symbol-level work, assistants should use the generated Rust, Python, and\n"
                "Node.js API references. For repository-specific automation, use the NeMo Flow\n"
                "agent skills under `skills/` and keep examples aligned with the public docs."
            ): ASSISTANT_SYMBOL_REFERENCE_TEXT,
            (
                "For symbol-level work, assistants should use the source directories, tests, and\n"
                "the generated Python Library Reference. For repository-specific automation, use\n"
                "the NeMo Flow agent skills under `skills/` and keep examples aligned with the\n"
                "public docs."
            ): ASSISTANT_SYMBOL_REFERENCE_TEXT,
            (
                "For symbol-level work, assistants should use the source directories, tests, and\n"
                "the generated Python and Node.js library references. For repository-specific\n"
                "automation, use the NeMo Flow agent skills under `skills/` and keep examples\n"
                "aligned with the public docs."
            ): ASSISTANT_SYMBOL_REFERENCE_TEXT,
        },
    }

    for path, replacements in replacements_by_path.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        updated = _dedupe_rust_reference_lines(_apply_replacements(text, replacements))
        if _write_if_changed(path, text, updated):
            changed += 1

    return changed


def _remove_unsupported_directive_options(pages_dir: Path) -> int:
    changed = 0
    for mdx_file in _mdx_files(pages_dir):
        lines = mdx_file.read_text(encoding="utf-8").splitlines()
        filtered = [line for line in lines if not UNSUPPORTED_OPTION_LINE.match(line.strip())]
        if _write_if_changed(mdx_file, "\n".join(lines) + "\n", "\n".join(filtered) + "\n"):
            changed += 1
    return changed


def _rewrite_repo_file_links(pages_dir: Path) -> int:
    changed = 0
    for mdx_file in _mdx_files(pages_dir):
        text = mdx_file.read_text(encoding="utf-8")
        updated = _apply_replacements(
            text,
            {f"]({old})": f"]({new})" for old, new in REPO_FILE_LINK_REPLACEMENTS.items()},
        )
        updated = re.sub(
            r"\]\(/troubleshooting(?=[)#])",
            "](/resources/troubleshooting",
            updated,
        )
        if _write_if_changed(mdx_file, text, updated):
            changed += 1
    return changed


def _frontmatter_value(frontmatter: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(?P<value>.*?)\s*$", frontmatter, re.MULTILINE)
    if match is None:
        return None
    value = match.group("value").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _set_frontmatter_value(frontmatter: str, key: str, value: str) -> str:
    line = f"{key}: {quote_yaml_string(value)}"
    if re.search(rf"^{re.escape(key)}:", frontmatter, re.MULTILINE):
        return re.sub(rf"^{re.escape(key)}:.*$", line, frontmatter, count=1, flags=re.MULTILINE)
    return frontmatter.rstrip() + "\n" + line


def _add_frontmatter_value_after(frontmatter: str, after_key: str, key: str, value: str) -> str:
    if re.search(rf"^{re.escape(key)}:", frontmatter, re.MULTILINE):
        return _set_frontmatter_value(frontmatter, key, value)

    line = f"{key}: {quote_yaml_string(value)}"
    return re.sub(
        rf"(^{re.escape(after_key)}:.*$)",
        rf"\1\n{line}",
        frontmatter,
        count=1,
        flags=re.MULTILINE,
    )


def _remove_duplicate_page_headings(pages_dir: Path) -> int:
    changed = 0
    for mdx_file in _mdx_files(pages_dir):
        text = mdx_file.read_text(encoding="utf-8")
        frontmatter_match = FRONTMATTER_RE.match(text)
        if frontmatter_match is None:
            continue

        body = text[frontmatter_match.end() :]
        h1_match = LEADING_H1_RE.match(body)
        if h1_match is None:
            continue

        frontmatter = frontmatter_match.group("body")
        h1_title = h1_match.group("title").strip()
        frontmatter_title = _frontmatter_value(frontmatter, "title")
        updated_frontmatter = frontmatter

        if frontmatter_title and frontmatter_title != h1_title:
            updated_frontmatter = _set_frontmatter_value(updated_frontmatter, "title", h1_title)
            updated_frontmatter = _add_frontmatter_value_after(
                updated_frontmatter,
                "title",
                "sidebar-title",
                frontmatter_title,
            )

        updated_body = h1_match.group("prefix") + body[h1_match.end() :]
        updated = f"---\n{updated_frontmatter}\n---\n{updated_body}"
        if _write_if_changed(mdx_file, text, updated):
            changed += 1
    return changed


def _repair_premature_tab_group_closes(pages_dir: Path) -> int:
    changed = 0
    for mdx_file in _mdx_files(pages_dir):
        lines = mdx_file.read_text(encoding="utf-8").splitlines()
        repaired: list[str] = []
        in_tabs = False
        in_tab = False

        for index, line in enumerate(lines):
            stripped = line.strip()

            if (
                in_tabs
                and not in_tab
                and stripped
                and not (stripped == "</Tabs>" or stripped.startswith("<Tab ") or stripped.startswith("</Tab"))
            ):
                if repaired and repaired[-1] != "":
                    repaired.append("")
                repaired.append("</Tabs>")
                repaired.append("")
                in_tabs = False

            if stripped == "<Tabs>":
                in_tabs = True
                repaired.append(line)
                continue

            if in_tabs and stripped == "</Tabs>":
                next_nonblank = next(
                    (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                    "",
                )
                if next_nonblank.startswith("<Tab "):
                    continue
                in_tabs = False
                repaired.append(line)
                continue

            if in_tabs and stripped.startswith("<Tab "):
                in_tab = True

            repaired.append(line)

            if in_tabs and stripped.startswith("</Tab"):
                in_tab = False

        if in_tabs:
            if repaired and repaired[-1] != "":
                repaired.append("")
            repaired.append("</Tabs>")

        updated = "\n".join(repaired) + "\n"
        original = "\n".join(lines) + "\n"
        if _write_if_changed(mdx_file, original, updated):
            changed += 1
    return changed


def _add_language_tab_sync(pages_dir: Path) -> int:
    changed = 0

    def replace(match: re.Match[str]) -> str:
        title = match.group("title")
        language = TAB_LANGUAGE_BY_TITLE[title]
        return f'{match.group("prefix")} language="{language}"{match.group("suffix")}>'

    for mdx_file in _mdx_files(pages_dir):
        text = mdx_file.read_text(encoding="utf-8")
        updated = LANGUAGE_TAB_RE.sub(replace, text)
        if _write_if_changed(mdx_file, text, updated):
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages_dir", type=Path)
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.is_dir():
        raise SystemExit(f"pages directory not found: {pages_dir}")

    removed_api_pages = _remove_omitted_binding_api_pages(pages_dir)
    api_mentions_changed = _rewrite_omitted_api_mentions(pages_dir)
    links_changed = _rewrite_repo_file_links(pages_dir)
    language_tabs_changed = _add_language_tab_sync(pages_dir)
    headings_changed = _remove_duplicate_page_headings(pages_dir)
    tab_groups_changed = _repair_premature_tab_group_closes(pages_dir)
    changed = _remove_unsupported_directive_options(pages_dir)
    print(
        f"NeMo Flow Fern cleanup updated {changed} page(s), "
        f"rewrote links in {links_changed} page(s), "
        f"added language tab metadata in {language_tabs_changed} page(s), "
        f"removed duplicate page headings in {headings_changed} page(s), "
        f"repaired tab groups in {tab_groups_changed} page(s), "
        f"removed omitted API docs at {removed_api_pages} path(s), "
        f"rewrote omitted API mentions in {api_mentions_changed} page(s)"
    )


if __name__ == "__main__":
    main()
