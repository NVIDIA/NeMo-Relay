#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook to check and fix SPDX copyright headers.

Ensures every source file has the required SPDX copyright header with the
current year. Adds missing headers and updates stale years automatically.

Usage:
    python scripts/lint/check_copyright.py [file ...]
"""

from __future__ import annotations

import datetime
import os
import re
import sys

CURRENT_YEAR = str(datetime.date.today().year)
GITIGNORE_BASENAME = ".gitignore"

COPYRIGHT_TEXT_TEMPLATE = (
    "SPDX-FileCopyrightText: Copyright (c) {year}, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)

LICENSE_TEXT = "SPDX-License-Identifier: Apache-2.0"

# Pattern to match the full SPDX-FileCopyrightText content (after any comment prefix).
# Matches everything from "SPDX-FileCopyrightText:" to end of the copyright text.
SPDX_RE = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) (\d{4})(?:-(\d{4}))?, "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)

# Files to always skip (by basename)
SKIP_BASENAMES = frozenset(
    {
        "Cargo.lock",
        "uv.lock",
        "package-lock.json",
        "package.json",
        "go.mod",
        "go.sum",
        "LICENSE",
        "index.js",
        "index.d.ts",
        "nemo_relay.h",
    }
)

# Extensions to skip (binary/generated)
SKIP_EXTENSIONS = frozenset(
    {
        ".node",
        ".so",
        ".dylib",
        ".dll",
        ".wasm",
        ".pyc",
        ".pyo",
        ".png",
        ".jpg",
        ".gif",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".json",
        ".lock",
    }
)

# Comment style: "line" prefix or "block" (start, middle, end)
LINE_COMMENT_STYLES: dict[str, str] = {
    ".rs": "// ",
    ".go": "// ",
    ".js": "// ",
    ".mjs": "// ",
    ".ts": "// ",
    ".py": "# ",
    ".pyi": "# ",
    ".toml": "# ",
    ".yaml": "# ",
    ".yml": "# ",
    ".sh": "# ",
    ".cfg": "# ",
    GITIGNORE_BASENAME: "# ",
}

BLOCK_COMMENT_STYLES: dict[str, tuple[str, str, str]] = {
    ".md": ("<!--\n", "\n", "\n-->"),
    ".mdx": ("{/* ", "\n", " */}"),
    ".html": ("<!--\n", "\n", "\n-->"),
    ".h": ("/* ", "\n * ", "\n */"),
    ".c": ("/* ", "\n * ", "\n */"),
}

MDX_SPDX_HEADER_RE = re.compile(
    r"(?P<start>\{/\*\s*|<!--\s*)"
    r"SPDX-FileCopyrightText: Copyright \(c\) (?P<start_year>\d{4})(?:-(?P<end_year>\d{4}))?, "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\s*"
    r"SPDX-License-Identifier: Apache-2\.0"
    r"\s*(?P<end>\*/\}|-->)",
    re.MULTILINE,
)

FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def get_comment_style(filepath: str) -> str | None:
    """Return the file extension key for comment style lookup, or None if unsupported."""
    basename = os.path.basename(filepath)
    # Handle dotfiles like .gitignore
    if basename == GITIGNORE_BASENAME:
        return GITIGNORE_BASENAME
    _, ext = os.path.splitext(basename)
    if ext in LINE_COMMENT_STYLES or ext in BLOCK_COMMENT_STYLES:
        return ext
    return None


def make_header(style_key: str, year: str | None = None) -> str:
    """Build the full SPDX header string for the given comment style."""
    copyright_text = COPYRIGHT_TEXT_TEMPLATE.format(year=year or CURRENT_YEAR)
    if style_key in LINE_COMMENT_STYLES:
        prefix = LINE_COMMENT_STYLES[style_key]
        return f"{prefix}{copyright_text}\n{prefix}{LICENSE_TEXT}\n"
    if style_key in BLOCK_COMMENT_STYLES:
        start, mid, end = BLOCK_COMMENT_STYLES[style_key]
        return f"{start}{copyright_text}{mid}{LICENSE_TEXT}{end}\n"
    return ""


def should_skip(filepath: str) -> bool:
    """Return True if this file should not be checked."""
    basename = os.path.basename(filepath)
    if basename in SKIP_BASENAMES:
        return True
    _, ext = os.path.splitext(basename)
    return ext in SKIP_EXTENSIONS


def compute_year_string(start_year: str, end_year: str | None) -> str:
    """Compute the correct year or year-range string for the copyright line."""
    if end_year == CURRENT_YEAR:
        return f"{start_year}-{end_year}"
    if start_year == CURRENT_YEAR:
        return start_year
    if end_year is not None:
        return f"{start_year}-{CURRENT_YEAR}"
    return f"{start_year}-{CURRENT_YEAR}"


def insertion_index(filepath: str, content: str) -> int:
    """Return where a missing header should be inserted."""
    if get_comment_style(filepath) == ".mdx":
        match = FRONTMATTER_RE.match(content)
        if match is not None:
            return match.end()
    return 0


def process_file(filepath: str) -> bool:
    """Check and fix the SPDX header in a single file.

    Returns True if the file was modified.
    """
    if should_skip(filepath):
        return False

    style_key = get_comment_style(filepath)
    if style_key is None:
        return False

    try:
        with open(filepath, encoding="utf-8", newline="") as f:
            content = f.read()
    except (UnicodeDecodeError, IsADirectoryError):
        return False

    if not content:
        return False

    # Search for existing SPDX header in the first 10 lines
    lines = content.split("\n")
    search_lines = lines[:10]
    search_text = "\n".join(search_lines)

    if style_key == ".mdx":
        match = MDX_SPDX_HEADER_RE.search(search_text)
        if match:
            year_str = compute_year_string(match.group("start_year"), match.group("end_year"))
            header = make_header(style_key, year_str).rstrip("\n")
            if header == match.group(0):
                return False
            new_content = content[: match.start()] + header + content[match.end() :]
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)
            return True

    match = SPDX_RE.search(search_text)

    if match:
        # Header exists — check year and normalize format
        start_year = match.group(1)
        end_year = match.group(2)
        year_str = compute_year_string(start_year, end_year)
        new_text = COPYRIGHT_TEXT_TEMPLATE.format(year=year_str)
        if new_text == match.group(0):
            return False  # Already correct and canonical

        # Replace the old text with canonical text
        new_content = content[: match.start()] + new_text + content[match.end() :]
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        return True

    # Header missing — add it
    header = make_header(style_key)

    # Preserve shebang if present
    if content.startswith("#!"):
        first_newline = content.index("\n")
        shebang = content[: first_newline + 1]
        rest = content[first_newline + 1 :]
        # Add blank line between shebang and header if not already there
        if rest and not rest.startswith("\n"):
            new_content = shebang + "\n" + header + "\n" + rest
        else:
            new_content = shebang + header + "\n" + rest
    else:
        index = insertion_index(filepath, content)
        new_content = content[:index] + header + "\n" + content[index:]

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write(new_content)
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scripts/lint/check_copyright.py [file ...]", file=sys.stderr)
        return 0

    modified = []
    for filepath in sys.argv[1:]:
        if not os.path.isfile(filepath):
            continue
        if process_file(filepath):
            modified.append(filepath)

    if modified:
        print("Fixed copyright headers in:")
        for f in modified:
            print(f"  {f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
