# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Fern reference-generation scripts."""

from __future__ import annotations

import html
import shutil
from collections.abc import Callable
from pathlib import Path

TextNormalizer = Callable[[str], str]


def identity(value: str) -> str:
    return value


def escape_mdx_text(
    value: str,
    *,
    normalize: TextNormalizer = identity,
    preserve_ascii_arrows: bool = False,
) -> str:
    """Escape text that will be emitted into MDX prose, not fenced code."""
    escaped = html.escape(normalize(value), quote=False)
    if preserve_ascii_arrows:
        escaped = escaped.replace("-&gt;", "->")
    return escaped.replace("{", r"\{").replace("}", r"\}")


def quote_yaml_string(value: str, *, normalize: TextNormalizer = identity) -> str:
    value = normalize(value)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def frontmatter(
    title: str,
    description: str,
    position: int,
    *,
    sidebar_title: str | None = None,
    slug: str | None = None,
    normalize: TextNormalizer = identity,
) -> str:
    lines = ["---", f"title: {quote_yaml_string(title, normalize=normalize)}"]
    if sidebar_title is not None:
        lines.append(f"sidebar-title: {quote_yaml_string(sidebar_title, normalize=normalize)}")
    if slug is not None:
        lines.append(f"slug: {quote_yaml_string(slug, normalize=normalize)}")
    lines.extend(
        [
            f"description: {quote_yaml_string(description, normalize=normalize)}",
            f"position: {position}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def display_path(path: Path, *, cwd: Path | None = None, resolve: bool = False) -> str:
    base = cwd or Path.cwd()
    display = path.resolve() if resolve else path
    try:
        return display.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
