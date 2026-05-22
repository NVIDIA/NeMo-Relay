#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservatively rewrite NeMo Flow identifiers to NeMo Relay identifiers."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("NVIDIA NeMo Flow", "NVIDIA NeMo Relay"),
    ("NeMo Flow", "NeMo Relay"),
    ("Nemo Flow", "Nemo Relay"),
    ("NeMo-Flow", "NeMo-Relay"),
    ("NemoFlow", "NemoRelay"),
    ("nemoFlow", "nemoRelay"),
    ("NEMO_FLOW", "NEMO_RELAY"),
    ("nemo-flow", "nemo-relay"),
    ("nemo_flow", "nemo_relay"),
)

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "_build",
    "_generated",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "target",
    "venv",
}

LOCKFILE_NAMES = {
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}

TEXT_SUFFIXES = {
    ".bash",
    ".c",
    ".cfg",
    ".cjs",
    ".cmake",
    ".cpp",
    ".css",
    ".cu",
    ".cuh",
    ".env",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".pyi",
    ".rs",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

TEXT_FILENAMES = {
    ".dockerignore",
    ".gitignore",
    ".gitlab-ci.yml",
    "Dockerfile",
    "Justfile",
    "Makefile",
    "justfile",
}


@dataclass(frozen=True)
class FileChange:
    path: Path
    count: int


@dataclass(frozen=True)
class PathChange:
    old: Path
    new: Path


def apply_replacements(text: str) -> tuple[str, int]:
    count = 0
    updated = text
    for old, new in REPLACEMENTS:
        occurrences = updated.count(old)
        if occurrences:
            updated = updated.replace(old, new)
            count += occurrences
    return updated, count


def should_skip_dir(name: str, include_generated: bool) -> bool:
    if include_generated and name == "_generated":
        return False
    return name in SKIP_DIRS


def should_scan_file(path: Path, include_lockfiles: bool) -> bool:
    if path.name in LOCKFILE_NAMES and not include_lockfiles:
        return False
    return path.name in TEXT_FILENAMES or path.suffix in TEXT_SUFFIXES


def iter_files(root: Path, include_lockfiles: bool, include_generated: bool):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not should_skip_dir(name, include_generated)]
        current = Path(current_root)
        for filename in files:
            path = current / filename
            if should_scan_file(path, include_lockfiles):
                yield path


def rewrite_file(path: Path, write: bool) -> FileChange | None:
    try:
        raw = path.read_bytes()
    except OSError as error:
        print(f"skip unreadable: {path}: {error}")
        return None

    if b"\0" in raw:
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None

    updated, count = apply_replacements(text)
    if count == 0:
        return None

    if write:
        path.write_text(updated, encoding="utf-8")

    return FileChange(path=path, count=count)


def updated_name(name: str) -> str:
    updated, _ = apply_replacements(name)
    return updated


def collect_path_changes(root: Path, include_generated: bool) -> list[PathChange]:
    changes: list[PathChange] = []
    paths: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not should_skip_dir(name, include_generated)]
        current = Path(current_root)
        paths.extend(current / name for name in [*files, *dirs])

    for old in sorted(paths, key=lambda path: len(path.parts), reverse=True):
        new_name = updated_name(old.name)
        if new_name != old.name:
            changes.append(PathChange(old=old, new=old.with_name(new_name)))
    return changes


def apply_path_changes(changes: list[PathChange], write: bool) -> list[PathChange]:
    applied: list[PathChange] = []
    for change in changes:
        if change.new.exists():
            print(f"skip rename conflict: {change.old} -> {change.new}")
            continue
        if write:
            change.old.rename(change.new)
        applied.append(change)
    return applied


def print_report(
    file_changes: list[FileChange],
    path_changes: list[PathChange],
    write: bool,
    max_report: int,
) -> None:
    mode = "updated" if write else "would update"
    rename_mode = "renamed" if write else "would rename"

    for change in file_changes[:max_report]:
        print(f"{mode}: {change.path} ({change.count} replacements)")
    if len(file_changes) > max_report:
        print(f"... {len(file_changes) - max_report} more file changes omitted")

    for change in path_changes[:max_report]:
        print(f"{rename_mode}: {change.old} -> {change.new}")
    if len(path_changes) > max_report:
        print(f"... {len(path_changes) - max_report} more path changes omitted")

    print(f"summary: {len(file_changes)} files {mode}; {len(path_changes)} paths {rename_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite explicit NeMo Flow names to NeMo Relay names.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository or project root to scan.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply changes. Without this flag, only report planned changes.",
    )
    parser.add_argument(
        "--rename-paths",
        action="store_true",
        help="Also report or apply file and directory renames.",
    )
    parser.add_argument(
        "--include-lockfiles",
        action="store_true",
        help="Rewrite lockfiles directly instead of leaving them for package tools.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Scan generated directories named _generated.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=200,
        help="Maximum file changes and path changes to print.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        parser.error(f"root does not exist: {root}")
    if not root.is_dir():
        parser.error(f"root must be a directory: {root}")

    file_changes = [
        change
        for path in iter_files(root, args.include_lockfiles, args.include_generated)
        if (change := rewrite_file(path, args.write)) is not None
    ]

    path_changes: list[PathChange] = []
    if args.rename_paths:
        path_changes = apply_path_changes(
            collect_path_changes(root, args.include_generated),
            args.write,
        )

    print_report(file_changes, path_changes, args.write, args.max_report)
    if not args.write:
        print("dry run only; pass --write to apply changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
