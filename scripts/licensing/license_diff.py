#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare grouped lockfile license inventories.

By default this compares the current working tree against ``HEAD``. Pass
``--base-ref`` to compare against another git ref, or pass ``--base-json`` and
optionally ``--current-json`` to compare pre-generated inventory files.

Base refs are checked in a temporary detached worktree so the same no-build
inventory collectors can read the lockfiles and Rust manifests from that ref.

The default output is Markdown. Pass ``--format json`` for a machine-readable
diff.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import attributions_lockfile_md
from attributions_lockfile_md import LicenseInventoryEntry

LANGUAGES = ("rust", "node", "python")
Language = str

Inventory = dict[str, list[LicenseInventoryEntry]]
Diff = dict[str, dict[str, Any]]

LANGUAGE_LABELS = {
    "rust": "Rust",
    "node": "Node",
    "python": "Python",
}


def _status(message: str) -> None:
    """Write a progress message to stderr so stdout remains parseable output."""
    print(f"[license-diff] {message}", file=sys.stderr)


def _configure_root(root: Path) -> None:
    """Point the shared attribution collectors at a repository root."""
    resolved = root.resolve()
    attributions_lockfile_md.ROOT = resolved
    attributions_lockfile_md.NODE = resolved


def generate_inventory(root: Path, languages: list[Language], *, label: str) -> Inventory:
    """Return package, version, and license rows grouped by language."""
    _configure_root(root)
    inventory: Inventory = {}

    for language in languages:
        language_label = LANGUAGE_LABELS.get(language, language)
        _status(f"{label}: generating {language_label} inventory")
        if language == "rust":
            workspace_members = attributions_lockfile_md._cargo_workspace_members()
            inventory[language] = attributions_lockfile_md._rust_license_inventory(
                attributions_lockfile_md._cargo_about_json(),
                workspace_members,
            )
        elif language == "node":
            inventory[language] = attributions_lockfile_md._node_license_inventory()
        elif language == "python":
            inventory[language] = attributions_lockfile_md._python_license_inventory()
        else:
            raise ValueError(f"Unsupported language: {language}")
        _status(f"{label}: {language_label} inventory complete ({len(inventory[language])} packages)")

    return inventory


def _entry_sort_key(entry: LicenseInventoryEntry) -> tuple[str, str, str]:
    """Return the stable package inventory sort key."""
    return (
        str(entry.get("package", "")).lower(),
        str(entry.get("version", "")),
        str(entry.get("license", "")),
    )


def _entry_identity(entry: LicenseInventoryEntry) -> tuple[str, str, str]:
    """Return the identity for one package/version/license row."""
    return (
        str(entry.get("package", "")),
        str(entry.get("version", "")),
        str(entry.get("license", "")),
    )


def _group_by_package(rows: list[LicenseInventoryEntry]) -> dict[str, list[LicenseInventoryEntry]]:
    """Group inventory rows by package name."""
    grouped: dict[str, list[LicenseInventoryEntry]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("package", "")), []).append(row)
    for package_rows in grouped.values():
        package_rows.sort(key=_entry_sort_key)
    return grouped


def compare_inventories(base: Inventory, current: Inventory) -> Diff:
    """Return added, removed, and updated/changed rows grouped by language."""
    diff: Diff = {}
    language_order = {language: index for index, language in enumerate(LANGUAGES)}
    for language in sorted(set(base) | set(current), key=lambda item: (language_order.get(item, 99), item)):
        base_by_package = _group_by_package(base.get(language, []))
        current_by_package = _group_by_package(current.get(language, []))
        added: list[LicenseInventoryEntry] = []
        removed: list[LicenseInventoryEntry] = []
        updated_changed: list[dict[str, Any]] = []

        for package in sorted(set(base_by_package) | set(current_by_package), key=str.lower):
            before = base_by_package.get(package, [])
            after = current_by_package.get(package, [])
            if not before:
                added.extend(after)
                continue
            if not after:
                removed.extend(before)
                continue

            before_keys = {_entry_identity(row) for row in before}
            after_keys = {_entry_identity(row) for row in after}
            if before_keys != after_keys:
                updated_changed.append(
                    {
                        "package": package,
                        "before": before,
                        "after": after,
                        "removed": [row for row in before if _entry_identity(row) not in after_keys],
                        "added": [row for row in after if _entry_identity(row) not in before_keys],
                    }
                )

        diff[language] = {
            "added": sorted(added, key=_entry_sort_key),
            "removed": sorted(removed, key=_entry_sort_key),
            "updated_changed": updated_changed,
        }

    return diff


def _markdown_package_row(entry: LicenseInventoryEntry) -> str:
    """Render one package/version/license row."""
    package = str(entry.get("package", ""))
    version = str(entry.get("version", ""))
    license_name = str(entry.get("license", "UNKNOWN"))
    return f"- `{package}` {version} ({license_name})"


def _append_markdown_rows(parts: list[str], rows: list[LicenseInventoryEntry]) -> None:
    """Append package rows or a stable empty-state line."""
    if not rows:
        parts.append("- None\n")
        return
    for row in rows:
        parts.append(f"{_markdown_package_row(row)}\n")


def _render_updated_changed(parts: list[str], updates: list[dict[str, Any]]) -> None:
    """Append Markdown for package rows that changed while the package name remained present."""
    if not updates:
        parts.append("- None\n")
        return

    for update in updates:
        package = str(update.get("package", ""))
        parts.append(f"#### `{package}`\n\n")
        parts.append("Before:\n\n")
        _append_markdown_rows(parts, list(update.get("before") or []))
        parts.append("\nAfter:\n\n")
        _append_markdown_rows(parts, list(update.get("after") or []))
        parts.append("\n")


def render_markdown(diff: Diff) -> str:
    """Render the comparison diff as Markdown."""
    parts = ["# Lockfile License Changes\n\n"]
    language_order = {language: index for index, language in enumerate(LANGUAGES)}
    for language in sorted(diff, key=lambda item: (language_order.get(item, 99), item)):
        language_diff = diff[language]
        label = LANGUAGE_LABELS.get(language, language)
        parts.append(f"## {label}\n\n")

        parts.append("### Added\n\n")
        _append_markdown_rows(parts, list(language_diff.get("added") or []))
        parts.append("\n")

        parts.append("### Removed\n\n")
        _append_markdown_rows(parts, list(language_diff.get("removed") or []))
        parts.append("\n")

        parts.append("### Updated/Changed\n\n")
        _render_updated_changed(parts, list(language_diff.get("updated_changed") or []))
        parts.append("\n")

    return "".join(parts).rstrip() + "\n"


def _load_inventory(path: Path) -> Inventory:
    """Load an inventory JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Inventory must be a JSON object: {path}")
    inventory: Inventory = {}
    for language, rows in payload.items():
        if not isinstance(rows, list):
            raise ValueError(f"Inventory rows for {language!r} must be a list, got {type(rows).__name__}: {path}")
        inventory[str(language)] = [cast(LicenseInventoryEntry, dict(row)) for row in rows]
    return inventory


def _filter_inventory(inventory: Inventory, languages: list[str]) -> Inventory:
    """Keep only the requested languages from a loaded inventory."""
    return {language: inventory.get(language, []) for language in languages if language in inventory}


def _worktree_inventory(root: Path, ref: str, languages: list[str]) -> Inventory:
    """Generate inventory for a git ref in a temporary detached worktree."""
    tmp_parent = Path(tempfile.mkdtemp(prefix="nemo-relay-license-base-"))
    worktree = tmp_parent / "repo"
    try:
        _status(f"checking out base ref {ref} into a temporary worktree")
        subprocess.run(
            ["git", "-C", str(root), "worktree", "add", "--detach", "--quiet", str(worktree), "--", ref],
            check=True,
        )
        inventory = generate_inventory(worktree, languages, label="base")
        _status("base inventory complete")
        return inventory
    finally:
        _status("removing temporary base worktree")
        subprocess.run(
            ["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        shutil.rmtree(tmp_parent, ignore_errors=True)


def _parse_languages(values: list[str]) -> list[str]:
    """Normalize CLI language arguments."""
    if not values or "all" in values:
        return list(LANGUAGES)
    unsupported = sorted(set(values) - set(LANGUAGES))
    if unsupported:
        supported = ", ".join((*LANGUAGES, "all"))
        raise ValueError(f"Unsupported language(s): {', '.join(unsupported)}. Choose from: {supported}")
    return values


def main() -> int:
    """Compare two license inventories and write the diff."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--base-ref", default=None, help="Git ref to compare against. Defaults to HEAD.")
    base.add_argument("--base-json", type=Path, help="Pre-generated base inventory JSON.")
    parser.add_argument("--current-json", type=Path, help="Pre-generated current inventory JSON.")
    parser.add_argument(
        "languages",
        nargs="*",
        metavar="{rust,node,python,all}",
        help="Language inventories to compare. Defaults to all.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the lockfiles.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Defaults to markdown.",
    )
    parser.add_argument("--output", "-o", type=Path, help="Write output to this path instead of stdout.")
    args = parser.parse_args()

    try:
        languages = _parse_languages(args.languages)
    except ValueError as exc:
        parser.error(str(exc))
    _status(f"selected languages: {', '.join(languages)}")
    if args.current_json:
        _status(f"loading current inventory from {args.current_json}")
        current = _filter_inventory(_load_inventory(args.current_json), languages)
    else:
        _status("generating current inventory")
        current = generate_inventory(args.root, languages, label="current")
        _status("current inventory complete")

    if args.base_json:
        _status(f"loading base inventory from {args.base_json}")
        base_inventory = _filter_inventory(_load_inventory(args.base_json), languages)
    else:
        base_inventory = _worktree_inventory(args.root, args.base_ref or "HEAD", languages)

    _status("comparing inventories")
    diff = compare_inventories(base_inventory, current)
    if args.format == "json":
        _status("rendering JSON output")
        rendered = json.dumps(diff, indent=2, sort_keys=True) + "\n"
    else:
        _status("rendering Markdown output")
        rendered = render_markdown(diff)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        _status(f"wrote {args.output}")
    else:
        sys.stdout.write(rendered)
    _status("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
