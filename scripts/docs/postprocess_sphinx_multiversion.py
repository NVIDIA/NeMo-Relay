# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

PATCH_RELEASE_DIR = re.compile(r"^(?:v)?(\d+)\.(\d+)\.(\d+)$")
MINOR_RELEASE_DIR = re.compile(r"^(?:v)?(\d+)\.(\d+)$")
PRERELEASE_DIR = re.compile(r"^(?:v)?\d+\.\d+\.\d+-(?:alpha|beta|rc)\.\d+$")
INDEX_HTML = "index.html"


def parse_patch_release_label(name: str) -> str | None:
    match = PATCH_RELEASE_DIR.fullmatch(name)
    if match is None:
        return None
    major, minor, _patch = match.groups()
    return f"v{major}.{minor}"


def parse_minor_release_key(name: str) -> tuple[int, int] | None:
    match = MINOR_RELEASE_DIR.fullmatch(name)
    if match is None:
        return None
    major, minor = match.groups()
    return (int(major), int(minor))


def parse_prerelease_version(name: str) -> Version | None:
    if PRERELEASE_DIR.fullmatch(name) is None:
        return None

    try:
        parsed = Version(name.lstrip("v"))
    except InvalidVersion:
        return None

    if parsed.is_prerelease:
        return parsed

    return None


def iter_version_directories(build_dir: Path):
    for child in sorted(build_dir.iterdir(), key=lambda path: path.name):
        if child.is_dir():
            yield child


def normalize_release_directories(build_dir: Path) -> None:
    # Pages are published at the minor line (`v1.3/`) rather than patch tags
    # (`v1.3.2/`). The docs build selects only the newest patch per minor, so
    # this rename is expected to be one-to-one.
    renames: list[tuple[Path, Path]] = []

    for child in iter_version_directories(build_dir):
        minor_label = parse_patch_release_label(child.name)
        if minor_label is None:
            continue

        target = build_dir / minor_label
        if child == target:
            continue
        if target.exists():
            raise RuntimeError(f"Release directory conflict: {target}")
        renames.append((child, target))

    for source, target in renames:
        source.rename(target)


def latest_stable_version(build_dir: Path) -> str | None:
    candidates: list[tuple[tuple[int, int], str]] = []

    for child in iter_version_directories(build_dir):
        version_key = parse_minor_release_key(child.name)
        if version_key is not None:
            candidates.append((version_key, child.name))

    if not candidates:
        return None

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return candidates[0][1]


def latest_prerelease_version(build_dir: Path) -> str | None:
    candidates: list[tuple[Version, Path]] = []

    for child in iter_version_directories(build_dir):
        version = parse_prerelease_version(child.name)
        if version is not None:
            candidates.append((version, child))

    if not candidates:
        return None

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    latest = candidates[0][1]
    if (latest / INDEX_HTML).exists():
        return latest.name

    return None


def resolve_default_version(build_dir: Path) -> str | None:
    # Prefer the newest stable release for the site root redirect. Prereleases
    # can be the default only when the most recent included prerelease was built.
    stable = latest_stable_version(build_dir)
    if stable is not None:
        return stable

    prerelease = latest_prerelease_version(build_dir)
    if prerelease is not None:
        return prerelease

    if (build_dir / "main" / INDEX_HTML).exists():
        return "main"

    return None


def render_redirect_html(version: str) -> str:
    escaped = html.escape(version)
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={escaped}/">
    <link rel="canonical" href="{escaped}/">
    <title>Redirecting to {escaped}</title>
  </head>
  <body>
    <p>Redirecting to <a href="{escaped}/">{escaped}</a>.</p>
  </body>
</html>
"""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/docs/postprocess_sphinx_multiversion.py <build-dir>", file=sys.stderr)
        return 2

    build_dir = Path(argv[1])
    if not build_dir.exists():
        print(f"Build directory does not exist: {build_dir}", file=sys.stderr)
        return 1
    if not build_dir.is_dir():
        print(f"Build path is not a directory: {build_dir}", file=sys.stderr)
        return 1

    try:
        normalize_release_directories(build_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    default_version = resolve_default_version(build_dir)
    if default_version is None:
        print("Could not determine a default docs version from the build output", file=sys.stderr)
        return 1

    target = build_dir / default_version / INDEX_HTML
    if not target.exists():
        print(f"Default version index not found: {target}", file=sys.stderr)
        return 1

    (build_dir / INDEX_HTML).write_text(render_redirect_html(default_version), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
