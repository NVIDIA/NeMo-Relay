# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep copyable Relay install pins on the workspace version.

Switchyard is excluded because its external native bundle is tested and pinned
independently from Relay's normal release-version update.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_TEXT = r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?"
RELEASE_LINE = re.compile(r"(?P<base>\d+\.\d+\.\d+)(?:-(?:alpha|beta|rc)\.\d+)?(?:\+[0-9A-Za-z.-]+)?")


def pin(prefix: str, suffix: str = "") -> re.Pattern[str]:
    """Match only a Relay install pin, leaving unrelated versions on its line alone."""
    return re.compile(f"(?P<prefix>{prefix})(?P<version>{VERSION_TEXT})(?P<suffix>{suffix})")


PIN_PATTERNS = (
    pin(r"raw\.githubusercontent\.com/NVIDIA/NeMo-Relay/", r"/install\.(?:sh|ps1)"),
    pin(r"\bNEMO_RELAY_VERSION\s*=\s*['\"]?"),
    pin(r"\buv add\s+['\"]?nemo-relay(?:\[[^'\"]+\])?=="),
    pin(r"\bnpm install\s+(?:--save-exact\s+)?nemo-relay-node@"),
    pin(r"\bnpm install\s+nemo-relay-openclaw@"),
    pin(r"\bopenclaw plugins install npm:nemo-relay-openclaw@"),
    pin(r"\bcargo add\s+['\"]?nemo-relay(?:-adaptive)?@=?"),
    pin(r'nemo-relay-(?:plugin|worker)\s*=\s*\{\s*version\s*=\s*"'),
)


def matches(line: str):
    for pattern in PIN_PATTERNS:
        yield from pattern.finditer(line)


def replace_version(line: str, previous: str, current: str) -> str:
    def replace(match: re.Match[str]) -> str:
        found = match.group("version")
        if found != previous:
            raise ValueError(f"expected Relay {previous}, found {found}")
        return match.group("prefix") + current + match.group("suffix")

    for pattern in PIN_PATTERNS:
        line = pattern.sub(replace, line)
    return line


def pages(root: Path) -> list[Path]:
    quickstarts = root / "docs/getting-started/quick-start"
    return [
        root / "docs/getting-started/installation.mdx",
        *(path for path in sorted(quickstarts.glob("*.mdx")) if path.stem not in {"index", "switchyard"}),
        root / "docs/supported-integrations/openclaw-plugin.mdx",
        root / "docs/build-plugins/native/build-and-package.mdx",
        root / "docs/build-plugins/workers/rust.mdx",
    ]


def historical(line: str) -> bool:
    # A tested-version statement is evidence, not an install pin. A release
    # bump must not rewrite it into a claim about an untested version.
    return "verified with Relay" in line


def base_version(version: str) -> str:
    match = RELEASE_LINE.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported Relay version: {version}")
    return match.group("base")


def update(root: Path, previous: str, current: str) -> None:
    previous, current = base_version(previous), base_version(current)
    check(root, previous)
    if previous == current:
        return
    changed = 0
    for page in pages(root):
        original = page.read_text(encoding="utf-8")
        revised = "".join(
            replace_version(line, previous, current) if not historical(line) else line
            for line in original.splitlines(keepends=True)
        )
        if revised != original:
            page.write_text(revised, encoding="utf-8")
            changed += 1
    if not changed:
        raise ValueError(f"no versioned docs contained Relay {previous}")
    check(root, current)


def check(root: Path, expected: str | None = None) -> None:
    if expected is None:
        with (root / "Cargo.toml").open("rb") as handle:
            expected = tomllib.load(handle)["workspace"]["package"]["version"]
    expected = base_version(expected)
    pins = 0
    for page in pages(root):
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            if historical(line):
                continue
            for match in matches(line):
                pins += 1
                if match.group("version") != expected:
                    raise ValueError(
                        f"{page.relative_to(root)}:{number}: expected Relay {expected}, found {match.group('version')}"
                    )
    if not pins:
        raise ValueError("no Relay documentation version pins found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("previous")
    update_parser.add_argument("current")
    args = parser.parse_args()
    if args.command == "check":
        check(ROOT)
    else:
        update(ROOT, args.previous, args.current)


if __name__ == "__main__":
    main()
