# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the independent native interface release and its exact SDK pin aligned."""

import argparse
import re
import tomllib
from pathlib import Path

SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def validate_version(version: str) -> None:
    """Require SemVer, including canonical numeric prerelease identifiers."""
    match = SEMVER.fullmatch(version)
    if match is None or any(
        part.isdigit() and len(part) > 1 and part.startswith("0") for part in (match.group(4) or "").split(".")
    ):
        raise ValueError(f"expected a SemVer interface version, received {version!r}")


def check(root: Path, expected: str | None = None) -> str:
    """Check the independent package version, exact workspace pin, and optional tag."""
    manifest = tomllib.loads((root / "crates/plugin-abi/Cargo.toml").read_text())
    version = manifest["package"]["version"]
    validate_version(version)
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    pin = workspace["workspace"]["dependencies"]["nemo-relay-plugin-abi"]["version"]
    if pin != f"={version}":
        raise ValueError(f"native interface {version} requires workspace pin ={version}, received {pin!r}")
    if expected is not None and expected != version:
        raise ValueError(f"native ABI release {expected!r} does not match interface version {version}")
    return version


def set_version(root: Path, version: str) -> None:
    """Update the interface package and SDK pin without changing frozen fixtures."""
    validate_version(version)
    check(root)
    paths = [root / "crates/plugin-abi/Cargo.toml", root / "Cargo.toml"]
    patterns = [
        r'(?m)^(version\s*=\s*")[^"]+(".*)$',
        r'(?m)^(nemo-relay-plugin-abi\s*=\s*\{\s*version\s*=\s*")[^"]+(".*)$',
    ]
    replacements = [version, f"={version}"]
    updates = []
    for path, pattern, replacement in zip(paths, patterns, replacements, strict=True):
        updated, count = re.subn(
            pattern, lambda match, value=replacement: f"{match[1]}{value}{match[2]}", path.read_text()
        )
        if count != 1:
            raise ValueError(f"expected one native ABI version field in {path}")
        updates.append(updated)
    for path, updated in zip(paths, updates, strict=True):
        path.write_text(updated)


def main() -> None:
    """Dispatch interface version inspection, validation, or updates from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["version", "check", "set"])
    parser.add_argument("version", nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    try:
        if args.action == "set":
            if args.version is None:
                parser.error("set requires a version")
            set_version(root, args.version)
        print(check(root, args.version))
    except (ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
