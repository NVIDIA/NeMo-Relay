# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the explicit native ABI contract and unified package release metadata."""

import argparse
import re
import tomllib
from pathlib import Path


def read_contract(root: Path) -> str:
    """Read the canonical ABI identifier shared by the interface and its build script."""
    source = (root / "crates/plugin-abi/src/contract.rs").read_text()
    matches = re.findall(r'^pub const NATIVE_API: &str = "([^"]+)";', source, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("expected one NATIVE_API contract declaration")
    contract = matches[0]
    match = re.fullmatch(r"v([1-9][0-9]*)\.(0|[1-9][0-9]*)", contract)
    if match is None or any(int(part) > 2**32 - 1 for part in match.groups()):
        raise ValueError(f"expected native contract v<major>.<minor>, received {contract!r}")
    return contract


def check(root: Path) -> str:
    """Check package alignment without using the package version as an ABI identifier."""
    contract = read_contract(root)
    manifest = tomllib.loads((root / "crates/plugin-abi/Cargo.toml").read_text())
    if manifest["package"]["version"] != {"workspace": True}:
        raise ValueError("native interface package must inherit the workspace release version")
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]
    version = workspace["package"]["version"]
    pin = workspace["dependencies"]["nemo-relay-plugin-abi"]["version"]
    if pin != f"={version}":
        raise ValueError(f"native interface SDK pin must match workspace release ={version}")
    return contract


def main() -> None:
    """Print the checked native contract independently of the Cargo release version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        print(check(Path(__file__).resolve().parent.parent))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
