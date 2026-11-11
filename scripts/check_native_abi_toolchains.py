# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the same native binaries with hosts built by two Rust toolchains."""

import argparse
import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path


def run(arguments: list[str], root: Path, environment: dict[str, str]) -> None:
    """Print and run a fixture check with the selected toolchain environment."""
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=root, env=environment, check=True)


def main() -> None:
    """Build fixtures with two compilers and cross-check unchanged binaries against their hosts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toolchain", help="Installed comparison toolchain, such as 1.94.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    environment = dict(os.environ)
    current_target = (
        root / (environment.get("NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR") or "target/native-abi-fixtures")
    ).resolve()
    target_name = args.toolchain
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", target_name) is None:
        target_name = "toolchain-" + hashlib.sha256(os.fsencode(args.toolchain)).hexdigest()
    target = (root / "target/native-abi-toolchains" / target_name).resolve()
    if current_target.is_relative_to(target) or target.is_relative_to(current_target):
        parser.error("current and comparison fixture target directories must not overlap")
    current = current_target / "debug"
    comparison = target / "debug"
    environment["NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR"] = str(current_target)
    run(["just", "build-native-abi-test-fixtures"], root, environment)
    older = environment | {"NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR": str(target)}
    run(["rustup", "run", args.toolchain, "just", "build-native-abi-test-fixtures"], root, older)
    # Current host, comparison-toolchain plugins, and comparison-toolchain future host.
    run(
        ["cargo", "test", "--locked", "-p", "nemo-relay-plugin-abi", "--test", "binary_compatibility"],
        root,
        environment | {"NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES": str(comparison)},
    )
    prefix, suffix = ("", ".dll") if os.name == "nt" else ("lib", ".dylib" if platform.system() == "Darwin" else ".so")
    executable = "nemo-relay-native-abi-fixture-future-host" + (".exe" if os.name == "nt" else "")
    # Both future hosts must accept each unchanged plugin binary, regardless of compiler.
    for host_directory in (current, comparison):
        for plugin_directory in (current, comparison):
            for plugin in ("baseline", "future", "package_next"):
                run(
                    [
                        str(host_directory / executable),
                        str(plugin_directory / f"{prefix}nemo_relay_native_abi_fixture_{plugin}{suffix}"),
                    ],
                    root,
                    environment,
                )

    # Alternate package releases also accept frozen binaries from both compilers.
    executable = "nemo-relay-native-abi-fixture-package-next-host" + (".exe" if os.name == "nt" else "")
    for host_directory in (current, comparison):
        for plugin_directory in (current, comparison):
            for plugin in ("baseline", "package_next"):
                run(
                    [
                        str(host_directory / executable),
                        str(plugin_directory / f"{prefix}nemo_relay_native_abi_fixture_{plugin}{suffix}"),
                    ],
                    root,
                    environment,
                )


if __name__ == "__main__":
    main()
