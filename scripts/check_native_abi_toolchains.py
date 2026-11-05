# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the same native binaries with hosts built by two Rust toolchains."""

import argparse
import os
import platform
import subprocess
from pathlib import Path


def run(arguments: list[str], root: Path, environment: dict[str, str]) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=root, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toolchain", help="Installed comparison toolchain, such as 1.94.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    current = root / "target/native-abi-fixtures/debug"
    target = root / "target/native-abi-toolchains" / args.toolchain
    comparison = target / "debug"
    environment = dict(os.environ)
    run(["just", "build-native-abi-test-fixtures"], root, environment)
    older = environment | {"NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR": str(target)}
    run(["rustup", "run", args.toolchain, "bash", "scripts/build_native_abi_fixtures.sh"], root, older)
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
            for plugin in ("baseline", "future"):
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
