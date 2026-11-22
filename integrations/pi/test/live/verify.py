# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Unix live acceptance matrix with two installed Pi releases and a built Relay CLI."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import provider_protocols

HERE = Path(__file__).resolve().parent


def verify(baseline: Path, current: Path, relay: Path) -> None:
    for pi, version in [(baseline, "0.84.0"), (current, "1.6.0")]:
        actual = subprocess.check_output([pi, "--version"], env={**os.environ, "PI_OFFLINE": "1"}, text=True).strip()
        if actual != version:
            raise ValueError(f"Expected Pi {version}, got {actual}")
        subprocess.run(["node", HERE / "user-bash.mjs", pi, version], check=True)
        subprocess.run([sys.executable, HERE / "managed_builtins.py", "--pi", pi, "--default-profile"], check=True)
        print(f"PASS Pi {version}: shell and managed default launch", flush=True)

    with tempfile.TemporaryDirectory(prefix="relay-pi-baseline-") as directory:
        provider_protocols.run(
            baseline, relay, Path(directory), "openai-completions", "api.openai.com", "openai_base_url"
        )
    for scenario in ["virtual_routing.py", "provider_protocols.py"]:
        subprocess.run([sys.executable, HERE / scenario, "--pi", current, "--relay", relay], check=True)
        print(f"PASS Pi 1.6.0: {scenario}", flush=True)
    subprocess.run([sys.executable, HERE / "managed_builtins.py", "--pi", current], check=True)
    for scenario in ["mixed-provider.mjs", "coercion.mjs", "lifecycle.mjs", "ui-waits.mjs"]:
        subprocess.run(["node", HERE / scenario, current], check=True)
        print(f"PASS Pi 1.6.0: {scenario}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "current", "relay"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.baseline.absolute(), args.current.absolute(), args.relay.absolute())
