# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Unix live acceptance matrix with Pi 1.6.0 and a built Relay CLI."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(pi: Path, relay: Path) -> None:
    actual = subprocess.check_output([pi, "--version"], env={**os.environ, "PI_OFFLINE": "1"}, text=True).strip()
    if actual != "1.6.0":
        raise ValueError(f"Expected Pi 1.6.0, got {actual}")
    subprocess.run(["node", HERE / "user-bash.mjs", pi], check=True)
    subprocess.run(["node", HERE / "terminate-on-deny.mjs", pi], check=True)
    subprocess.run(["node", HERE / "cancel-authorization.mjs", pi], check=True)
    subprocess.run([sys.executable, HERE / "managed_builtins.py", "--pi", pi, "--default-profile"], check=True)
    for scenario in ["virtual_routing.py", "provider_protocols.py"]:
        subprocess.run([sys.executable, HERE / scenario, "--pi", pi, "--relay", relay], check=True)
        print(f"PASS Pi 1.6.0: {scenario}", flush=True)
    subprocess.run([sys.executable, HERE / "managed_builtins.py", "--pi", pi], check=True)
    for scenario in ["mixed-provider.mjs", "coercion.mjs", "lifecycle.mjs", "ui-waits.mjs"]:
        subprocess.run(["node", HERE / scenario, pi], check=True)
        print(f"PASS Pi 1.6.0: {scenario}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pi", "relay"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.pi.absolute(), args.relay.absolute())
