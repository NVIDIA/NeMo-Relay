# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and close the configured native plugin without starting Hermes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_relay import plugin


async def exercise(config: Path) -> dict[str, object]:
    specs = plugin.load_dynamic_plugin_activation_specs(config)
    if len(specs) != 1 or specs[0].plugin_id != "nvidia.switchyard":
        raise AssertionError("expected one nvidia.switchyard activation spec")
    host = await plugin.initialize_with_dynamic_plugins(
        {"version": 1, "components": []},
        specs,
    )
    try:
        report = host.report
        if not host.is_active:
            raise AssertionError("dynamic plugin host did not become active")
        return report.to_dict() if hasattr(report, "to_dict") else report
    finally:
        await host.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugins", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(exercise(args.plugins.resolve()))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
