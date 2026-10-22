# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_relay import diagnostics, plugin


async def inspect_runtime():
    config = plugin.PluginConfig(diagnostics={"enabled": True})
    plugin.validate(config)
    activation = await plugin.initialize(config)
    try:
        reader = diagnostics.reader()  # Does not start or enable collection.
        page = reader.snapshot({"component_key": "logging"})
        failures = page["lifetime"]["totals"]["failures_total"]["value"]
        assert type(failures) is int
        wire = json.loads(reader.snapshot_json())
        assert isinstance(wire["sequence"], str)
        info = reader.info()
        reserved = info["binding_reserved_bytes"]
        logging_coverage = info["logging_coverage"]
        # Inspect these values only inside the authorized application boundary.
    finally:
        await activation.close()
    final_page = reader.snapshot()  # The service outlives this activation.
    return {"reserved_bytes": reserved, "logging_coverage": logging_coverage, "snapshot": final_page}
