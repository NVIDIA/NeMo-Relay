# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with ``uv run python examples/diagnostics/python.py`` from the repository."""

import asyncio
import json

from nemo_relay import diagnostics, plugin


async def main() -> None:
    config = plugin.PluginConfig(diagnostics={"enabled": True})
    plugin.validate(config)  # Validates startup settings without starting the service.
    activation = await plugin.initialize(config)
    try:
        reader = diagnostics.reader()
        page = reader.snapshot()
        assert type(page["sequence"]) is int
        assert isinstance(json.loads(reader.snapshot_json())["sequence"], str)
        assert reader.info()["binding_copy_concurrency"] == 2

        # Process each page without accumulating a second retained history.
        query = {"max_components": 1, "max_events": 1}
        for _ in range(32):
            page = reader.snapshot(query)
            # Consume the selected component/event rows within this trust boundary.
            if page["continuation"] is None:
                break
            query["continuation"] = page["continuation"]
        else:
            raise RuntimeError("The caller's page budget was exhausted")

        try:
            reader.snapshot({"unknown": True})
        except diagnostics.DiagnosticsError as error:
            assert error.reason == "invalid_query"
        else:
            raise AssertionError("An unknown query field was accepted")
    finally:
        await activation.close()

    # Plugin teardown releases its components; the process-owned reader stays valid.
    assert reader.snapshot()["source"] == page["source"]


if __name__ == "__main__":
    asyncio.run(main())
