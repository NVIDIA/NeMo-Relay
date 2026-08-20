# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary Python plugin coverage for export activation hooks."""

from __future__ import annotations

import pytest

from nemo_relay import plugin


@pytest.mark.asyncio
@pytest.mark.parametrize(("decision", "expected"), [("allow", 1), ("deny", 0)])
async def test_plugin_can_gate_its_own_export_target(decision: str, expected: int) -> None:
    kind = f"tests.python_export_activation_{decision}"
    activated: list[str] = []

    class SelfGatedPlugin:
        def validate(self, plugin_config):
            del plugin_config
            return []

        def register(self, plugin_config, context):
            del plugin_config

            async def policy(request):
                assert request == {
                    "target_kind": "tests.telemetry.otlp",
                    "config": {"country": "US"},
                }
                return decision

            async def activate():
                activated.append("exporter")

            context.register_export_activation_policy(policy)
            context.register_export_target(
                {
                    "id": "self-otel",
                    "target_kind": "tests.telemetry.otlp",
                    "activation_policy": {
                        "provider": kind,
                        "timeout_millis": 30_000,
                        "config": {"country": "US"},
                    },
                },
                activate,
            )

    plugin.register(kind, SelfGatedPlugin())
    try:
        await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind, config={})]))
        assert len(activated) == expected
    finally:
        await plugin.clear_async()
        plugin.deregister(kind)
