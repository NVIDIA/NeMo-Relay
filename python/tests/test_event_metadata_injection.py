# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast
from uuid import uuid4

import nemo_relay
from nemo_relay import event_metadata, plugin, scope, scope_local, subscribers


async def test_global_python_injectors_support_sync_async_and_failure_safe_output(subscribed_events):
    sync_name = f"python-sync-{uuid4()}"
    async_name = f"python-async-{uuid4()}"
    failure_name = f"python-failure-{uuid4()}"
    after_failure_name = f"python-after-failure-{uuid4()}"
    invalid_name = f"python-invalid-{uuid4()}"

    async def inject_async(_event: nemo_relay.Event) -> nemo_relay.EventMetadata:
        return {
            "python.injector.async": True,
            "python.injector.shared": "async-later",
        }

    event_metadata.register_injector(
        sync_name,
        10,
        lambda _event: {
            "python.existing": "ignored",
            "python.injector.shared": "sync-first",
            "python.injector.sync": "added",
        },
    )
    event_metadata.register_injector(async_name, 20, inject_async)
    invalid_injector = cast(
        nemo_relay.EventMetadataInjectorCallback,
        lambda _event: ["not", "a", "mapping"],
    )

    def fail(_event: nemo_relay.Event) -> nemo_relay.EventMetadata:
        raise RuntimeError("python injector failure")

    event_metadata.register_injector(failure_name, 30, fail)
    event_metadata.register_injector(
        after_failure_name,
        40,
        lambda _event: {"python.injector.after_failure": "added"},
    )
    event_metadata.register_injector(invalid_name, 50, invalid_injector)
    try:
        scope.event("python-global-injection", metadata={"python.existing": "preserved"})
        await subscribers.flush_async()
    finally:
        event_metadata.deregister_injector(invalid_name)
        event_metadata.deregister_injector(after_failure_name)
        event_metadata.deregister_injector(failure_name)
        event_metadata.deregister_injector(async_name)
        event_metadata.deregister_injector(sync_name)

    event = next(event for event in subscribed_events if event.name == "python-global-injection")
    assert event.metadata == {
        "python.existing": "preserved",
        "python.injector.after_failure": "added",
        "python.injector.async": True,
        "python.injector.shared": "sync-first",
        "python.injector.sync": "added",
    }


async def test_python_injectors_accept_numeric_lists_with_integer_and_fractional_values(
    subscribed_events,
):
    integers_name = f"python-integers-{uuid4()}"
    doubles_name = f"python-doubles-{uuid4()}"
    mixed_name = f"python-mixed-numbers-{uuid4()}"
    mixed_injector = cast(
        nemo_relay.EventMetadataInjectorCallback,
        lambda _event: {"python.injector.mixed_numbers": [1, 2.5]},
    )

    event_metadata.register_injector(
        integers_name,
        10,
        lambda _event: {"python.injector.integers": [1, 2]},
    )
    event_metadata.register_injector(
        doubles_name,
        20,
        lambda _event: {"python.injector.doubles": [1.0, 2.5]},
    )
    event_metadata.register_injector(mixed_name, 30, mixed_injector)
    try:
        scope.event("python-homogeneous-numeric-lists")
        await subscribers.flush_async()
    finally:
        event_metadata.deregister_injector(mixed_name)
        event_metadata.deregister_injector(doubles_name)
        event_metadata.deregister_injector(integers_name)

    event = next(event for event in subscribed_events if event.name == "python-homogeneous-numeric-lists")
    assert event.metadata == {
        "python.injector.doubles": [1.0, 2.5],
        "python.injector.integers": [1, 2],
        "python.injector.mixed_numbers": [1, 2.5],
    }


async def test_scope_local_python_injector_applies_only_to_owned_events(subscribed_events):
    with scope.scope("python-scope-owner", nemo_relay.ScopeType.Agent) as owner:
        scope_local.register_event_metadata_injector(
            owner,
            "python-scope-local-first",
            10,
            lambda _event: {
                "python.injector.scope_local": "active",
                "python.injector.scope_order": "first",
            },
        )
        scope_local.register_event_metadata_injector(
            owner,
            "python-scope-local-later",
            20,
            lambda _event: {"python.injector.scope_order": "later"},
        )
        scope.event("python-scope-inside")
        with scope.scope("python-scope-child", nemo_relay.ScopeType.Function):
            pass

    scope.event("python-scope-outside")
    await subscribers.flush_async()

    events = {event.name: event for event in subscribed_events}
    assert events["python-scope-inside"].metadata["python.injector.scope_local"] == "active"
    assert events["python-scope-inside"].metadata["python.injector.scope_order"] == "first"
    assert events["python-scope-child"].metadata["python.injector.scope_local"] == "active"
    assert events["python-scope-owner"].metadata["python.injector.scope_local"] == "active"
    assert events["python-scope-outside"].metadata is None


async def test_scope_local_python_injector_can_be_deregistered_while_owner_is_active(subscribed_events):
    with scope.scope("python-scope-deregister-owner", nemo_relay.ScopeType.Agent) as owner:
        scope_local.register_event_metadata_injector(
            owner,
            "python-scope-deregister",
            10,
            lambda _event: {"python.injector.scope_local": "active"},
        )
        scope.event("python-scope-before-deregister")
        assert scope_local.deregister_event_metadata_injector(owner, "python-scope-deregister") is True
        assert scope_local.deregister_event_metadata_injector(owner, "python-scope-deregister") is False
        scope.event("python-scope-after-deregister")

    await subscribers.flush_async()

    events = {event.name: event for event in subscribed_events}
    assert events["python-scope-before-deregister"].metadata == {"python.injector.scope_local": "active"}
    assert events["python-scope-after-deregister"].metadata is None


async def test_in_process_python_plugin_registers_configured_injector_and_cleans_up(subscribed_events):
    class ConfiguredMetadataPlugin:
        def validate(self, _config: nemo_relay.JsonObject):
            return None

        def register(
            self,
            config: nemo_relay.JsonObject,
            context: plugin.PluginContext,
        ) -> None:
            configured = cast(nemo_relay.EventMetadata, config["metadata"])
            context.register_event_metadata_injector(
                "configured",
                10,
                lambda _event: configured,
            )

    kind = f"python.test_event_metadata.{uuid4()}"
    plugin.register(kind, cast(plugin.Plugin, ConfiguredMetadataPlugin()))
    try:
        await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    plugin.ComponentSpec(
                        kind=kind,
                        config={"metadata": {"python.injector.plugin": "configured"}},
                    )
                ]
            )
        )
        scope.event("python-plugin-configured")
        await subscribers.flush_async()

        await plugin.clear_async()
        scope.event("python-plugin-cleared")
        await subscribers.flush_async()
    finally:
        await plugin.clear_async()
        plugin.deregister(kind)

    events = {event.name: event for event in subscribed_events}
    assert events["python-plugin-configured"].metadata == {"python.injector.plugin": "configured"}
    assert events["python-plugin-cleared"].metadata is None
