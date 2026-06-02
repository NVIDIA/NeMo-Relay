# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NAT telemetry exporter shim."""

import json
from uuid import uuid4

import nemo_relay
from nemo_relay import ScopeType, scope, subscribers
from nemo_relay.nat_exporter import NatTelemetryExporter


def test_nat_telemetry_exporter_writes_relay_event_jsonl(tmp_path):
    exporter = NatTelemetryExporter(tmp_path / "relay-events.jsonl")
    subscriber_name = f"nat_exporter_{uuid4().hex}"

    exporter.register(subscriber_name)
    try:
        handle = scope.push("nat_exporter_root", ScopeType.Agent, data={"input": True})
        try:
            scope.event("nat_exporter_mark", handle=handle, data={"step": 1})
        finally:
            scope.pop(handle, output={"done": True})
            exporter.force_flush()
    finally:
        exporter.deregister(subscriber_name)
        exporter.shutdown()
        subscribers.deregister(subscriber_name)

    lines = [json.loads(line) for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()]

    assert [line["kind"] for line in lines] == ["scope", "mark", "scope"]
    assert lines[0]["name"] == "nat_exporter_root"
    assert lines[1]["data"] == {"step": 1}
    assert lines[2]["scope_category"] == "end"
    assert "<NatTelemetryExporter" in repr(exporter)


def test_nat_telemetry_exporter_is_top_level_export():
    assert getattr(nemo_relay, "NatTelemetryExporter") is NatTelemetryExporter
