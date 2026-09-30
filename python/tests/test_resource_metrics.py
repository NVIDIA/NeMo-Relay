# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the typed resource metrics plugin API."""

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import get_args

import pytest

import nemo_relay
from nemo_relay import plugin, resource_metrics


def _assert_measurement_state(
    measurement: resource_metrics.ResourceMeasurement[int] | resource_metrics.ResourceMeasurement[float] | None,
) -> None:
    if measurement is None:
        return
    assert is_dataclass(measurement)
    assert tuple(asdict(measurement)) == ("value", "unit")
    assert measurement.unit in get_args(resource_metrics.MeasurementUnit)
    assert not hasattr(measurement, "timestamp")
    assert not hasattr(measurement, "status")
    assert not hasattr(measurement, "reason")


@pytest.mark.parametrize("unit", ["seconds", "gibibytes", "megabits_per_second", "millicores", "fraction", "packets"])
def test_measurement_helper_accepts_units_beyond_the_default_set(unit: resource_metrics.MeasurementUnit) -> None:
    _assert_measurement_state(resource_metrics.ResourceMeasurement(value=1, unit=unit))


def _config(**resource_config: object) -> plugin.PluginConfig:
    return plugin.PluginConfig(components=[plugin.ComponentSpec(kind="resource_metrics", config=resource_config)])


async def test_collect_raises_when_component_is_disabled() -> None:
    config = plugin.PluginConfig(components=[plugin.ComponentSpec(kind="resource_metrics", enabled=False)])
    async with plugin.activate(config):
        with pytest.raises(RuntimeError, match="requires an active resource_metrics component"):
            await resource_metrics.collect()


async def test_collect_uses_active_plugin_config_without_polling() -> None:
    async with plugin.activate(_config(polling={"enabled": False})):
        snapshot = await resource_metrics.collect()
        assert snapshot.operating_system in {"linux", "macos", "windows", "unsupported"}
        assert snapshot.measurement_scope == "process_tree"
        assert snapshot.process_sampling is not None
        assert "cpu.total_time" in snapshot.process_sampling.field_sampled_processes
        assert snapshot.process_sampling.field_sampled_processes["disk.read_throughput"] == 0
        assert snapshot.process_sampling.field_sampled_processes["disk.write_throughput"] == 0
        assert all(
            0 <= count <= snapshot.process_sampling.sampled_processes
            for count in snapshot.process_sampling.field_sampled_processes.values()
        )
        assert snapshot.timestamp.tzinfo is not None
        assert snapshot.process is not None
        assert snapshot.process.active_count is not None
        assert snapshot.process.active_count.value > 0
        assert snapshot.cpu is not None
        assert snapshot.memory is not None
        assert snapshot.memory.system_total is not None
        assert snapshot.memory.system_total.unit == "kibibytes"
        if snapshot.cpu.user_time is not None:
            assert snapshot.cpu.user_time.unit == "milliseconds"
        if snapshot.memory.resident is not None:
            assert snapshot.memory.resident.unit == "kibibytes"
        assert snapshot.disk is not None
        assert snapshot.gpu is not None
        assert snapshot.network is not None
        assert snapshot.network.measurement_scope == "global"
        for measurement in (
            snapshot.cpu.user_time,
            snapshot.cpu.total_time,
            snapshot.cpu.consumption_rate,
            snapshot.memory.resident,
            snapshot.memory.system_used,
            snapshot.memory.system_total,
            snapshot.memory.system_available,
            snapshot.memory.limit,
            snapshot.process.active_count,
            snapshot.process.thread_count,
            snapshot.disk.read_data,
            snapshot.disk.write_data,
            snapshot.disk.read_operations,
            snapshot.disk.write_operations,
        ):
            _assert_measurement_state(measurement)
        assert isinstance(snapshot.disk.filesystems, list)
        assert snapshot.gpu.device_metrics is None or isinstance(snapshot.gpu.device_metrics, list)
        assert snapshot.gpu.process_metrics is None or isinstance(snapshot.gpu.process_metrics, list)

    assert nemo_relay.resource_metrics.collect is resource_metrics.collect


async def test_disabled_categories_and_subgroups_are_none_or_empty() -> None:
    async with plugin.activate(
        _config(
            polling={"enabled": False},
            cpu={"enabled": False},
            memory={"enabled": False},
            process={"enabled": False},
            disk={"enabled": True, "process_io": False, "filesystem_paths": []},
            gpu={"enabled": False},
        )
    ):
        snapshot = await resource_metrics.collect()
        assert snapshot.cpu is None
        assert snapshot.memory is None
        assert snapshot.process is None
        assert snapshot.gpu is None
        assert snapshot.disk is not None
        assert snapshot.disk.read_data is None
        assert snapshot.disk.write_data is None
        assert snapshot.disk.read_throughput is None
        assert snapshot.disk.write_throughput is None
        assert snapshot.disk.read_operations is None
        assert snapshot.disk.write_operations is None
        assert snapshot.disk.filesystems == []


async def test_global_scope_reports_system_memory() -> None:
    async with plugin.activate(_config(measurement_scope="global", polling={"enabled": False})):
        snapshot = await resource_metrics.collect()
        assert snapshot.measurement_scope == "global"
        assert snapshot.memory is not None
        assert snapshot.memory.system_total is not None
        assert snapshot.memory.system_total.unit == "kibibytes"
        assert snapshot.memory.resident is None


def test_snapshot_parser_reconstructs_typed_records() -> None:
    timestamp = datetime.now(UTC).isoformat()
    available = {"value": 3, "unit": "events"}
    unavailable = None
    payload: dict[str, object] = {
        "timestamp": timestamp,
        "operating_system": "linux",
        "measurement_scope": "process_tree",
        "process_sampling": {
            "visible_processes": 2,
            "sampled_processes": 1,
            "field_sampled_processes": {"cpu.total_time": 1, "disk.read_data": 0},
        },
        "cpu": {
            "user_time": {"value": 1, "unit": "milliseconds"},
            "system_time": unavailable,
            "total_time": unavailable,
            "consumption_rate": unavailable,
            "throttled_time": unavailable,
            "effective_limit": unavailable,
            "some_pressure_stall_time": unavailable,
            "full_pressure_stall_time": unavailable,
            "limit_events": [{"resource": "cpu", "event": "throttled", "count": available}],
        },
        "memory": None,
        "process": None,
        "disk": {
            "read_data": unavailable,
            "write_data": unavailable,
            "read_throughput": unavailable,
            "write_throughput": unavailable,
            "read_operations": unavailable,
            "write_operations": unavailable,
            "filesystems": [
                {
                    "path": "/",
                    "total_capacity": unavailable,
                    "available_capacity": unavailable,
                    "free_capacity": unavailable,
                }
            ],
        },
        "network": None,
        "gpu": {
            "device_metrics": None,
            "process_metrics": [
                {
                    "vendor": "nvidia",
                    "device_identifier": "gpu-test",
                    "device_index": 0,
                    "process_id": 42,
                    "memory_used": unavailable,
                    "compute_utilization": unavailable,
                }
            ],
        },
    }
    parsed = resource_metrics._snapshot(payload)
    assert isinstance(parsed, resource_metrics.ResourceMetricsSnapshot)
    assert parsed.timestamp.tzinfo is not None
    assert parsed.measurement_scope == "process_tree"
    assert parsed.process_sampling is not None
    assert parsed.process_sampling.visible_processes == 2
    assert parsed.process_sampling.sampled_processes == 1
    assert parsed.process_sampling.field_sampled_processes == {"cpu.total_time": 1, "disk.read_data": 0}
    assert isinstance(parsed.cpu, resource_metrics.CpuMetrics)
    assert isinstance(parsed.cpu.limit_events[0], resource_metrics.ResourceLimitEventCount)
    assert parsed.memory is None
    assert parsed.network is None
    assert isinstance(parsed.disk, resource_metrics.DiskMetrics)
    assert isinstance(parsed.disk.filesystems[0], resource_metrics.FilesystemCapacityMetrics)
    assert isinstance(parsed.gpu, resource_metrics.GpuMetrics)
    assert parsed.gpu.device_metrics is None
    assert isinstance(parsed.gpu.process_metrics[0], resource_metrics.AcceleratorProcessMetrics)


async def test_configured_units_and_system_network_scope() -> None:
    async with plugin.activate(
        _config(
            polling={"enabled": False},
            units={
                "cpu": {"user_time": "seconds"},
                "memory": {"system_total": "mebibytes"},
                "network": {"system": {"received_data": "megabytes"}},
            },
        )
    ):
        snapshot = await resource_metrics.collect()
        assert snapshot.memory is not None
        assert snapshot.memory.system_total is not None
        assert snapshot.memory.system_total.unit == "mebibytes"
        assert isinstance(snapshot.memory.system_total.value, int | float)
        assert snapshot.network is not None
        assert snapshot.network.measurement_scope == "global"
        if snapshot.cpu is not None and snapshot.cpu.user_time is not None:
            assert snapshot.cpu.user_time.unit == "seconds"
        if snapshot.network.system.received_data is not None:
            assert snapshot.network.system.received_data.unit == "megabytes"
