# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for typed resource metrics and managed polling."""

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import nemo_relay
from nemo_relay import resource_metrics


def _assert_measurement_state(
    measurement: resource_metrics.ResourceMeasurement[int] | resource_metrics.ResourceMeasurement[float],
) -> None:
    assert is_dataclass(measurement)
    assert tuple(asdict(measurement)) == ("value", "unit", "timestamp")
    assert measurement.timestamp.tzinfo is not None
    assert measurement.unit is None or measurement.unit in {
        "nanoseconds",
        "microseconds",
        "clock_ticks",
        "hundred_nanosecond_intervals",
        "nanoseconds_per_second",
        "clock_ticks_per_second",
        "hundred_nanosecond_intervals_per_second",
        "bytes",
        "kibibytes",
        "pages",
        "logical_processors",
        "percentage",
        "processes",
        "threads",
        "file_descriptors",
        "handles",
        "events",
    }
    assert not hasattr(measurement, "status")
    assert not hasattr(measurement, "reason")
    if measurement.value is None:
        assert measurement.unit is None
    else:
        assert measurement.unit is not None


def _expected_native_units() -> tuple[str, str, str]:
    if sys.platform == "darwin":
        return "nanoseconds", "nanoseconds_per_second", "bytes"
    if sys.platform.startswith("linux"):
        return "clock_ticks", "clock_ticks_per_second", "pages"
    if sys.platform == "win32":
        return "hundred_nanosecond_intervals", "hundred_nanosecond_intervals_per_second", "bytes"
    return "", "", ""


def _expected_operating_system() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return "unsupported"


_SCALAR_MEASUREMENT_FIELDS = (
    "cpu_user_time",
    "cpu_system_time",
    "cpu_total_time",
    "cpu_consumption_rate",
    "cpu_throttled_time",
    "effective_cpu_limit",
    "cpu_some_pressure_stall_time",
    "cpu_full_pressure_stall_time",
    "resident_memory",
    "private_memory",
    "physical_footprint",
    "virtual_memory",
    "peak_resident_memory",
    "memory_limit",
    "environment_accounted_memory",
    "memory_some_pressure_stall_time",
    "memory_full_pressure_stall_time",
    "out_of_memory_event_count",
    "active_process_count",
    "descendant_process_count",
    "thread_count",
    "lifetime_process_creation_count",
    "open_file_descriptor_count",
    "windows_handle_count",
)


def test_fresh_and_managed_resource_metrics_are_typed_and_bounded(tmp_path: Path) -> None:
    fresh = resource_metrics.collect()
    assert fresh.operating_system == _expected_operating_system()
    assert not hasattr(fresh, "measurement_scope")
    assert not hasattr(fresh, "collected_at")
    assert not hasattr(fresh, "sample_interval_millis")
    assert fresh.active_process_count.value is not None
    assert fresh.active_process_count.value > 0
    cpu_time_unit, cpu_rate_unit, memory_unit = _expected_native_units()
    assert fresh.cpu_user_time.unit == cpu_time_unit
    assert fresh.cpu_system_time.unit == cpu_time_unit
    assert fresh.cpu_total_time.unit == cpu_time_unit
    if fresh.cpu_consumption_rate.value is None:
        assert fresh.cpu_consumption_rate.unit is None
    else:
        assert fresh.cpu_consumption_rate.unit == cpu_rate_unit
        assert type(fresh.cpu_consumption_rate.value) is float
    assert fresh.resident_memory.unit == memory_unit
    assert fresh.cpu_consumption_rate.value is None
    for field in _SCALAR_MEASUREMENT_FIELDS:
        measurement = getattr(fresh, field)
        _assert_measurement_state(measurement)
        if measurement.value is not None and field.startswith("cpu_") and field.endswith("_time"):
            assert type(measurement.value) is int
    assert isinstance(fresh.resource_limit_events, list)
    assert all(is_dataclass(event) for event in fresh.resource_limit_events)
    assert isinstance(fresh.accelerator_devices, list)
    assert isinstance(fresh.accelerator_processes, list)

    if sys.platform == "darwin":
        assert fresh.physical_footprint.value is not None
        assert fresh.virtual_memory.value is not None
        assert fresh.open_file_descriptor_count.value is not None
        for field in (
            "private_memory",
            "peak_resident_memory",
            "cpu_throttled_time",
            "effective_cpu_limit",
            "cpu_some_pressure_stall_time",
            "cpu_full_pressure_stall_time",
            "memory_limit",
            "environment_accounted_memory",
            "memory_some_pressure_stall_time",
            "memory_full_pressure_stall_time",
            "out_of_memory_event_count",
            "lifetime_process_creation_count",
            "windows_handle_count",
        ):
            unavailable = getattr(fresh, field)
            assert unavailable.value is None
            assert unavailable.unit is None
    assert resource_metrics.latest() is None
    assert resource_metrics.history() == []

    output_path = tmp_path / "resource-metrics.jsonl"
    config = resource_metrics.ResourceMetricsConfig(
        polling=resource_metrics.ResourceMetricsPollingConfig(
            enabled=True,
            interval_millis=10,
            retained_snapshots=2,
        ),
        file=resource_metrics.ResourceMetricsFileConfig(
            enabled=True,
            path=output_path,
            max_file_size_bytes=1_000_000,
            retained_files=2,
        ),
    )
    with resource_metrics.configure(config):
        deadline = time.monotonic() + 2
        while len(resource_metrics.history()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        snapshots = resource_metrics.history()
        assert len(snapshots) == 2
        assert resource_metrics.latest() == snapshots[-1]
        before_fresh = resource_metrics.history()
        independent = resource_metrics.collect()
        assert independent.active_process_count.value is not None
        assert resource_metrics.history() == before_fresh
        for snapshot in snapshots:
            assert snapshot.operating_system == _expected_operating_system()
            for field in _SCALAR_MEASUREMENT_FIELDS:
                _assert_measurement_state(getattr(snapshot, field))

    file_snapshots = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(file_snapshots) >= 2
    assert all("collected_at" not in snapshot for snapshot in file_snapshots)
    assert all("sample_interval_millis" not in snapshot for snapshot in file_snapshots)
    assert all("measurement_scope" not in snapshot for snapshot in file_snapshots)
    assert all(snapshot["operating_system"] == _expected_operating_system() for snapshot in file_snapshots)
    for snapshot in file_snapshots:
        assert isinstance(snapshot["resource_limit_events"], list)
        assert isinstance(snapshot["accelerator_devices"], list)
        assert isinstance(snapshot["accelerator_processes"], list)
        for field in _SCALAR_MEASUREMENT_FIELDS:
            measurement = snapshot[field]
            assert tuple(measurement) == ("value", "unit", "timestamp")
            assert "reason" not in measurement
            assert "status" not in measurement
            if measurement["value"] is None:
                assert measurement["unit"] is None

    assert nemo_relay.resource_metrics.collect is resource_metrics.collect


def test_snapshot_parser_reconstructs_typed_limit_and_accelerator_records() -> None:
    timestamp = datetime.now(UTC).isoformat()
    unavailable = {"value": None, "unit": None, "timestamp": timestamp}
    payload: dict[str, object] = {
        "operating_system": _expected_operating_system(),
        **{field: unavailable for field in _SCALAR_MEASUREMENT_FIELDS},
        "resource_limit_events": [
            {
                "resource": "cpu",
                "event": "throttled",
                "count": {"value": 2, "unit": "events", "timestamp": timestamp},
            }
        ],
        "accelerator_devices": [
            {
                "vendor": "nvidia",
                "device_identifier": "test-device",
                "device_index": 0,
                "memory_used": unavailable,
                "compute_utilization": unavailable,
            }
        ],
        "accelerator_processes": [
            {
                "vendor": "nvidia",
                "device_identifier": "test-device",
                "device_index": 0,
                "process_id": 42,
                "memory_used": unavailable,
                "compute_utilization": unavailable,
            }
        ],
    }
    parsed = resource_metrics._snapshot(payload)

    assert isinstance(parsed, resource_metrics.ResourceMetricsSnapshot)
    assert parsed.operating_system == _expected_operating_system()
    assert isinstance(parsed.resource_limit_events[0], resource_metrics.ResourceLimitEventCount)
    assert parsed.resource_limit_events[0].resource == "cpu"
    assert parsed.resource_limit_events[0].count.value == 2
    assert isinstance(parsed.accelerator_devices[0], resource_metrics.AcceleratorDeviceMetrics)
    assert parsed.accelerator_devices[0].memory_used.value is None
    assert parsed.accelerator_devices[0].memory_used.unit is None
    assert isinstance(parsed.accelerator_processes[0], resource_metrics.AcceleratorProcessMetrics)
    assert parsed.accelerator_processes[0].process_id == 42
    assert parsed.accelerator_processes[0].compute_utilization.value is None
