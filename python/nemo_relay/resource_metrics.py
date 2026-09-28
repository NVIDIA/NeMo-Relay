# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed access to Relay system resource metric snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from nemo_relay._native import (
    ResourceMetricsRuntime,
    _collect_resource_metrics,
    _configure_resource_metrics,
    _latest_resource_metrics,
    _resource_metrics_history,
)

OperatingSystem: TypeAlias = Literal["linux", "macos", "windows", "unsupported"]
MeasurementUnit: TypeAlias = Literal[
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
]
ResourceLimitResource: TypeAlias = Literal["cpu", "memory", "processes"]
ResourceLimitEventKind: TypeAlias = Literal["throttled", "high", "maximum", "out_of_memory", "terminated"]
AcceleratorVendor: TypeAlias = Literal["nvidia", "amd", "intel", "apple", "other"]
MeasurementValue = TypeVar("MeasurementValue", int, float)


@dataclass(frozen=True)
class ResourceMeasurement(Generic[MeasurementValue]):
    """One timestamped value in the operating system's native unit."""

    value: MeasurementValue | None
    unit: MeasurementUnit | None
    timestamp: datetime


@dataclass(frozen=True)
class ResourceLimitEventCount:
    """One typed cumulative resource-limit event counter."""

    resource: ResourceLimitResource
    event: ResourceLimitEventKind
    count: ResourceMeasurement[int]


@dataclass(frozen=True)
class AcceleratorDeviceMetrics:
    """Device-wide accelerator measurements."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    memory_used: ResourceMeasurement[int]
    compute_utilization: ResourceMeasurement[float]


@dataclass(frozen=True)
class AcceleratorProcessMetrics:
    """Accelerator measurements attributed to one Relay-owned process."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    process_id: int
    memory_used: ResourceMeasurement[int]
    compute_utilization: ResourceMeasurement[float]


@dataclass(frozen=True)
class ResourceMetricsSnapshot:
    """One point-in-time observation of Relay-owned resources."""

    operating_system: OperatingSystem
    cpu_user_time: ResourceMeasurement[int]
    cpu_system_time: ResourceMeasurement[int]
    cpu_total_time: ResourceMeasurement[int]
    cpu_consumption_rate: ResourceMeasurement[float]
    cpu_throttled_time: ResourceMeasurement[int]
    effective_cpu_limit: ResourceMeasurement[float]
    cpu_some_pressure_stall_time: ResourceMeasurement[int]
    cpu_full_pressure_stall_time: ResourceMeasurement[int]
    resident_memory: ResourceMeasurement[int]
    private_memory: ResourceMeasurement[int]
    physical_footprint: ResourceMeasurement[int]
    virtual_memory: ResourceMeasurement[int]
    peak_resident_memory: ResourceMeasurement[int]
    memory_limit: ResourceMeasurement[int]
    environment_accounted_memory: ResourceMeasurement[int]
    memory_some_pressure_stall_time: ResourceMeasurement[int]
    memory_full_pressure_stall_time: ResourceMeasurement[int]
    out_of_memory_event_count: ResourceMeasurement[int]
    active_process_count: ResourceMeasurement[int]
    descendant_process_count: ResourceMeasurement[int]
    thread_count: ResourceMeasurement[int]
    lifetime_process_creation_count: ResourceMeasurement[int]
    open_file_descriptor_count: ResourceMeasurement[int]
    windows_handle_count: ResourceMeasurement[int]
    resource_limit_events: list[ResourceLimitEventCount]
    accelerator_devices: list[AcceleratorDeviceMetrics]
    accelerator_processes: list[AcceleratorProcessMetrics]


@dataclass(frozen=True)
class ResourceMetricsPollingConfig:
    """Managed polling behavior."""

    enabled: bool = False
    interval_millis: int = 5_000
    retained_snapshots: int = 120


@dataclass(frozen=True)
class ResourceMetricsFileConfig:
    """Rotating JSONL output for successful polling snapshots."""

    enabled: bool = False
    path: str | Path = "resource-metrics.jsonl"
    max_file_size_bytes: int = 10 * 1024 * 1024
    retained_files: int = 5


@dataclass(frozen=True)
class ResourceMetricsConfig:
    """Resource metrics polling and file-output configuration."""

    polling: ResourceMetricsPollingConfig = ResourceMetricsPollingConfig()
    file: ResourceMetricsFileConfig = ResourceMetricsFileConfig()


def collect() -> ResourceMetricsSnapshot:
    """Acquire a fresh snapshot without requiring polling."""
    return _snapshot(cast(dict[str, object], _collect_resource_metrics()))


def latest() -> ResourceMetricsSnapshot | None:
    """Return the latest successful polling snapshot, if one exists."""
    value = _latest_resource_metrics()
    return None if value is None else _snapshot(cast(dict[str, object], value))


def history() -> list[ResourceMetricsSnapshot]:
    """Return retained successful polling snapshots from oldest to newest."""
    return [_snapshot(value) for value in cast(list[dict[str, object]], _resource_metrics_history())]


def configure(config: ResourceMetricsConfig) -> ResourceMetricsRuntime:
    """Configure optional polling and file output until the returned runtime is closed."""
    value = asdict(config)
    value["file"]["path"] = str(config.file.path)
    return _configure_resource_metrics(value)


def _measurement(value: object) -> ResourceMeasurement[int] | ResourceMeasurement[float]:
    data = cast(dict[str, object], value)
    return ResourceMeasurement(
        value=cast(int | float | None, data.get("value")),
        unit=cast(MeasurementUnit | None, data.get("unit")),
        timestamp=datetime.fromisoformat(cast(str, data["timestamp"]).replace("Z", "+00:00")),
    )


def _limit_event(value: object) -> ResourceLimitEventCount:
    data = cast(dict[str, object], value)
    return ResourceLimitEventCount(
        resource=cast(ResourceLimitResource, data["resource"]),
        event=cast(ResourceLimitEventKind, data["event"]),
        count=cast(ResourceMeasurement[int], _measurement(data["count"])),
    )


def _accelerator_device(value: object) -> AcceleratorDeviceMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorDeviceMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        memory_used=cast(ResourceMeasurement[int], _measurement(data["memory_used"])),
        compute_utilization=cast(ResourceMeasurement[float], _measurement(data["compute_utilization"])),
    )


def _accelerator_process(value: object) -> AcceleratorProcessMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorProcessMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        process_id=cast(int, data["process_id"]),
        memory_used=cast(ResourceMeasurement[int], _measurement(data["memory_used"])),
        compute_utilization=cast(ResourceMeasurement[float], _measurement(data["compute_utilization"])),
    )


def _snapshot(value: dict[str, object]) -> ResourceMetricsSnapshot:
    return ResourceMetricsSnapshot(
        operating_system=cast(OperatingSystem, value["operating_system"]),
        cpu_user_time=cast(ResourceMeasurement[int], _measurement(value["cpu_user_time"])),
        cpu_system_time=cast(ResourceMeasurement[int], _measurement(value["cpu_system_time"])),
        cpu_total_time=cast(ResourceMeasurement[int], _measurement(value["cpu_total_time"])),
        cpu_consumption_rate=cast(ResourceMeasurement[float], _measurement(value["cpu_consumption_rate"])),
        cpu_throttled_time=cast(ResourceMeasurement[int], _measurement(value["cpu_throttled_time"])),
        effective_cpu_limit=cast(ResourceMeasurement[float], _measurement(value["effective_cpu_limit"])),
        cpu_some_pressure_stall_time=cast(
            ResourceMeasurement[int], _measurement(value["cpu_some_pressure_stall_time"])
        ),
        cpu_full_pressure_stall_time=cast(
            ResourceMeasurement[int], _measurement(value["cpu_full_pressure_stall_time"])
        ),
        resident_memory=cast(ResourceMeasurement[int], _measurement(value["resident_memory"])),
        private_memory=cast(ResourceMeasurement[int], _measurement(value["private_memory"])),
        physical_footprint=cast(ResourceMeasurement[int], _measurement(value["physical_footprint"])),
        virtual_memory=cast(ResourceMeasurement[int], _measurement(value["virtual_memory"])),
        peak_resident_memory=cast(ResourceMeasurement[int], _measurement(value["peak_resident_memory"])),
        memory_limit=cast(ResourceMeasurement[int], _measurement(value["memory_limit"])),
        environment_accounted_memory=cast(
            ResourceMeasurement[int], _measurement(value["environment_accounted_memory"])
        ),
        memory_some_pressure_stall_time=cast(
            ResourceMeasurement[int], _measurement(value["memory_some_pressure_stall_time"])
        ),
        memory_full_pressure_stall_time=cast(
            ResourceMeasurement[int], _measurement(value["memory_full_pressure_stall_time"])
        ),
        out_of_memory_event_count=cast(ResourceMeasurement[int], _measurement(value["out_of_memory_event_count"])),
        active_process_count=cast(ResourceMeasurement[int], _measurement(value["active_process_count"])),
        descendant_process_count=cast(ResourceMeasurement[int], _measurement(value["descendant_process_count"])),
        thread_count=cast(ResourceMeasurement[int], _measurement(value["thread_count"])),
        lifetime_process_creation_count=cast(
            ResourceMeasurement[int], _measurement(value["lifetime_process_creation_count"])
        ),
        open_file_descriptor_count=cast(ResourceMeasurement[int], _measurement(value["open_file_descriptor_count"])),
        windows_handle_count=cast(ResourceMeasurement[int], _measurement(value["windows_handle_count"])),
        resource_limit_events=[_limit_event(item) for item in cast(list[object], value["resource_limit_events"])],
        accelerator_devices=[_accelerator_device(item) for item in cast(list[object], value["accelerator_devices"])],
        accelerator_processes=[
            _accelerator_process(item) for item in cast(list[object], value["accelerator_processes"])
        ],
    )
