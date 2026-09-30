# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed access to Relay system resource metric snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from nemo_relay._native import _collect_resource_metrics

OperatingSystem: TypeAlias = Literal["linux", "macos", "windows", "unsupported"]
MeasurementScope: TypeAlias = Literal["global", "application_process", "process_tree"]
MeasurementUnit: TypeAlias = Literal[
    "milliseconds",
    "bytes",
    "kibibytes",
    "logical_processors",
    "percentage",
    "processes",
    "threads",
    "file_descriptors",
    "handles",
    "events",
    "operations",
    "microseconds",
    "seconds",
    "minutes",
    "kilobytes",
    "megabytes",
    "gigabytes",
    "terabytes",
    "mebibytes",
    "gibibytes",
    "tebibytes",
    "bytes_per_second",
    "kibibytes_per_second",
    "mebibytes_per_second",
    "gibibytes_per_second",
    "bits_per_second",
    "megabits_per_second",
    "gigabits_per_second",
    "millicores",
    "fraction",
    "packets",
    "errors",
]
ResourceLimitResource: TypeAlias = Literal["cpu", "memory", "processes"]
ResourceLimitEventKind: TypeAlias = Literal["throttled", "high", "maximum", "out_of_memory", "terminated"]
AcceleratorVendor: TypeAlias = Literal["nvidia", "amd", "intel", "apple", "other"]
MeasurementValue = TypeVar("MeasurementValue", bound=int | float)


@dataclass(frozen=True)
class ResourceMeasurement(Generic[MeasurementValue]):
    """One available value and its selected unit."""

    value: MeasurementValue
    unit: MeasurementUnit


@dataclass(frozen=True)
class ResourceLimitEventCount:
    """One typed cumulative resource-limit event counter."""

    resource: ResourceLimitResource
    event: ResourceLimitEventKind
    count: ResourceMeasurement[int | float] | None


@dataclass(frozen=True)
class AcceleratorDeviceMetrics:
    """Device-wide accelerator measurements."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    memory_used: ResourceMeasurement[int | float] | None
    compute_utilization: ResourceMeasurement[float] | None


@dataclass(frozen=True)
class AcceleratorProcessMetrics:
    """Accelerator measurements attributed to one process in the selected scope."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    process_id: int
    memory_used: ResourceMeasurement[int | float] | None
    compute_utilization: ResourceMeasurement[float] | None


@dataclass(frozen=True)
class FilesystemCapacityMetrics:
    """Capacity for one configured filesystem path."""

    path: str
    total_capacity: ResourceMeasurement[int | float] | None
    available_capacity: ResourceMeasurement[int | float] | None
    free_capacity: ResourceMeasurement[int | float] | None


@dataclass(frozen=True)
class CpuMetrics:
    """CPU measurements and CPU limit events."""

    user_time: ResourceMeasurement[int | float] | None
    system_time: ResourceMeasurement[int | float] | None
    total_time: ResourceMeasurement[int | float] | None
    consumption_rate: ResourceMeasurement[float] | None
    throttled_time: ResourceMeasurement[int | float] | None
    effective_limit: ResourceMeasurement[float] | None
    some_pressure_stall_time: ResourceMeasurement[int | float] | None
    full_pressure_stall_time: ResourceMeasurement[int | float] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class MemoryMetrics:
    """System and process memory measurements and memory limit events."""

    system_used: ResourceMeasurement[int | float] | None
    system_total: ResourceMeasurement[int | float] | None
    system_available: ResourceMeasurement[int | float] | None
    resident: ResourceMeasurement[int | float] | None
    private: ResourceMeasurement[int | float] | None
    physical_footprint: ResourceMeasurement[int | float] | None
    virtual_memory: ResourceMeasurement[int | float] | None
    peak_resident: ResourceMeasurement[int | float] | None
    limit: ResourceMeasurement[int | float] | None
    environment_accounted: ResourceMeasurement[int | float] | None
    some_pressure_stall_time: ResourceMeasurement[int | float] | None
    full_pressure_stall_time: ResourceMeasurement[int | float] | None
    out_of_memory_event_count: ResourceMeasurement[int | float] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class ProcessMetrics:
    """Process counts, limits, handles, and descriptors."""

    active_count: ResourceMeasurement[int | float] | None
    descendant_count: ResourceMeasurement[int | float] | None
    thread_count: ResourceMeasurement[int | float] | None
    lifetime_creation_count: ResourceMeasurement[int | float] | None
    open_file_descriptor_count: ResourceMeasurement[int | float] | None
    windows_handle_count: ResourceMeasurement[int | float] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class DiskMetrics:
    """Disk I/O for the measured process scope and filesystem capacity."""

    read_data: ResourceMeasurement[int | float] | None
    write_data: ResourceMeasurement[int | float] | None
    read_throughput: ResourceMeasurement[float] | None
    write_throughput: ResourceMeasurement[float] | None
    read_operations: ResourceMeasurement[int | float] | None
    write_operations: ResourceMeasurement[int | float] | None
    filesystems: list[FilesystemCapacityMetrics]


@dataclass(frozen=True)
class GpuMetrics:
    """Accelerator device-wide and process-attributed measurements."""

    device_metrics: list[AcceleratorDeviceMetrics] | None
    process_metrics: list[AcceleratorProcessMetrics] | None


@dataclass(frozen=True)
class NetworkTrafficMetrics:
    """System interface traffic counters and transfer rates."""

    received_data: ResourceMeasurement[int | float] | None
    transmitted_data: ResourceMeasurement[int | float] | None
    receive_throughput: ResourceMeasurement[float] | None
    transmit_throughput: ResourceMeasurement[float] | None
    received_packets: ResourceMeasurement[int | float] | None
    transmitted_packets: ResourceMeasurement[int | float] | None
    receive_errors: ResourceMeasurement[int | float] | None
    transmit_errors: ResourceMeasurement[int | float] | None


@dataclass(frozen=True)
class NetworkInterfaceMetrics:
    """Traffic measured on one system network interface."""

    name: str
    traffic: NetworkTrafficMetrics


@dataclass(frozen=True)
class NetworkMetrics:
    """System-wide traffic, independent of the process measurement scope."""

    measurement_scope: MeasurementScope
    system: NetworkTrafficMetrics
    interfaces: list[NetworkInterfaceMetrics]


@dataclass(frozen=True)
class ProcessSamplingMetadata:
    """Coverage of process queries used for process-summed measurements."""

    visible_processes: int
    sampled_processes: int
    field_sampled_processes: dict[str, int]


@dataclass(frozen=True)
class ResourceMetricsSnapshot:
    """One point-in-time observation, its metadata, and enabled categories."""

    timestamp: datetime
    operating_system: OperatingSystem
    measurement_scope: MeasurementScope
    process_sampling: ProcessSamplingMetadata | None
    cpu: CpuMetrics | None
    memory: MemoryMetrics | None
    process: ProcessMetrics | None
    disk: DiskMetrics | None
    gpu: GpuMetrics | None
    network: NetworkMetrics | None


async def collect() -> ResourceMetricsSnapshot:
    """Acquire a fresh sample using the active resource_metrics plugin configuration."""
    return _snapshot(cast(dict[str, object], await _collect_resource_metrics()))


def _measurement(value: object) -> ResourceMeasurement[int | float] | ResourceMeasurement[float] | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return ResourceMeasurement(
        value=cast(int | float, data["value"]),
        unit=cast(MeasurementUnit, data["unit"]),
    )


def _limit_event(value: object) -> ResourceLimitEventCount:
    data = cast(dict[str, object], value)
    return ResourceLimitEventCount(
        resource=cast(ResourceLimitResource, data["resource"]),
        event=cast(ResourceLimitEventKind, data["event"]),
        count=cast(ResourceMeasurement[int | float] | None, _measurement(data["count"])),
    )


def _accelerator_device(value: object) -> AcceleratorDeviceMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorDeviceMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        memory_used=cast(ResourceMeasurement[int | float] | None, _measurement(data["memory_used"])),
        compute_utilization=cast(ResourceMeasurement[float] | None, _measurement(data["compute_utilization"])),
    )


def _accelerator_process(value: object) -> AcceleratorProcessMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorProcessMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        process_id=cast(int, data["process_id"]),
        memory_used=cast(ResourceMeasurement[int | float] | None, _measurement(data["memory_used"])),
        compute_utilization=cast(ResourceMeasurement[float] | None, _measurement(data["compute_utilization"])),
    )


def _filesystem(value: object) -> FilesystemCapacityMetrics:
    data = cast(dict[str, object], value)
    return FilesystemCapacityMetrics(
        path=cast(str, data["path"]),
        total_capacity=cast(ResourceMeasurement[int | float] | None, _measurement(data["total_capacity"])),
        available_capacity=cast(ResourceMeasurement[int | float] | None, _measurement(data["available_capacity"])),
        free_capacity=cast(ResourceMeasurement[int | float] | None, _measurement(data["free_capacity"])),
    )


def _events(value: object) -> list[ResourceLimitEventCount]:
    return [_limit_event(item) for item in cast(list[object], value)]


def _cpu(value: object) -> CpuMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return CpuMetrics(
        user_time=cast(ResourceMeasurement[int | float] | None, _measurement(data["user_time"])),
        system_time=cast(ResourceMeasurement[int | float] | None, _measurement(data["system_time"])),
        total_time=cast(ResourceMeasurement[int | float] | None, _measurement(data["total_time"])),
        consumption_rate=cast(ResourceMeasurement[float] | None, _measurement(data["consumption_rate"])),
        throttled_time=cast(ResourceMeasurement[int | float] | None, _measurement(data["throttled_time"])),
        effective_limit=cast(ResourceMeasurement[float] | None, _measurement(data["effective_limit"])),
        some_pressure_stall_time=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["some_pressure_stall_time"])
        ),
        full_pressure_stall_time=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["full_pressure_stall_time"])
        ),
        limit_events=_events(data["limit_events"]),
    )


def _memory(value: object) -> MemoryMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return MemoryMetrics(
        system_used=cast(ResourceMeasurement[int | float] | None, _measurement(data["system_used"])),
        system_total=cast(ResourceMeasurement[int | float] | None, _measurement(data["system_total"])),
        system_available=cast(ResourceMeasurement[int | float] | None, _measurement(data["system_available"])),
        resident=cast(ResourceMeasurement[int | float] | None, _measurement(data["resident"])),
        private=cast(ResourceMeasurement[int | float] | None, _measurement(data["private"])),
        physical_footprint=cast(ResourceMeasurement[int | float] | None, _measurement(data["physical_footprint"])),
        virtual_memory=cast(ResourceMeasurement[int | float] | None, _measurement(data["virtual_memory"])),
        peak_resident=cast(ResourceMeasurement[int | float] | None, _measurement(data["peak_resident"])),
        limit=cast(ResourceMeasurement[int | float] | None, _measurement(data["limit"])),
        environment_accounted=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["environment_accounted"])
        ),
        some_pressure_stall_time=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["some_pressure_stall_time"])
        ),
        full_pressure_stall_time=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["full_pressure_stall_time"])
        ),
        out_of_memory_event_count=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["out_of_memory_event_count"])
        ),
        limit_events=_events(data["limit_events"]),
    )


def _process(value: object) -> ProcessMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return ProcessMetrics(
        active_count=cast(ResourceMeasurement[int | float] | None, _measurement(data["active_count"])),
        descendant_count=cast(ResourceMeasurement[int | float] | None, _measurement(data["descendant_count"])),
        thread_count=cast(ResourceMeasurement[int | float] | None, _measurement(data["thread_count"])),
        lifetime_creation_count=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["lifetime_creation_count"])
        ),
        open_file_descriptor_count=cast(
            ResourceMeasurement[int | float] | None, _measurement(data["open_file_descriptor_count"])
        ),
        windows_handle_count=cast(ResourceMeasurement[int | float] | None, _measurement(data["windows_handle_count"])),
        limit_events=_events(data["limit_events"]),
    )


def _disk(value: object) -> DiskMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return DiskMetrics(
        read_data=cast(ResourceMeasurement[int | float] | None, _measurement(data["read_data"])),
        write_data=cast(ResourceMeasurement[int | float] | None, _measurement(data["write_data"])),
        read_throughput=cast(ResourceMeasurement[float] | None, _measurement(data["read_throughput"])),
        write_throughput=cast(ResourceMeasurement[float] | None, _measurement(data["write_throughput"])),
        read_operations=cast(ResourceMeasurement[int | float] | None, _measurement(data["read_operations"])),
        write_operations=cast(ResourceMeasurement[int | float] | None, _measurement(data["write_operations"])),
        filesystems=[_filesystem(item) for item in cast(list[object], data["filesystems"])],
    )


def _gpu(value: object) -> GpuMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    devices = data["device_metrics"]
    processes = data["process_metrics"]
    return GpuMetrics(
        device_metrics=(
            None if devices is None else [_accelerator_device(item) for item in cast(list[object], devices)]
        ),
        process_metrics=(
            None if processes is None else [_accelerator_process(item) for item in cast(list[object], processes)]
        ),
    )


def _network_traffic(value: object) -> NetworkTrafficMetrics:
    data = cast(dict[str, object], value)
    return NetworkTrafficMetrics(
        received_data=cast(ResourceMeasurement[int | float] | None, _measurement(data["received_data"])),
        transmitted_data=cast(ResourceMeasurement[int | float] | None, _measurement(data["transmitted_data"])),
        receive_throughput=cast(ResourceMeasurement[float] | None, _measurement(data["receive_throughput"])),
        transmit_throughput=cast(ResourceMeasurement[float] | None, _measurement(data["transmit_throughput"])),
        received_packets=cast(ResourceMeasurement[int | float] | None, _measurement(data["received_packets"])),
        transmitted_packets=cast(ResourceMeasurement[int | float] | None, _measurement(data["transmitted_packets"])),
        receive_errors=cast(ResourceMeasurement[int | float] | None, _measurement(data["receive_errors"])),
        transmit_errors=cast(ResourceMeasurement[int | float] | None, _measurement(data["transmit_errors"])),
    )


def _network(value: object) -> NetworkMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return NetworkMetrics(
        measurement_scope=cast(MeasurementScope, data["measurement_scope"]),
        system=_network_traffic(data["system"]),
        interfaces=[
            NetworkInterfaceMetrics(name=cast(str, item["name"]), traffic=_network_traffic(item["traffic"]))
            for item in cast(list[dict[str, object]], data["interfaces"])
        ],
    )


def _snapshot(value: dict[str, object]) -> ResourceMetricsSnapshot:
    timestamp = datetime.fromisoformat(cast(str, value["timestamp"]).replace("Z", "+00:00"))
    sampling = value.get("process_sampling")
    return ResourceMetricsSnapshot(
        timestamp=timestamp,
        operating_system=cast(OperatingSystem, value["operating_system"]),
        measurement_scope=cast(MeasurementScope, value["measurement_scope"]),
        process_sampling=(
            None
            if sampling is None
            else ProcessSamplingMetadata(
                visible_processes=cast(int, cast(dict[str, object], sampling)["visible_processes"]),
                sampled_processes=cast(int, cast(dict[str, object], sampling)["sampled_processes"]),
                field_sampled_processes=cast(
                    dict[str, int], cast(dict[str, object], sampling)["field_sampled_processes"]
                ),
            )
        ),
        cpu=_cpu(value["cpu"]),
        memory=_memory(value["memory"]),
        process=_process(value["process"]),
        disk=_disk(value["disk"]),
        gpu=_gpu(value["gpu"]),
        network=_network(value["network"]),
    )
