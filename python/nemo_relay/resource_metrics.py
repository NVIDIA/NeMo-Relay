# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed access to Relay system resource metric snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from nemo_relay._native import _collect_resource_metrics

OperatingSystem: TypeAlias = Literal["linux", "macos", "windows", "unsupported"]
MeasurementScope: TypeAlias = Literal["global", "application_process", "process_tree"]


class DurationUnit(StrEnum):
    """Units for duration measurements."""

    MICROSECONDS = "microseconds"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    MINUTES = "minutes"


class CapacityUnit(StrEnum):
    """Units for capacity measurements."""

    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    TERABYTES = "terabytes"
    KIBIBYTES = "kibibytes"
    MEBIBYTES = "mebibytes"
    GIBIBYTES = "gibibytes"
    TEBIBYTES = "tebibytes"


class DataUnit(StrEnum):
    """Units for data measurements."""

    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    TERABYTES = "terabytes"
    KIBIBYTES = "kibibytes"
    MEBIBYTES = "mebibytes"
    GIBIBYTES = "gibibytes"
    TEBIBYTES = "tebibytes"


class BandwidthUnit(StrEnum):
    """Units for bandwidth measurements."""

    BYTES_PER_SECOND = "bytes_per_second"
    KIBIBYTES_PER_SECOND = "kibibytes_per_second"
    MEBIBYTES_PER_SECOND = "mebibytes_per_second"
    GIBIBYTES_PER_SECOND = "gibibytes_per_second"
    BITS_PER_SECOND = "bits_per_second"
    MEGABITS_PER_SECOND = "megabits_per_second"
    GIGABITS_PER_SECOND = "gigabits_per_second"


class CpuUnit(StrEnum):
    """Units for CPU measurements."""

    LOGICAL_PROCESSORS = "logical_processors"
    MILLICORES = "millicores"


class UtilizationUnit(StrEnum):
    """Units for utilization measurements."""

    PERCENTAGE = "percentage"
    FRACTION = "fraction"


class CountUnit(StrEnum):
    """Units for count measurements."""

    PROCESSES = "processes"
    THREADS = "threads"
    FILE_DESCRIPTORS = "file_descriptors"
    HANDLES = "handles"
    EVENTS = "events"
    OPERATIONS = "operations"
    PACKETS = "packets"
    ERRORS = "errors"


ResourceLimitResource: TypeAlias = Literal["cpu", "memory", "processes"]
ResourceLimitEventKind: TypeAlias = Literal["throttled", "high", "maximum", "out_of_memory", "terminated"]
AcceleratorVendor: TypeAlias = Literal["nvidia", "amd", "intel", "apple", "other"]
UnitType = TypeVar(
    "UnitType",
    bound=DurationUnit | CapacityUnit | DataUnit | BandwidthUnit | CpuUnit | UtilizationUnit | CountUnit,
)
MeasurementValue = TypeVar("MeasurementValue", bound=int | float)


@dataclass(frozen=True)
class ResourceMeasurement(Generic[MeasurementValue, UnitType]):
    """One available value and its selected unit."""

    value: MeasurementValue
    unit: UnitType


@dataclass(frozen=True)
class ResourceLimitEventCount:
    """One typed cumulative resource-limit event counter."""

    resource: ResourceLimitResource
    event: ResourceLimitEventKind
    count: ResourceMeasurement[int | float, CountUnit] | None


@dataclass(frozen=True)
class AcceleratorDeviceMetrics:
    """Device-wide accelerator measurements."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    memory_used: ResourceMeasurement[int | float, CapacityUnit] | None
    compute_utilization: ResourceMeasurement[float, UtilizationUnit] | None


@dataclass(frozen=True)
class AcceleratorProcessMetrics:
    """Accelerator measurements attributed to one process in the selected scope."""

    vendor: AcceleratorVendor
    device_identifier: str
    device_index: int | None
    process_id: int
    memory_used: ResourceMeasurement[int | float, CapacityUnit] | None
    compute_utilization: ResourceMeasurement[float, UtilizationUnit] | None


@dataclass(frozen=True)
class FilesystemCapacityMetrics:
    """Capacity for one configured filesystem path."""

    path: str
    total_capacity: ResourceMeasurement[int | float, CapacityUnit] | None
    available_capacity: ResourceMeasurement[int | float, CapacityUnit] | None
    free_capacity: ResourceMeasurement[int | float, CapacityUnit] | None


@dataclass(frozen=True)
class CpuMetrics:
    """CPU measurements and CPU limit events."""

    user_time: ResourceMeasurement[int | float, DurationUnit] | None
    system_time: ResourceMeasurement[int | float, DurationUnit] | None
    total_time: ResourceMeasurement[int | float, DurationUnit] | None
    consumption_rate: ResourceMeasurement[float, CpuUnit] | None
    throttled_time: ResourceMeasurement[int | float, DurationUnit] | None
    effective_limit: ResourceMeasurement[float, CpuUnit] | None
    some_pressure_stall_time: ResourceMeasurement[int | float, DurationUnit] | None
    full_pressure_stall_time: ResourceMeasurement[int | float, DurationUnit] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class MemoryMetrics:
    """System and process memory measurements and memory limit events."""

    system_used: ResourceMeasurement[int | float, CapacityUnit] | None
    system_total: ResourceMeasurement[int | float, CapacityUnit] | None
    system_available: ResourceMeasurement[int | float, CapacityUnit] | None
    resident: ResourceMeasurement[int | float, CapacityUnit] | None
    private: ResourceMeasurement[int | float, CapacityUnit] | None
    physical_footprint: ResourceMeasurement[int | float, CapacityUnit] | None
    virtual_memory: ResourceMeasurement[int | float, CapacityUnit] | None
    peak_resident: ResourceMeasurement[int | float, CapacityUnit] | None
    limit: ResourceMeasurement[int | float, CapacityUnit] | None
    environment_accounted: ResourceMeasurement[int | float, CapacityUnit] | None
    some_pressure_stall_time: ResourceMeasurement[int | float, DurationUnit] | None
    full_pressure_stall_time: ResourceMeasurement[int | float, DurationUnit] | None
    out_of_memory_event_count: ResourceMeasurement[int | float, CountUnit] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class ProcessMetrics:
    """Process counts, limits, handles, and descriptors."""

    active_count: ResourceMeasurement[int | float, CountUnit] | None
    descendant_count: ResourceMeasurement[int | float, CountUnit] | None
    thread_count: ResourceMeasurement[int | float, CountUnit] | None
    lifetime_creation_count: ResourceMeasurement[int | float, CountUnit] | None
    open_file_descriptor_count: ResourceMeasurement[int | float, CountUnit] | None
    windows_handle_count: ResourceMeasurement[int | float, CountUnit] | None
    limit_events: list[ResourceLimitEventCount]


@dataclass(frozen=True)
class DiskMetrics:
    """Disk I/O for the measured process scope and filesystem capacity."""

    read_data: ResourceMeasurement[int | float, DataUnit] | None
    write_data: ResourceMeasurement[int | float, DataUnit] | None
    read_throughput: ResourceMeasurement[float, BandwidthUnit] | None
    write_throughput: ResourceMeasurement[float, BandwidthUnit] | None
    read_operations: ResourceMeasurement[int | float, CountUnit] | None
    write_operations: ResourceMeasurement[int | float, CountUnit] | None
    filesystems: list[FilesystemCapacityMetrics]


@dataclass(frozen=True)
class GpuMetrics:
    """Accelerator device-wide and process-attributed measurements."""

    device_metrics: list[AcceleratorDeviceMetrics] | None
    process_metrics: list[AcceleratorProcessMetrics] | None


@dataclass(frozen=True)
class NetworkTrafficMetrics:
    """System interface traffic counters and transfer rates."""

    received_data: ResourceMeasurement[int | float, DataUnit] | None
    transmitted_data: ResourceMeasurement[int | float, DataUnit] | None
    receive_throughput: ResourceMeasurement[float, BandwidthUnit] | None
    transmit_throughput: ResourceMeasurement[float, BandwidthUnit] | None
    received_packets: ResourceMeasurement[int | float, CountUnit] | None
    transmitted_packets: ResourceMeasurement[int | float, CountUnit] | None
    receive_errors: ResourceMeasurement[int | float, CountUnit] | None
    transmit_errors: ResourceMeasurement[int | float, CountUnit] | None


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


def _measurement(value: object, unit_type: type[UnitType]) -> ResourceMeasurement[int | float, UnitType] | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return ResourceMeasurement(
        value=cast(int | float, data["value"]),
        unit=unit_type(cast(str, data["unit"])),
    )


def _limit_event(value: object) -> ResourceLimitEventCount:
    data = cast(dict[str, object], value)
    return ResourceLimitEventCount(
        resource=cast(ResourceLimitResource, data["resource"]),
        event=cast(ResourceLimitEventKind, data["event"]),
        count=_measurement(data["count"], CountUnit),
    )


def _accelerator_device(value: object) -> AcceleratorDeviceMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorDeviceMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        memory_used=_measurement(data["memory_used"], CapacityUnit),
        compute_utilization=_measurement(data["compute_utilization"], UtilizationUnit),
    )


def _accelerator_process(value: object) -> AcceleratorProcessMetrics:
    data = cast(dict[str, object], value)
    return AcceleratorProcessMetrics(
        vendor=cast(AcceleratorVendor, data["vendor"]),
        device_identifier=cast(str, data["device_identifier"]),
        device_index=cast(int | None, data.get("device_index")),
        process_id=cast(int, data["process_id"]),
        memory_used=_measurement(data["memory_used"], CapacityUnit),
        compute_utilization=_measurement(data["compute_utilization"], UtilizationUnit),
    )


def _filesystem(value: object) -> FilesystemCapacityMetrics:
    data = cast(dict[str, object], value)
    return FilesystemCapacityMetrics(
        path=cast(str, data["path"]),
        total_capacity=_measurement(data["total_capacity"], CapacityUnit),
        available_capacity=_measurement(data["available_capacity"], CapacityUnit),
        free_capacity=_measurement(data["free_capacity"], CapacityUnit),
    )


def _events(value: object) -> list[ResourceLimitEventCount]:
    return [_limit_event(item) for item in cast(list[object], value)]


def _cpu(value: object) -> CpuMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return CpuMetrics(
        user_time=_measurement(data["user_time"], DurationUnit),
        system_time=_measurement(data["system_time"], DurationUnit),
        total_time=_measurement(data["total_time"], DurationUnit),
        consumption_rate=_measurement(data["consumption_rate"], CpuUnit),
        throttled_time=_measurement(data["throttled_time"], DurationUnit),
        effective_limit=_measurement(data["effective_limit"], CpuUnit),
        some_pressure_stall_time=_measurement(data["some_pressure_stall_time"], DurationUnit),
        full_pressure_stall_time=_measurement(data["full_pressure_stall_time"], DurationUnit),
        limit_events=_events(data["limit_events"]),
    )


def _memory(value: object) -> MemoryMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return MemoryMetrics(
        system_used=_measurement(data["system_used"], CapacityUnit),
        system_total=_measurement(data["system_total"], CapacityUnit),
        system_available=_measurement(data["system_available"], CapacityUnit),
        resident=_measurement(data["resident"], CapacityUnit),
        private=_measurement(data["private"], CapacityUnit),
        physical_footprint=_measurement(data["physical_footprint"], CapacityUnit),
        virtual_memory=_measurement(data["virtual_memory"], CapacityUnit),
        peak_resident=_measurement(data["peak_resident"], CapacityUnit),
        limit=_measurement(data["limit"], CapacityUnit),
        environment_accounted=_measurement(data["environment_accounted"], CapacityUnit),
        some_pressure_stall_time=_measurement(data["some_pressure_stall_time"], DurationUnit),
        full_pressure_stall_time=_measurement(data["full_pressure_stall_time"], DurationUnit),
        out_of_memory_event_count=_measurement(data["out_of_memory_event_count"], CountUnit),
        limit_events=_events(data["limit_events"]),
    )


def _process(value: object) -> ProcessMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return ProcessMetrics(
        active_count=_measurement(data["active_count"], CountUnit),
        descendant_count=_measurement(data["descendant_count"], CountUnit),
        thread_count=_measurement(data["thread_count"], CountUnit),
        lifetime_creation_count=_measurement(data["lifetime_creation_count"], CountUnit),
        open_file_descriptor_count=_measurement(data["open_file_descriptor_count"], CountUnit),
        windows_handle_count=_measurement(data["windows_handle_count"], CountUnit),
        limit_events=_events(data["limit_events"]),
    )


def _disk(value: object) -> DiskMetrics | None:
    if value is None:
        return None
    data = cast(dict[str, object], value)
    return DiskMetrics(
        read_data=_measurement(data["read_data"], DataUnit),
        write_data=_measurement(data["write_data"], DataUnit),
        read_throughput=_measurement(data["read_throughput"], BandwidthUnit),
        write_throughput=_measurement(data["write_throughput"], BandwidthUnit),
        read_operations=_measurement(data["read_operations"], CountUnit),
        write_operations=_measurement(data["write_operations"], CountUnit),
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
        received_data=_measurement(data["received_data"], DataUnit),
        transmitted_data=_measurement(data["transmitted_data"], DataUnit),
        receive_throughput=_measurement(data["receive_throughput"], BandwidthUnit),
        transmit_throughput=_measurement(data["transmit_throughput"], BandwidthUnit),
        received_packets=_measurement(data["received_packets"], CountUnit),
        transmitted_packets=_measurement(data["transmitted_packets"], CountUnit),
        receive_errors=_measurement(data["receive_errors"], CountUnit),
        transmit_errors=_measurement(data["transmit_errors"], CountUnit),
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
