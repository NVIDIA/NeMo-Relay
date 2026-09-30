// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdbool.h>
typedef struct FfiResourceMetricsCollection FfiResourceMetricsCollection;
extern int32_t nemo_relay_resource_metrics_collect_start(FfiResourceMetricsCollection** out_collection);
extern int32_t nemo_relay_resource_metrics_collect_poll(FfiResourceMetricsCollection* collection, bool* out_done, char** out_json);
extern void nemo_relay_resource_metrics_collect_free(FfiResourceMetricsCollection* collection);
extern void nemo_relay_string_free(char* ptr);
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"strconv"
	"time"
)

// ResourceMeasurementUnit is the selected unit attached to an available measurement.
type ResourceMeasurementUnit string

const (
	ResourceUnitMilliseconds       ResourceMeasurementUnit = "milliseconds"
	ResourceUnitBytes              ResourceMeasurementUnit = "bytes"
	ResourceUnitKibibytes          ResourceMeasurementUnit = "kibibytes"
	ResourceUnitLogicalProcessors  ResourceMeasurementUnit = "logical_processors"
	ResourceUnitPercentage         ResourceMeasurementUnit = "percentage"
	ResourceUnitProcesses          ResourceMeasurementUnit = "processes"
	ResourceUnitThreads            ResourceMeasurementUnit = "threads"
	ResourceUnitFileDescriptors    ResourceMeasurementUnit = "file_descriptors"
	ResourceUnitHandles            ResourceMeasurementUnit = "handles"
	ResourceUnitEvents             ResourceMeasurementUnit = "events"
	ResourceUnitOperations         ResourceMeasurementUnit = "operations"
	ResourceUnitMicroseconds       ResourceMeasurementUnit = "microseconds"
	ResourceUnitSeconds            ResourceMeasurementUnit = "seconds"
	ResourceUnitMinutes            ResourceMeasurementUnit = "minutes"
	ResourceUnitKilobytes          ResourceMeasurementUnit = "kilobytes"
	ResourceUnitMegabytes          ResourceMeasurementUnit = "megabytes"
	ResourceUnitGigabytes          ResourceMeasurementUnit = "gigabytes"
	ResourceUnitTerabytes          ResourceMeasurementUnit = "terabytes"
	ResourceUnitMebibytes          ResourceMeasurementUnit = "mebibytes"
	ResourceUnitGibibytes          ResourceMeasurementUnit = "gibibytes"
	ResourceUnitTebibytes          ResourceMeasurementUnit = "tebibytes"
	ResourceUnitBytesPerSecond     ResourceMeasurementUnit = "bytes_per_second"
	ResourceUnitKibibytesPerSecond ResourceMeasurementUnit = "kibibytes_per_second"
	ResourceUnitMebibytesPerSecond ResourceMeasurementUnit = "mebibytes_per_second"
	ResourceUnitGibibytesPerSecond ResourceMeasurementUnit = "gibibytes_per_second"
	ResourceUnitBitsPerSecond      ResourceMeasurementUnit = "bits_per_second"
	ResourceUnitMegabitsPerSecond  ResourceMeasurementUnit = "megabits_per_second"
	ResourceUnitGigabitsPerSecond  ResourceMeasurementUnit = "gigabits_per_second"
	ResourceUnitMillicores         ResourceMeasurementUnit = "millicores"
	ResourceUnitFraction           ResourceMeasurementUnit = "fraction"
	ResourceUnitPackets            ResourceMeasurementUnit = "packets"
	ResourceUnitErrors             ResourceMeasurementUnit = "errors"
)

type ResourceOperatingSystem string

const (
	ResourceOperatingSystemLinux       ResourceOperatingSystem = "linux"
	ResourceOperatingSystemMacOS       ResourceOperatingSystem = "macos"
	ResourceOperatingSystemWindows     ResourceOperatingSystem = "windows"
	ResourceOperatingSystemUnsupported ResourceOperatingSystem = "unsupported"
)

type ResourceMeasurementScope string

const (
	ResourceMeasurementScopeGlobal             ResourceMeasurementScope = "global"
	ResourceMeasurementScopeApplicationProcess ResourceMeasurementScope = "application_process"
	ResourceMeasurementScopeProcessTree        ResourceMeasurementScope = "process_tree"
)

type ResourceLimitResource string

const (
	ResourceLimitCPU       ResourceLimitResource = "cpu"
	ResourceLimitMemory    ResourceLimitResource = "memory"
	ResourceLimitProcesses ResourceLimitResource = "processes"
)

type ResourceLimitEventKind string

const (
	ResourceLimitThrottled   ResourceLimitEventKind = "throttled"
	ResourceLimitHigh        ResourceLimitEventKind = "high"
	ResourceLimitMaximum     ResourceLimitEventKind = "maximum"
	ResourceLimitOutOfMemory ResourceLimitEventKind = "out_of_memory"
	ResourceLimitTerminated  ResourceLimitEventKind = "terminated"
)

type AcceleratorVendor string

const (
	AcceleratorVendorNVIDIA AcceleratorVendor = "nvidia"
	AcceleratorVendorAMD    AcceleratorVendor = "amd"
	AcceleratorVendorIntel  AcceleratorVendor = "intel"
	AcceleratorVendorApple  AcceleratorVendor = "apple"
	AcceleratorVendorOther  AcceleratorVendor = "other"
)

// ResourceMetricValue holds either an exact integer or a fractional measurement.
type ResourceMetricValue struct {
	Integer *uint64
	Decimal *float64
}

func (v *ResourceMetricValue) UnmarshalJSON(data []byte) error {
	if integer, err := strconv.ParseUint(string(data), 10, 64); err == nil {
		v.Integer = &integer
		v.Decimal = nil
		return nil
	}
	decimal, err := strconv.ParseFloat(string(data), 64)
	if err != nil || math.IsNaN(decimal) || math.IsInf(decimal, 0) {
		return fmt.Errorf("invalid resource metric number: %s", data)
	}
	v.Integer = nil
	v.Decimal = &decimal
	return nil
}

func (v ResourceMetricValue) MarshalJSON() ([]byte, error) {
	if v.Integer != nil {
		return json.Marshal(*v.Integer)
	}
	if v.Decimal != nil {
		return json.Marshal(*v.Decimal)
	}
	return nil, fmt.Errorf("resource metric value has no number")
}

// ResourceMeasurement contains only the available measurement's value and unit.
type ResourceMeasurement[T any] struct {
	Value T                       `json:"value"`
	Unit  ResourceMeasurementUnit `json:"unit"`
}

type ResourceLimitEventCount struct {
	Resource ResourceLimitResource                     `json:"resource"`
	Event    ResourceLimitEventKind                    `json:"event"`
	Count    *ResourceMeasurement[ResourceMetricValue] `json:"count"`
}

type AcceleratorDeviceMetrics struct {
	Vendor             AcceleratorVendor                         `json:"vendor"`
	DeviceIdentifier   string                                    `json:"device_identifier"`
	DeviceIndex        *uint32                                   `json:"device_index"`
	MemoryUsed         *ResourceMeasurement[ResourceMetricValue] `json:"memory_used"`
	ComputeUtilization *ResourceMeasurement[float64]             `json:"compute_utilization"`
}

type AcceleratorProcessMetrics struct {
	Vendor             AcceleratorVendor                         `json:"vendor"`
	DeviceIdentifier   string                                    `json:"device_identifier"`
	DeviceIndex        *uint32                                   `json:"device_index"`
	ProcessID          uint32                                    `json:"process_id"`
	MemoryUsed         *ResourceMeasurement[ResourceMetricValue] `json:"memory_used"`
	ComputeUtilization *ResourceMeasurement[float64]             `json:"compute_utilization"`
}

type FilesystemCapacityMetrics struct {
	Path              string                                    `json:"path"`
	TotalCapacity     *ResourceMeasurement[ResourceMetricValue] `json:"total_capacity"`
	AvailableCapacity *ResourceMeasurement[ResourceMetricValue] `json:"available_capacity"`
	FreeCapacity      *ResourceMeasurement[ResourceMetricValue] `json:"free_capacity"`
}

type CPUMetrics struct {
	UserTime              *ResourceMeasurement[ResourceMetricValue] `json:"user_time"`
	SystemTime            *ResourceMeasurement[ResourceMetricValue] `json:"system_time"`
	TotalTime             *ResourceMeasurement[ResourceMetricValue] `json:"total_time"`
	ConsumptionRate       *ResourceMeasurement[float64]             `json:"consumption_rate"`
	ThrottledTime         *ResourceMeasurement[ResourceMetricValue] `json:"throttled_time"`
	EffectiveLimit        *ResourceMeasurement[float64]             `json:"effective_limit"`
	SomePressureStallTime *ResourceMeasurement[ResourceMetricValue] `json:"some_pressure_stall_time"`
	FullPressureStallTime *ResourceMeasurement[ResourceMetricValue] `json:"full_pressure_stall_time"`
	LimitEvents           []ResourceLimitEventCount                 `json:"limit_events"`
}

type MemoryMetrics struct {
	SystemUsed            *ResourceMeasurement[ResourceMetricValue] `json:"system_used"`
	SystemTotal           *ResourceMeasurement[ResourceMetricValue] `json:"system_total"`
	SystemAvailable       *ResourceMeasurement[ResourceMetricValue] `json:"system_available"`
	Resident              *ResourceMeasurement[ResourceMetricValue] `json:"resident"`
	Private               *ResourceMeasurement[ResourceMetricValue] `json:"private"`
	PhysicalFootprint     *ResourceMeasurement[ResourceMetricValue] `json:"physical_footprint"`
	VirtualMemory         *ResourceMeasurement[ResourceMetricValue] `json:"virtual_memory"`
	PeakResident          *ResourceMeasurement[ResourceMetricValue] `json:"peak_resident"`
	Limit                 *ResourceMeasurement[ResourceMetricValue] `json:"limit"`
	EnvironmentAccounted  *ResourceMeasurement[ResourceMetricValue] `json:"environment_accounted"`
	SomePressureStallTime *ResourceMeasurement[ResourceMetricValue] `json:"some_pressure_stall_time"`
	FullPressureStallTime *ResourceMeasurement[ResourceMetricValue] `json:"full_pressure_stall_time"`
	OutOfMemoryEventCount *ResourceMeasurement[ResourceMetricValue] `json:"out_of_memory_event_count"`
	LimitEvents           []ResourceLimitEventCount                 `json:"limit_events"`
}

type ProcessMetrics struct {
	ActiveCount             *ResourceMeasurement[ResourceMetricValue] `json:"active_count"`
	DescendantCount         *ResourceMeasurement[ResourceMetricValue] `json:"descendant_count"`
	ThreadCount             *ResourceMeasurement[ResourceMetricValue] `json:"thread_count"`
	LifetimeCreationCount   *ResourceMeasurement[ResourceMetricValue] `json:"lifetime_creation_count"`
	OpenFileDescriptorCount *ResourceMeasurement[ResourceMetricValue] `json:"open_file_descriptor_count"`
	WindowsHandleCount      *ResourceMeasurement[ResourceMetricValue] `json:"windows_handle_count"`
	LimitEvents             []ResourceLimitEventCount                 `json:"limit_events"`
}

type DiskMetrics struct {
	ReadData        *ResourceMeasurement[ResourceMetricValue] `json:"read_data"`
	WriteData       *ResourceMeasurement[ResourceMetricValue] `json:"write_data"`
	ReadThroughput  *ResourceMeasurement[float64]             `json:"read_throughput"`
	WriteThroughput *ResourceMeasurement[float64]             `json:"write_throughput"`
	ReadOperations  *ResourceMeasurement[ResourceMetricValue] `json:"read_operations"`
	WriteOperations *ResourceMeasurement[ResourceMetricValue] `json:"write_operations"`
	Filesystems     []FilesystemCapacityMetrics               `json:"filesystems"`
}

type NetworkTrafficMetrics struct {
	ReceivedData       *ResourceMeasurement[ResourceMetricValue] `json:"received_data"`
	TransmittedData    *ResourceMeasurement[ResourceMetricValue] `json:"transmitted_data"`
	ReceiveThroughput  *ResourceMeasurement[float64]             `json:"receive_throughput"`
	TransmitThroughput *ResourceMeasurement[float64]             `json:"transmit_throughput"`
	ReceivedPackets    *ResourceMeasurement[ResourceMetricValue] `json:"received_packets"`
	TransmittedPackets *ResourceMeasurement[ResourceMetricValue] `json:"transmitted_packets"`
	ReceiveErrors      *ResourceMeasurement[ResourceMetricValue] `json:"receive_errors"`
	TransmitErrors     *ResourceMeasurement[ResourceMetricValue] `json:"transmit_errors"`
}

type NetworkInterfaceMetrics struct {
	Name    string                `json:"name"`
	Traffic NetworkTrafficMetrics `json:"traffic"`
}

type NetworkMetrics struct {
	MeasurementScope ResourceMeasurementScope  `json:"measurement_scope"`
	System           NetworkTrafficMetrics     `json:"system"`
	Interfaces       []NetworkInterfaceMetrics `json:"interfaces"`
}

type GPUMetrics struct {
	DeviceMetrics  []AcceleratorDeviceMetrics  `json:"device_metrics"`
	ProcessMetrics []AcceleratorProcessMetrics `json:"process_metrics"`
}

// ResourceMetricsSnapshot is the canonical point-in-time snapshot returned by the Rust core.
type ResourceMetricsSnapshot struct {
	Timestamp        time.Time                `json:"timestamp"`
	OperatingSystem  ResourceOperatingSystem  `json:"operating_system"`
	MeasurementScope ResourceMeasurementScope `json:"measurement_scope"`
	ProcessSampling  *ProcessSamplingMetadata `json:"process_sampling"`
	CPU              *CPUMetrics              `json:"cpu"`
	Memory           *MemoryMetrics           `json:"memory"`
	Process          *ProcessMetrics          `json:"process"`
	Disk             *DiskMetrics             `json:"disk"`
	GPU              *GPUMetrics              `json:"gpu"`
	Network          *NetworkMetrics          `json:"network"`
}

// ProcessSamplingMetadata reports process-query coverage for aggregate measurements.
type ProcessSamplingMetadata struct {
	VisibleProcesses      uint64            `json:"visible_processes"`
	SampledProcesses      uint64            `json:"sampled_processes"`
	FieldSampledProcesses map[string]uint64 `json:"field_sampled_processes"`
}

// CollectResourceMetrics awaits one sample using the active plugin's categories
// and target. Context cancellation stops waiting; an OS query already running
// on a worker may finish afterward. Polling does not need to be enabled.
func CollectResourceMetrics(ctx context.Context) (ResourceMetricsSnapshot, error) {
	if ctx == nil {
		return ResourceMetricsSnapshot{}, fmt.Errorf("resource metrics context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return ResourceMetricsSnapshot{}, err
	}
	var collection *C.FfiResourceMetricsCollection
	err := func() error {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		return checkStatus(C.nemo_relay_resource_metrics_collect_start(&collection))
	}()
	if err != nil {
		return ResourceMetricsSnapshot{}, err
	}
	if collection == nil {
		return ResourceMetricsSnapshot{}, fmt.Errorf("resource metrics FFI returned a null collection")
	}
	defer C.nemo_relay_resource_metrics_collect_free(collection)
	return waitForResourceMetrics(ctx, func() (bool, []byte, error) {
		var done C.bool
		var out *C.char
		err := func() error {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			return checkStatus(C.nemo_relay_resource_metrics_collect_poll(collection, &done, &out))
		}()
		if err != nil || !bool(done) {
			return false, nil, err
		}
		if out == nil {
			return false, nil, fmt.Errorf("resource metrics FFI returned a null snapshot")
		}
		defer C.nemo_relay_string_free(out)
		return true, []byte(C.GoString(out)), nil
	})
}

func waitForResourceMetrics(ctx context.Context, poll func() (bool, []byte, error)) (ResourceMetricsSnapshot, error) {
	delay := time.Millisecond
	const maxDelay = 50 * time.Millisecond
	timer := time.NewTimer(delay)
	defer timer.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return ResourceMetricsSnapshot{}, err
		}
		done, encoded, err := poll()
		if err != nil {
			return ResourceMetricsSnapshot{}, err
		}
		if done {
			var snapshot ResourceMetricsSnapshot
			if err := json.Unmarshal(encoded, &snapshot); err != nil {
				return ResourceMetricsSnapshot{}, fmt.Errorf("decode resource metrics snapshot: %w", err)
			}
			return snapshot, nil
		}
		select {
		case <-ctx.Done():
			return ResourceMetricsSnapshot{}, ctx.Err()
		case <-timer.C:
		}
		delay = min(delay*2, maxDelay)
		timer.Reset(delay)
	}
}
