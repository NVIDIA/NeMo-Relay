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

// DurationUnit identifies units for duration measurements.
type DurationUnit string

const (
	DurationMicroseconds DurationUnit = "microseconds"
	DurationMilliseconds DurationUnit = "milliseconds"
	DurationSeconds      DurationUnit = "seconds"
	DurationMinutes      DurationUnit = "minutes"
)

func (unit *DurationUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch DurationUnit(name) {
	case DurationMicroseconds, DurationMilliseconds, DurationSeconds, DurationMinutes:
		*unit = DurationUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid duration unit: %q", name)
	}
}

// CapacityUnit identifies units for capacity measurements.
type CapacityUnit string

const (
	CapacityBytes     CapacityUnit = "bytes"
	CapacityKilobytes CapacityUnit = "kilobytes"
	CapacityMegabytes CapacityUnit = "megabytes"
	CapacityGigabytes CapacityUnit = "gigabytes"
	CapacityTerabytes CapacityUnit = "terabytes"
	CapacityKibibytes CapacityUnit = "kibibytes"
	CapacityMebibytes CapacityUnit = "mebibytes"
	CapacityGibibytes CapacityUnit = "gibibytes"
	CapacityTebibytes CapacityUnit = "tebibytes"
)

func (unit *CapacityUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch CapacityUnit(name) {
	case CapacityBytes, CapacityKilobytes, CapacityMegabytes, CapacityGigabytes, CapacityTerabytes, CapacityKibibytes, CapacityMebibytes, CapacityGibibytes, CapacityTebibytes:
		*unit = CapacityUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid capacity unit: %q", name)
	}
}

// DataUnit identifies units for data measurements.
type DataUnit string

const (
	DataBytes     DataUnit = "bytes"
	DataKilobytes DataUnit = "kilobytes"
	DataMegabytes DataUnit = "megabytes"
	DataGigabytes DataUnit = "gigabytes"
	DataTerabytes DataUnit = "terabytes"
	DataKibibytes DataUnit = "kibibytes"
	DataMebibytes DataUnit = "mebibytes"
	DataGibibytes DataUnit = "gibibytes"
	DataTebibytes DataUnit = "tebibytes"
)

func (unit *DataUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch DataUnit(name) {
	case DataBytes, DataKilobytes, DataMegabytes, DataGigabytes, DataTerabytes, DataKibibytes, DataMebibytes, DataGibibytes, DataTebibytes:
		*unit = DataUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid data unit: %q", name)
	}
}

// BandwidthUnit identifies units for bandwidth measurements.
type BandwidthUnit string

const (
	BandwidthBytesPerSecond     BandwidthUnit = "bytes_per_second"
	BandwidthKibibytesPerSecond BandwidthUnit = "kibibytes_per_second"
	BandwidthMebibytesPerSecond BandwidthUnit = "mebibytes_per_second"
	BandwidthGibibytesPerSecond BandwidthUnit = "gibibytes_per_second"
	BandwidthBitsPerSecond      BandwidthUnit = "bits_per_second"
	BandwidthMegabitsPerSecond  BandwidthUnit = "megabits_per_second"
	BandwidthGigabitsPerSecond  BandwidthUnit = "gigabits_per_second"
)

func (unit *BandwidthUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch BandwidthUnit(name) {
	case BandwidthBytesPerSecond, BandwidthKibibytesPerSecond, BandwidthMebibytesPerSecond, BandwidthGibibytesPerSecond, BandwidthBitsPerSecond, BandwidthMegabitsPerSecond, BandwidthGigabitsPerSecond:
		*unit = BandwidthUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid bandwidth unit: %q", name)
	}
}

// CpuUnit identifies units for CPU measurements.
type CpuUnit string

const (
	CpuLogicalProcessors CpuUnit = "logical_processors"
	CpuMillicores        CpuUnit = "millicores"
)

func (unit *CpuUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch CpuUnit(name) {
	case CpuLogicalProcessors, CpuMillicores:
		*unit = CpuUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid cpu unit: %q", name)
	}
}

// UtilizationUnit identifies units for utilization measurements.
type UtilizationUnit string

const (
	UtilizationPercentage UtilizationUnit = "percentage"
	UtilizationFraction   UtilizationUnit = "fraction"
)

func (unit *UtilizationUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch UtilizationUnit(name) {
	case UtilizationPercentage, UtilizationFraction:
		*unit = UtilizationUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid utilization unit: %q", name)
	}
}

// CountUnit identifies units for count measurements.
type CountUnit string

const (
	CountProcesses       CountUnit = "processes"
	CountThreads         CountUnit = "threads"
	CountFileDescriptors CountUnit = "file_descriptors"
	CountHandles         CountUnit = "handles"
	CountEvents          CountUnit = "events"
	CountOperations      CountUnit = "operations"
	CountPackets         CountUnit = "packets"
	CountErrors          CountUnit = "errors"
)

func (unit *CountUnit) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	switch CountUnit(name) {
	case CountProcesses, CountThreads, CountFileDescriptors, CountHandles, CountEvents, CountOperations, CountPackets, CountErrors:
		*unit = CountUnit(name)
		return nil
	default:
		return fmt.Errorf("invalid count unit: %q", name)
	}
}

// ResourceUnit is one semantic measurement unit category.
type ResourceUnit interface {
	DurationUnit | CapacityUnit | DataUnit | BandwidthUnit | CpuUnit | UtilizationUnit | CountUnit
}

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
type ResourceMeasurement[T any, U ResourceUnit] struct {
	Value T `json:"value"`
	Unit  U `json:"unit"`
}

type ResourceLimitEventCount struct {
	Resource ResourceLimitResource                                `json:"resource"`
	Event    ResourceLimitEventKind                               `json:"event"`
	Count    *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"count"`
}

type AcceleratorDeviceMetrics struct {
	Vendor             AcceleratorVendor                                       `json:"vendor"`
	DeviceIdentifier   string                                                  `json:"device_identifier"`
	DeviceIndex        *uint32                                                 `json:"device_index"`
	MemoryUsed         *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"memory_used"`
	ComputeUtilization *ResourceMeasurement[float64, UtilizationUnit]          `json:"compute_utilization"`
}

type AcceleratorProcessMetrics struct {
	Vendor             AcceleratorVendor                                       `json:"vendor"`
	DeviceIdentifier   string                                                  `json:"device_identifier"`
	DeviceIndex        *uint32                                                 `json:"device_index"`
	ProcessID          uint32                                                  `json:"process_id"`
	MemoryUsed         *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"memory_used"`
	ComputeUtilization *ResourceMeasurement[float64, UtilizationUnit]          `json:"compute_utilization"`
}

type FilesystemCapacityMetrics struct {
	Path              string                                                  `json:"path"`
	TotalCapacity     *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"total_capacity"`
	AvailableCapacity *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"available_capacity"`
	FreeCapacity      *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"free_capacity"`
}

type CPUMetrics struct {
	UserTime              *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"user_time"`
	SystemTime            *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"system_time"`
	TotalTime             *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"total_time"`
	ConsumptionRate       *ResourceMeasurement[float64, CpuUnit]                  `json:"consumption_rate"`
	ThrottledTime         *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"throttled_time"`
	EffectiveLimit        *ResourceMeasurement[float64, CpuUnit]                  `json:"effective_limit"`
	SomePressureStallTime *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"some_pressure_stall_time"`
	FullPressureStallTime *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"full_pressure_stall_time"`
	LimitEvents           []ResourceLimitEventCount                               `json:"limit_events"`
}

type MemoryMetrics struct {
	SystemUsed            *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"system_used"`
	SystemTotal           *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"system_total"`
	SystemAvailable       *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"system_available"`
	Resident              *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"resident"`
	Private               *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"private"`
	PhysicalFootprint     *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"physical_footprint"`
	VirtualMemory         *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"virtual_memory"`
	PeakResident          *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"peak_resident"`
	Limit                 *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"limit"`
	EnvironmentAccounted  *ResourceMeasurement[ResourceMetricValue, CapacityUnit] `json:"environment_accounted"`
	SomePressureStallTime *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"some_pressure_stall_time"`
	FullPressureStallTime *ResourceMeasurement[ResourceMetricValue, DurationUnit] `json:"full_pressure_stall_time"`
	OutOfMemoryEventCount *ResourceMeasurement[ResourceMetricValue, CountUnit]    `json:"out_of_memory_event_count"`
	LimitEvents           []ResourceLimitEventCount                               `json:"limit_events"`
}

type ProcessMetrics struct {
	ActiveCount             *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"active_count"`
	DescendantCount         *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"descendant_count"`
	ThreadCount             *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"thread_count"`
	LifetimeCreationCount   *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"lifetime_creation_count"`
	OpenFileDescriptorCount *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"open_file_descriptor_count"`
	WindowsHandleCount      *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"windows_handle_count"`
	LimitEvents             []ResourceLimitEventCount                            `json:"limit_events"`
}

type DiskMetrics struct {
	ReadData        *ResourceMeasurement[ResourceMetricValue, DataUnit]  `json:"read_data"`
	WriteData       *ResourceMeasurement[ResourceMetricValue, DataUnit]  `json:"write_data"`
	ReadThroughput  *ResourceMeasurement[float64, BandwidthUnit]         `json:"read_throughput"`
	WriteThroughput *ResourceMeasurement[float64, BandwidthUnit]         `json:"write_throughput"`
	ReadOperations  *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"read_operations"`
	WriteOperations *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"write_operations"`
	Filesystems     []FilesystemCapacityMetrics                          `json:"filesystems"`
}

type NetworkTrafficMetrics struct {
	ReceivedData       *ResourceMeasurement[ResourceMetricValue, DataUnit]  `json:"received_data"`
	TransmittedData    *ResourceMeasurement[ResourceMetricValue, DataUnit]  `json:"transmitted_data"`
	ReceiveThroughput  *ResourceMeasurement[float64, BandwidthUnit]         `json:"receive_throughput"`
	TransmitThroughput *ResourceMeasurement[float64, BandwidthUnit]         `json:"transmit_throughput"`
	ReceivedPackets    *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"received_packets"`
	TransmittedPackets *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"transmitted_packets"`
	ReceiveErrors      *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"receive_errors"`
	TransmitErrors     *ResourceMeasurement[ResourceMetricValue, CountUnit] `json:"transmit_errors"`
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
	return collectResourceMetrics(ctx, startResourceMetricsCollection)
}

type resourceMetricsCollection = C.FfiResourceMetricsCollection

func startResourceMetricsCollection() (*resourceMetricsCollection, error) {
	var collection *resourceMetricsCollection
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	err := checkStatus(C.nemo_relay_resource_metrics_collect_start(&collection))
	return collection, err
}

func collectResourceMetrics(ctx context.Context, start func() (*resourceMetricsCollection, error)) (ResourceMetricsSnapshot, error) {
	if ctx == nil {
		return ResourceMetricsSnapshot{}, fmt.Errorf("resource metrics context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return ResourceMetricsSnapshot{}, err
	}
	collection, err := start()
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
		defer C.nemo_relay_string_free(out)
		encoded, err := resourceMetricsJSON(out)
		return true, encoded, err
	})
}

func resourceMetricsJSON(out *C.char) ([]byte, error) {
	if out == nil {
		return nil, fmt.Errorf("resource metrics FFI returned a null snapshot")
	}
	return []byte(C.GoString(out)), nil
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
