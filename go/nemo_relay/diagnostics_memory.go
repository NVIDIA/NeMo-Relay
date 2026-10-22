// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// ResourceMemoryNumber preserves integer strings or finite decimal values by kind.
// Value is absent for non_finite. Decode integers as DiagnosticNumber, decimals as float64.
type ResourceMemoryNumber struct {
	Kind  string          `json:"kind"`
	Value json.RawMessage `json:"value,omitempty"`
}

// ResourceMemoryUnit preserves the original configured dimension and unit.
type ResourceMemoryUnit struct {
	Dimension string `json:"dimension"`
	Name      string `json:"name"`
}

// DiagnosticMemoryUsage reports reserved diagnostic backing and overlapping occupancy gauges.
type DiagnosticMemoryUsage struct {
	ReservedBytes        DiagnosticNumber `json:"reserved_bytes"`
	Components           DiagnosticNumber `json:"components"`
	Counters             DiagnosticNumber `json:"counters"`
	Events               DiagnosticNumber `json:"events"`
	EventSlotBytes       DiagnosticNumber `json:"event_slot_bytes"`
	EventCapacity        DiagnosticNumber `json:"event_capacity"`
	HistoryReservedBytes DiagnosticNumber `json:"history_reserved_bytes"`
	HistoryOccupiedBytes DiagnosticNumber `json:"history_occupied_bytes"`
	DetailBytes          DiagnosticNumber `json:"detail_bytes"`
	DetailCapacityBytes  DiagnosticNumber `json:"detail_capacity_bytes"`
	DetailReservedBytes  DiagnosticNumber `json:"detail_reserved_bytes"`
	DetailMetadataBytes  DiagnosticNumber `json:"detail_metadata_bytes"`
	DetailLogicalBytes   DiagnosticNumber `json:"detail_logical_bytes"`
	DetailSlackBytes     DiagnosticNumber `json:"detail_slack_bytes"`
	MetadataBytes        DiagnosticNumber `json:"metadata_bytes"`
	WindowBuckets        DiagnosticNumber `json:"window_buckets"`
	Extensions           DiagnosticNumber `json:"extensions"`
	QueryWorkspaces      DiagnosticNumber `json:"query_workspaces"`
	Projections          DiagnosticNumber `json:"projections"`
	ProjectionWorkspaces DiagnosticNumber `json:"projection_workspaces"`
}

// EntryMemoryFootprint reports one retained event header and its exclusive packed text chain.
type EntryMemoryFootprint struct {
	HeaderBytes         DiagnosticNumber `json:"header_bytes"`
	DetailPayloadBytes  DiagnosticNumber `json:"detail_payload_bytes"`
	DetailMetadataBytes DiagnosticNumber `json:"detail_metadata_bytes"`
	LogicalTextBytes    DiagnosticNumber `json:"logical_text_bytes"`
	DetailSlackBytes    DiagnosticNumber `json:"detail_slack_bytes"`
	OccupiedBytes       DiagnosticNumber `json:"occupied_bytes"`
}

// ComponentMemoryFootprint reports registration-owned diagnostic storage, not plugin heap usage.
type ComponentMemoryFootprint struct {
	RegistrationBytes  DiagnosticNumber `json:"registration_bytes"`
	CounterBytes       DiagnosticNumber `json:"counter_bytes"`
	WindowBytes        DiagnosticNumber `json:"window_bytes"`
	ExtensionBytes     DiagnosticNumber `json:"extension_bytes"`
	RetainedEventCount DiagnosticNumber `json:"retained_event_count"`
	RetainedEventBytes DiagnosticNumber `json:"retained_event_bytes"`
	OccupiedBytes      DiagnosticNumber `json:"occupied_bytes"`
}

// EntryMemoryUsage identifies a retained observation and its backing footprint.
type EntryMemoryUsage struct {
	Sequence  DiagnosticNumber  `json:"sequence"`
	Component ComponentIdentity `json:"component"`
	EntryMemoryFootprint
}

// ComponentMemoryUsage identifies a registration and its complete attributed backing.
type ComponentMemoryUsage struct {
	Component ComponentIdentity `json:"component"`
	ComponentMemoryFootprint
}

// MemorySelectionTotals reports matching row counts and overlapping byte totals.
type MemorySelectionTotals struct {
	Components             DiagnosticNumber `json:"components"`
	Events                 DiagnosticNumber `json:"events"`
	ComponentOccupiedBytes DiagnosticNumber `json:"component_occupied_bytes"`
	EntryOccupiedBytes     DiagnosticNumber `json:"entry_occupied_bytes"`
}

// DiagnosticMemoryPage contains one bounded attribution page and independent resource context.
type DiagnosticMemoryPage struct {
	Loss                            DiagnosticLoss         `json:"loss"`
	SchemaVersion                   uint32                 `json:"schema_version"`
	Source                          SourceIdentity         `json:"source"`
	CapturedAt                      string                 `json:"captured_at"`
	Sequence                        DiagnosticNumber       `json:"sequence"`
	Aggregate                       DiagnosticMemoryUsage  `json:"aggregate"`
	ComponentOccupiedBytes          DiagnosticNumber       `json:"component_occupied_bytes"`
	SharedOrUnassignedReservedBytes DiagnosticNumber       `json:"shared_or_unassigned_reserved_bytes"`
	Selection                       MemorySelectionTotals  `json:"selection"`
	Page                            MemorySelectionTotals  `json:"page"`
	Components                      []ComponentMemoryUsage `json:"components"`
	Entries                         []EntryMemoryUsage     `json:"entries"`
	Resource                        ResourceMemoryView     `json:"resource"`
	Continuation                    *DiagnosticCursor      `json:"continuation"`
}

// ResourceMemoryMeasurement preserves a resource field's numeric kind, unit, scope, and coverage.
type ResourceMemoryMeasurement struct {
	Value            ResourceMemoryNumber `json:"value"`
	Unit             ResourceMemoryUnit   `json:"unit"`
	MeasurementScope string               `json:"measurement_scope"`
	SampledProcesses *DiagnosticNumber    `json:"sampled_processes"`
}

// MemoryLimitEvent preserves one cumulative environment limit event.
type MemoryLimitEvent struct {
	Resource string                     `json:"resource"`
	Event    string                     `json:"event"`
	Count    *ResourceMemoryMeasurement `json:"count"`
}

// ResourceMemoryMetrics contains memory-only measurements and at most eight limit events.
type ResourceMemoryMetrics struct {
	SystemUsed            *ResourceMemoryMeasurement `json:"system_used"`
	SystemTotal           *ResourceMemoryMeasurement `json:"system_total"`
	SystemAvailable       *ResourceMemoryMeasurement `json:"system_available"`
	Resident              *ResourceMemoryMeasurement `json:"resident"`
	Private               *ResourceMemoryMeasurement `json:"private"`
	PhysicalFootprint     *ResourceMemoryMeasurement `json:"physical_footprint"`
	VirtualMemory         *ResourceMemoryMeasurement `json:"virtual_memory"`
	PeakResident          *ResourceMemoryMeasurement `json:"peak_resident"`
	Limit                 *ResourceMemoryMeasurement `json:"limit"`
	EnvironmentAccounted  *ResourceMemoryMeasurement `json:"environment_accounted"`
	SomePressureStallTime *ResourceMemoryMeasurement `json:"some_pressure_stall_time"`
	FullPressureStallTime *ResourceMemoryMeasurement `json:"full_pressure_stall_time"`
	OutOfMemoryEventCount *ResourceMemoryMeasurement `json:"out_of_memory_event_count"`
	LimitEvents           [8]*MemoryLimitEvent       `json:"limit_events"`
}

// ResourceProcessIdentity identifies a root verified during resource collection.
type ResourceProcessIdentity struct {
	ProcessID     uint32           `json:"process_id"`
	StartIdentity DiagnosticNumber `json:"start_identity"`
}

// ResourceMemorySample identifies one collection and preserves its interval and memory values.
type ResourceMemorySample struct {
	CollectorInstanceID string                   `json:"collector_instance_id"`
	RuntimeGeneration   DiagnosticNumber         `json:"runtime_generation"`
	TargetGeneration    DiagnosticNumber         `json:"target_generation"`
	Stream              string                   `json:"stream"`
	Ordinal             DiagnosticNumber         `json:"ordinal"`
	StartedAt           string                   `json:"started_at"`
	CompletedAt         string                   `json:"completed_at"`
	CollectionMillis    DiagnosticNumber         `json:"collection_millis"`
	OperatingSystem     string                   `json:"operating_system"`
	MeasurementScope    string                   `json:"measurement_scope"`
	Root                *ResourceProcessIdentity `json:"root"`
	VisibleProcesses    *DiagnosticNumber        `json:"visible_processes"`
	SampledProcesses    *DiagnosticNumber        `json:"sampled_processes"`
	Memory              ResourceMemoryMetrics    `json:"memory"`
}

// ResourceMemoryView reports independent collection availability, association, and freshness.
type ResourceMemoryView struct {
	Availability        string                `json:"availability"`
	Sample              *ResourceMemorySample `json:"sample"`
	Association         string                `json:"association"`
	AgeMillis           *DiagnosticNumber     `json:"age_millis"`
	Stale               bool                  `json:"stale"`
	ProjectionSlotBytes DiagnosticNumber      `json:"projection_slot_bytes"`
}

// UnmarshalJSON rejects unsupported memory envelopes independently of snapshot versions.
func (p *DiagnosticMemoryPage) UnmarshalJSON(data []byte) error {
	type plain DiagnosticMemoryPage
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.SchemaVersion != 1 {
		return &DiagnosticsError{Reason: "unsupported_schema"}
	}
	*p = DiagnosticMemoryPage(decoded)
	return nil
}

// InspectMemory returns a caller-owned exact backing/resource view.
func (reader *DiagnosticsReader) InspectMemory(query DiagnosticQuery) (DiagnosticMemoryPage, error) {
	var page DiagnosticMemoryPage
	raw, err := reader.InspectMemoryJSON(query)
	if err != nil {
		return page, err
	}
	err = json.Unmarshal([]byte(raw), &page)
	return page, err
}
