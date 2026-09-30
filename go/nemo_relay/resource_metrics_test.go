// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"errors"
	"testing"
)

func TestCollectResourceMetricsRequiresActiveComponent(t *testing.T) {
	if err := closeTestPluginHost(); err != nil {
		t.Fatalf("closeTestPluginHost() error = %v", err)
	}
	if _, err := CollectResourceMetrics(context.Background()); err == nil {
		t.Fatal("CollectResourceMetrics() succeeded without an active resource_metrics component")
	}
}

func TestCollectResourceMetricsReturnsCanonicalSnapshot(t *testing.T) {
	if _, err := initializeTestPluginHost(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{{
			Kind:    "resource_metrics",
			Enabled: true,
			Config:  map[string]any{"polling": map[string]any{"enabled": false}},
		}},
	}); err != nil {
		t.Fatalf("initializeTestPluginHost() error = %v", err)
	}
	t.Cleanup(func() {
		if err := closeTestPluginHost(); err != nil {
			t.Errorf("closeTestPluginHost() error = %v", err)
		}
	})

	snapshot, err := CollectResourceMetrics(context.Background())
	if err != nil {
		t.Fatalf("CollectResourceMetrics() error = %v", err)
	}
	if snapshot.OperatingSystem == "" {
		t.Fatal("snapshot operating_system is empty")
	}
	if snapshot.Timestamp.IsZero() {
		t.Fatal("snapshot timestamp is zero")
	}
	if snapshot.MeasurementScope != ResourceMeasurementScopeProcessTree || snapshot.Process == nil || snapshot.Process.ActiveCount == nil {
		t.Fatalf("snapshot metadata or process category is missing: %+v", snapshot)
	}
	if snapshot.ProcessSampling == nil || snapshot.ProcessSampling.SampledProcesses == 0 || snapshot.ProcessSampling.SampledProcesses > snapshot.ProcessSampling.VisibleProcesses {
		t.Fatalf("snapshot process sampling metadata is invalid: %+v", snapshot.ProcessSampling)
	}
	if _, ok := snapshot.ProcessSampling.FieldSampledProcesses["cpu.total_time"]; !ok {
		t.Fatal("snapshot is missing canonical CPU field coverage")
	}
	for _, field := range []string{"disk.read_throughput", "disk.write_throughput"} {
		count, ok := snapshot.ProcessSampling.FieldSampledProcesses[field]
		if !ok || count != 0 {
			t.Fatalf("first-sample coverage for %s = %d (present: %t), want 0", field, count, ok)
		}
	}
	for field, count := range snapshot.ProcessSampling.FieldSampledProcesses {
		if count > snapshot.ProcessSampling.SampledProcesses {
			t.Fatalf("field %s coverage exceeds sampled process count: %d", field, count)
		}
	}
	if snapshot.CPU != nil && snapshot.CPU.UserTime != nil && snapshot.CPU.UserTime.Unit != "milliseconds" {
		t.Fatalf("CPU user-time unit = %v, want milliseconds", snapshot.CPU.UserTime.Unit)
	}
	if snapshot.Memory != nil && snapshot.Memory.Resident != nil && snapshot.Memory.Resident.Unit != "kibibytes" {
		t.Fatalf("resident-memory unit = %v, want kibibytes", snapshot.Memory.Resident.Unit)
	}
	if snapshot.Memory == nil || snapshot.Memory.SystemTotal == nil || snapshot.Memory.SystemTotal.Unit != "kibibytes" {
		t.Fatalf("process-tree snapshot should include system memory in kibibytes: %+v", snapshot.Memory)
	}
}

func TestCollectResourceMetricsSupportsGlobalScope(t *testing.T) {
	if _, err := initializeTestPluginHost(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{{
			Kind:    "resource_metrics",
			Enabled: true,
			Config: map[string]any{
				"measurement_scope": "global",
				"polling":           map[string]any{"enabled": false},
			},
		}},
	}); err != nil {
		t.Fatalf("initializeTestPluginHost() error = %v", err)
	}
	t.Cleanup(func() {
		if err := closeTestPluginHost(); err != nil {
			t.Errorf("closeTestPluginHost() error = %v", err)
		}
	})

	snapshot, err := CollectResourceMetrics(context.Background())
	if err != nil {
		t.Fatalf("CollectResourceMetrics() error = %v", err)
	}
	if snapshot.MeasurementScope != ResourceMeasurementScopeGlobal {
		t.Fatalf("measurement scope = %q, want global", snapshot.MeasurementScope)
	}
	if snapshot.Memory == nil || snapshot.Memory.SystemTotal == nil {
		t.Fatalf("global system memory is missing: %+v", snapshot.Memory)
	}
	if snapshot.Memory.SystemTotal.Unit != "kibibytes" {
		t.Fatalf("system memory unit = %q, want kibibytes", snapshot.Memory.SystemTotal.Unit)
	}
}

func TestCollectResourceMetricsAppliesUnitsAndLabelsNetworkGlobal(t *testing.T) {
	if _, err := initializeTestPluginHost(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{{
			Kind: "resource_metrics", Enabled: true,
			Config: map[string]any{
				"polling": map[string]any{"enabled": false},
				"units": map[string]any{
					"memory":  map[string]any{"system_total": "mebibytes"},
					"network": map[string]any{"system": map[string]any{"received_data": "megabytes"}},
				},
			},
		}},
	}); err != nil {
		t.Fatalf("initializeTestPluginHost: %v", err)
	}
	t.Cleanup(func() {
		if err := closeTestPluginHost(); err != nil {
			t.Errorf("closeTestPluginHost: %v", err)
		}
	})
	snapshot, err := CollectResourceMetrics(context.Background())
	if err != nil {
		t.Fatalf("CollectResourceMetrics: %v", err)
	}
	if snapshot.Memory == nil || snapshot.Memory.SystemTotal == nil || snapshot.Memory.SystemTotal.Unit != ResourceUnitMebibytes {
		t.Fatalf("configured memory unit missing: %+v", snapshot.Memory)
	}
	value := snapshot.Memory.SystemTotal.Value
	if value.Integer == nil && value.Decimal == nil {
		t.Fatal("memory value is neither integer nor decimal")
	}
	if snapshot.Network == nil || snapshot.Network.MeasurementScope != ResourceMeasurementScopeGlobal {
		t.Fatalf("network must be global: %+v", snapshot.Network)
	}
	if received := snapshot.Network.System.ReceivedData; received != nil && received.Unit != ResourceUnitMegabytes {
		t.Fatalf("configured network unit = %q", received.Unit)
	}
}

func TestCollectResourceMetricsHonorsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := CollectResourceMetrics(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("CollectResourceMetrics() error = %v, want context.Canceled", err)
	}
}
