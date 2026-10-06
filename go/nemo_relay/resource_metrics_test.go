// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestResourceMetricValuePreservesNumbersAndRejectsInvalidJSON(t *testing.T) {
	for _, encoded := range []string{"0", "18446744073709551615", "1.25", "-1.25"} {
		var value ResourceMetricValue
		if err := json.Unmarshal([]byte(encoded), &value); err != nil {
			t.Fatalf("decode %s: %v", encoded, err)
		}
		actual, err := json.Marshal(value)
		if err != nil || string(actual) != encoded {
			t.Fatalf("round trip %s: %s, %v", encoded, actual, err)
		}
	}
	var value ResourceMetricValue
	for _, encoded := range []string{"null", "\"1\"", "1e9999"} {
		if err := json.Unmarshal([]byte(encoded), &value); err == nil {
			t.Fatalf("accepted invalid measurement %s", encoded)
		}
	}
	if _, err := json.Marshal(ResourceMetricValue{}); err == nil {
		t.Fatal("serialized a measurement without a value")
	}
	if _, err := CollectResourceMetrics(nil); err == nil {
		t.Fatal("accepted a nil context")
	}
}

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
	if snapshot.Memory == nil || snapshot.Memory.SystemTotal == nil || snapshot.Memory.SystemTotal.Unit != CapacityMebibytes {
		t.Fatalf("configured memory unit missing: %+v", snapshot.Memory)
	}
	value := snapshot.Memory.SystemTotal.Value
	if value.Integer == nil && value.Decimal == nil {
		t.Fatal("memory value is neither integer nor decimal")
	}
	if snapshot.Network == nil || snapshot.Network.MeasurementScope != ResourceMeasurementScopeGlobal {
		t.Fatalf("network must be global: %+v", snapshot.Network)
	}
	if received := snapshot.Network.System.ReceivedData; received != nil && received.Unit != DataMegabytes {
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

func TestResourceMetricsWaitHandlesPendingCompletionAndCancellation(t *testing.T) {
	polls := 0
	snapshot, err := waitForResourceMetrics(context.Background(), func() (bool, []byte, error) {
		polls++
		return polls == 9, []byte(`{"measurement_scope":"global"}`), nil
	})
	if err != nil || snapshot.MeasurementScope != ResourceMeasurementScopeGlobal || polls != 9 {
		t.Fatalf("pending collection: snapshot=%+v polls=%d error=%v", snapshot, polls, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err = waitForResourceMetrics(ctx, func() (bool, []byte, error) {
		cancel()
		return false, nil, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pending cancellation: %v", err)
	}
	polls = 0
	_, err = waitForResourceMetrics(ctx, func() (bool, []byte, error) {
		polls++
		return true, nil, nil
	})
	if !errors.Is(err, context.Canceled) || polls != 0 {
		t.Fatalf("cancelled collection called poll: polls=%d error=%v", polls, err)
	}
}

func TestResourceMetricsWaitPropagatesBackendAndDecodingErrors(t *testing.T) {
	backendError := errors.New("resource collection failed")
	_, err := waitForResourceMetrics(context.Background(), func() (bool, []byte, error) {
		return false, nil, backendError
	})
	if !errors.Is(err, backendError) {
		t.Fatalf("backend error: %v", err)
	}
	for _, encoded := range []string{"invalid", `{"cpu":{"user_time":{"value":"invalid","unit":"milliseconds"}}}`} {
		_, err := waitForResourceMetrics(context.Background(), func() (bool, []byte, error) {
			return true, []byte(encoded), nil
		})
		if err == nil {
			t.Fatalf("accepted invalid snapshot %s", encoded)
		}
	}
}

func TestResourceMetricsCollectionRejectsStartupErrorsAndNullFFIOutputs(t *testing.T) {
	startupError := errors.New("collection startup failed")
	_, err := collectResourceMetrics(context.Background(), func() (*resourceMetricsCollection, error) {
		return nil, startupError
	})
	if !errors.Is(err, startupError) {
		t.Fatalf("startup error: %v", err)
	}
	_, err = collectResourceMetrics(context.Background(), func() (*resourceMetricsCollection, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("accepted a null collection from FFI")
	}
	if data, err := resourceMetricsJSON(nil); err == nil || data != nil {
		t.Fatalf("accepted a null snapshot from FFI: data=%s error=%v", data, err)
	}
}

func TestResourceMetricsValidatesThroughTheNativePluginHostWithAnExplicitConfigFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "resource-metrics.toml")
	document := `# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[[components]]
kind = "resource_metrics"
enabled = true

[components.config.polling]
enabled = false
interval_millis = 3210

[components.config.units.memory]
resident = "mebibytes"
`
	if err := os.WriteFile(path, []byte(document), 0600); err != nil {
		t.Fatal(err)
	}
	report, err := Validate(PluginConfig{Version: 1}, &path)
	if err != nil {
		t.Fatalf("validate explicit resource metrics configuration: %v", err)
	}
	for _, diagnostic := range report.Config.Diagnostics {
		if diagnostic.Level == DiagnosticLevelError {
			t.Fatalf("invalid resource metrics configuration: %+v", diagnostic)
		}
	}
	components, ok := report.ResolvedConfig["components"].([]any)
	if !ok {
		t.Fatalf("resolved components are missing: %+v", report.ResolvedConfig)
	}
	for _, raw := range components {
		component := raw.(map[string]any)
		if component["kind"] != "resource_metrics" {
			continue
		}
		config := component["config"].(map[string]any)
		polling := config["polling"].(map[string]any)
		units := config["units"].(map[string]any)
		memory := units["memory"].(map[string]any)
		if polling["enabled"] != false || polling["interval_millis"] != float64(3210) || memory["resident"] != "mebibytes" {
			t.Fatalf("resolved resource metrics options do not match the file: %+v", config)
		}
		return
	}
	t.Fatalf("resource metrics component is missing: %+v", report.ResolvedConfig)
}

func checkResourceUnits[U ResourceUnit](t *testing.T, units []U) {
	t.Helper()
	for _, unit := range units {
		original := ResourceMeasurement[uint64, U]{Value: 1, Unit: unit}
		wire, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var decoded ResourceMeasurement[uint64, U]
		if err := json.Unmarshal(wire, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded != original {
			t.Fatalf("measurement changed: %+v != %+v", decoded, original)
		}
		for _, invalid := range []string{`{"value":1,"unit":"invalid"}`, `{"value":1,"unit":42}`} {
			if err := json.Unmarshal([]byte(invalid), &decoded); err == nil {
				t.Fatalf("accepted %s", invalid)
			}
		}
	}
}

func TestResourceMeasurementsUseSemanticUnitCategories(t *testing.T) {
	checkResourceUnits(t, []DurationUnit{DurationMicroseconds, DurationMilliseconds, DurationSeconds, DurationMinutes})
	checkResourceUnits(t, []CapacityUnit{CapacityBytes, CapacityKilobytes, CapacityMegabytes, CapacityGigabytes, CapacityTerabytes, CapacityKibibytes, CapacityMebibytes, CapacityGibibytes, CapacityTebibytes})
	checkResourceUnits(t, []DataUnit{DataBytes, DataKilobytes, DataMegabytes, DataGigabytes, DataTerabytes, DataKibibytes, DataMebibytes, DataGibibytes, DataTebibytes})
	checkResourceUnits(t, []BandwidthUnit{BandwidthBytesPerSecond, BandwidthKibibytesPerSecond, BandwidthMebibytesPerSecond, BandwidthGibibytesPerSecond, BandwidthBitsPerSecond, BandwidthMegabitsPerSecond, BandwidthGigabitsPerSecond})
	checkResourceUnits(t, []CpuUnit{CpuLogicalProcessors, CpuMillicores})
	checkResourceUnits(t, []UtilizationUnit{UtilizationPercentage, UtilizationFraction})
	checkResourceUnits(t, []CountUnit{CountProcesses, CountThreads, CountFileDescriptors, CountHandles, CountEvents, CountOperations, CountPackets, CountErrors})
}

func TestDurationMeasurementsRejectOtherUnitCategories(t *testing.T) {
	for _, unit := range []string{"bytes", "megabits_per_second", "logical_processors", "fraction", "events"} {
		var measurement ResourceMeasurement[uint64, DurationUnit]
		if err := json.Unmarshal([]byte(`{"value":1,"unit":"`+unit+`"}`), &measurement); err == nil {
			t.Fatalf("accepted duration unit %q", unit)
		}
	}
}
