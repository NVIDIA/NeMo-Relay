// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"sync"
	"testing"
	"time"
)

func TestEventSchemaSeverityAndMetricParity(t *testing.T) {
	var (
		events []Event
		mu     sync.Mutex
	)
	name := "go_signal_api_" + time.Now().Format(otelTimeFormat)
	requireNoError(t, RegisterSubscriber(name, func(event Event) {
		if event.Name() == "go_structured_log" || event.Name() == "go_metric" {
			mu.Lock()
			events = append(events, event)
			mu.Unlock()
		}
	}), "RegisterSubscriber failed")
	defer func() { _ = DeregisterSubscriber(name) }()

	runWithTestScopeStack(t, func() {
		err := EmitEvent(
			"go_structured_log",
			WithEventData(json.RawMessage(`{"message":"hello"}`)),
			WithEventDataSchema(DataSchema{Name: "example.log", Version: "1"}),
			WithEventMetadata(json.RawMessage(`{"nemo_relay.log.severity":"debug"}`)),
			WithEventSeverity(LogSeverityWarn),
		)
		requireNoError(t, err, "EmitEvent failed")

		err = EmitMetric("go_metric", []MetricMeasurement{{
			Name:      "example.tokens.saved",
			Kind:      MetricKindCounter,
			ValueType: MetricValueTypeU64,
			Value:     uint64(42),
			Unit:      "{token}",
			Attributes: map[string]interface{}{
				"model": "example-model",
			},
		}})
		requireNoError(t, err, "EmitMetric failed")
	})
	requireNoError(t, FlushSubscribers(), "FlushSubscribers failed")

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 2 {
		t.Fatalf("expected 2 marks, got %d", len(events))
	}
	var schema DataSchema
	if err := json.Unmarshal(events[0].DataSchema(), &schema); err != nil {
		t.Fatalf("decode log schema: %v", err)
	}
	if schema.Name != "example.log" || schema.Version != "1" {
		t.Fatalf("unexpected log schema: %#v", schema)
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal(events[0].Metadata(), &metadata); err != nil {
		t.Fatalf("decode log metadata: %v", err)
	}
	if metadata["nemo_relay.log.severity"] != "warn" {
		t.Fatalf("typed severity did not override metadata: %#v", metadata)
	}
	if !bytes.Contains(events[1].DataSchema(), []byte("nemo.relay.metric_measurements")) {
		t.Fatalf("metric schema missing: %s", events[1].DataSchema())
	}
	if !bytes.Contains(events[1].Data(), []byte("example.tokens.saved")) {
		t.Fatalf("metric envelope missing measurement: %s", events[1].Data())
	}
}

func TestEventAndMetricValidationErrors(t *testing.T) {
	if err := EmitEvent("invalid_severity", WithEventSeverity(LogSeverity("verbose"))); err == nil {
		t.Fatal("expected invalid severity to fail")
	}
	if err := EmitEvent(
		"invalid_metadata",
		WithEventMetadata(json.RawMessage(`[]`)),
		WithEventSeverity(LogSeverityInfo),
	); err == nil {
		t.Fatal("expected severity with non-object metadata to fail")
	}
	if err := EmitMetric("empty_metric", nil); err == nil {
		t.Fatal("expected empty measurements to fail")
	}
}

func TestWithMetricParentRetainsScopeHandle(t *testing.T) {
	parent := &ScopeHandle{}
	options := &metricOptions{}

	WithMetricParent(parent)(options)

	if options.parentHandle != parent {
		t.Fatal("metric options did not retain the parent scope handle")
	}
}

func TestMetricMeasurementBoundarySerializationPreservesNilAndEmpty(t *testing.T) {
	encoded, err := json.Marshal([]MetricMeasurement{
		{Boundaries: nil},
		{Boundaries: []float64{}},
	})
	requireNoError(t, err, "marshal metric measurements failed")

	var measurements []map[string]interface{}
	requireNoError(t, json.Unmarshal(encoded, &measurements), "decode metric measurements failed")
	if boundaries, ok := measurements[0]["boundaries"]; !ok || boundaries != nil {
		t.Fatalf("nil boundaries must serialize as null, got %#v", boundaries)
	}
	boundaries, ok := measurements[1]["boundaries"].([]interface{})
	if !ok || len(boundaries) != 0 {
		t.Fatalf("explicit empty boundaries must serialize as [], got %#v", measurements[1]["boundaries"])
	}
}

func TestOpenTelemetrySignalConfigRejectsFractionalMillisecondDurations(t *testing.T) {
	tests := []struct {
		name      string
		normalize func() error
	}{
		{
			name: "log timeout",
			normalize: func() error {
				_, err := normalizeOpenTelemetryLogConfig(OpenTelemetryLogConfig{
					Endpoint: "http://localhost:4318",
					Timeout:  time.Nanosecond,
				})
				return err
			},
		},
		{
			name: "log scheduled delay",
			normalize: func() error {
				_, err := normalizeOpenTelemetryLogConfig(OpenTelemetryLogConfig{
					Endpoint:       "http://localhost:4318",
					ScheduledDelay: time.Millisecond + 500*time.Microsecond,
				})
				return err
			},
		},
		{
			name: "metric timeout",
			normalize: func() error {
				_, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{
					Endpoint: "http://localhost:4318",
					Timeout:  500 * time.Microsecond,
				})
				return err
			},
		},
		{
			name: "metric export interval",
			normalize: func() error {
				_, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{
					Endpoint:       "http://localhost:4318",
					ExportInterval: 2*time.Millisecond + 500*time.Microsecond,
				})
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.normalize(); err == nil {
				t.Fatal("expected fractional-millisecond duration to fail")
			}
		})
	}
}

func TestOpenTelemetryLogSubscriberLifecycleAndDerivation(t *testing.T) {
	requests := make(chan otelRequest, 4)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	config := NewOpenTelemetryLogConfig(server.URL + "/v1/traces")
	config.ServiceName = "go-log-test"
	subscriber, err := NewOpenTelemetryLogSubscriber(config)
	requireNoError(t, err, "NewOpenTelemetryLogSubscriber failed")
	defer subscriber.Close()
	name := "go_otel_log_" + time.Now().Format(otelTimeFormat)
	requireNoError(t, subscriber.Register(name), "log Register failed")
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		requireNoError(t, EmitEvent("go_exported_log", WithEventSeverity(LogSeverityError)), "EmitEvent failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "log ForceFlush failed")

	select {
	case request := <-requests:
		if request.Path != "/v1/logs" {
			t.Fatalf("expected /v1/logs path, got %q", request.Path)
		}
		if !bytes.Contains(request.Body, []byte("go_exported_log")) {
			t.Fatal("log export did not contain mark name")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP log request")
	}
	requireNoError(t, subscriber.Deregister(name), "log Deregister failed")
	requireNoError(t, subscriber.Shutdown(), "log Shutdown failed")
}

func TestOpenTelemetryMetricSubscriberLifecycleAndDerivation(t *testing.T) {
	requests := make(chan otelRequest, 4)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	config := NewOpenTelemetryMetricConfig(server.URL + "/v1/traces")
	config.ServiceName = "go-metric-test"
	subscriber, err := NewOpenTelemetryMetricSubscriber(config)
	requireNoError(t, err, "NewOpenTelemetryMetricSubscriber failed")
	defer subscriber.Close()
	name := "go_otel_metric_" + time.Now().Format(otelTimeFormat)
	requireNoError(t, subscriber.Register(name), "metric Register failed")
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		requireNoError(t, EmitMetric("go_exported_metric", []MetricMeasurement{{
			Name:      "example.requests",
			Kind:      MetricKindCounter,
			ValueType: MetricValueTypeU64,
			Value:     uint64(1),
		}}), "EmitMetric failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "metric ForceFlush failed")

	select {
	case request := <-requests:
		if request.Path != "/v1/metrics" {
			t.Fatalf("expected /v1/metrics path, got %q", request.Path)
		}
		if !bytes.Contains(request.Body, []byte("example.requests")) {
			t.Fatal("metric export did not contain instrument name")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP metric request")
	}
	requireNoError(t, subscriber.Deregister(name), "metric Deregister failed")
	requireNoError(t, subscriber.Shutdown(), "metric Shutdown failed")
}
