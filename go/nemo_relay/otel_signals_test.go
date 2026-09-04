// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	emitEventFailedMsg = "EmitEvent failed"
	otelEndpoint       = "http://localhost:4318"
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
			WithEventData(json.RawMessage(`{"message":"ignored"}`)),
			WithEventData(json.RawMessage(`{"message":"hello"}`)),
			WithEventDataSchema(DataSchema{Name: "ignored", Version: "0"}),
			WithEventDataSchema(DataSchema{Name: "example.log", Version: "1"}),
			WithEventMetadata(json.RawMessage(`{"context":"ignored"}`)),
			WithEventMetadata(json.RawMessage(`{"nemo_relay.log.severity":"debug"}`)),
			WithEventSeverity(LogSeverityWarn),
		)
		requireNoError(t, err, emitEventFailedMsg)

		err = EmitMetric("go_metric", []MetricMeasurement{{
			Name:      "example.tokens.saved",
			Kind:      MetricKindCounter,
			ValueType: MetricValueTypeU64,
			Value:     uint64(42),
			Unit:      "{token}",
			Attributes: map[string]interface{}{
				"model": "example-model",
			},
		}},
			WithMetricMetadata(json.RawMessage(`{"context":"ignored"}`)),
			WithMetricMetadata(json.RawMessage(`{"context":"final"}`)),
		)
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
	if !bytes.Equal(events[0].Data(), []byte(`{"message":"hello"}`)) {
		t.Fatalf("final event data should replace the earlier value: %s", events[0].Data())
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal(events[0].Metadata(), &metadata); err != nil {
		t.Fatalf("decode log metadata: %v", err)
	}
	if metadata["nemo_relay.log.severity"] != "warn" {
		t.Fatalf("typed severity did not override metadata: %#v", metadata)
	}
	if _, ok := metadata["context"]; ok {
		t.Fatalf("final event metadata should replace the earlier value: %#v", metadata)
	}
	if !bytes.Contains(events[1].DataSchema(), []byte("nemo.relay.metric_measurements")) {
		t.Fatalf("metric schema missing: %s", events[1].DataSchema())
	}
	if !bytes.Contains(events[1].Data(), []byte("example.tokens.saved")) {
		t.Fatalf("metric envelope missing measurement: %s", events[1].Data())
	}
	var metricMetadata map[string]interface{}
	if err := json.Unmarshal(events[1].Metadata(), &metricMetadata); err != nil {
		t.Fatalf("decode metric metadata: %v", err)
	}
	if metricMetadata["context"] != "final" {
		t.Fatalf("final metric metadata should replace the earlier value: %#v", metricMetadata)
	}
}

func TestEventAndMetricValidationErrors(t *testing.T) {
	if err := EmitEvent("invalid_severity", WithEventSeverity(LogSeverity("verbose"))); err == nil {
		t.Fatal("expected invalid severity to fail")
	} else if !strings.Contains(err.Error(), "invalid log severity") {
		t.Fatalf("unexpected severity error: %v", err)
	}
	runWithTestScopeStack(t, func() {
		if err := EmitEvent(
			"invalid_metadata",
			WithEventMetadata(json.RawMessage(`[]`)),
			WithEventSeverity(LogSeverityInfo),
		); err == nil {
			t.Fatal("expected severity with non-object metadata to fail")
		} else if !strings.Contains(err.Error(), "mark metadata must be a JSON object") {
			t.Fatalf("unexpected metadata error: %v", err)
		}
		if err := EmitMetric("empty_metric", []MetricMeasurement{}); err == nil {
			t.Fatal("expected empty measurements to fail")
		} else if !strings.Contains(err.Error(), "measurements must contain at least one entry") {
			t.Fatalf("unexpected empty measurement error: %v", err)
		}
	})
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
					Endpoint: otelEndpoint,
					Timeout:  time.Nanosecond,
				})
				return err
			},
		},
		{
			name: "log scheduled delay",
			normalize: func() error {
				_, err := normalizeOpenTelemetryLogConfig(OpenTelemetryLogConfig{
					Endpoint:       otelEndpoint,
					ScheduledDelay: time.Millisecond + 500*time.Microsecond,
				})
				return err
			},
		},
		{
			name: "metric timeout",
			normalize: func() error {
				_, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{
					Endpoint: otelEndpoint,
					Timeout:  500 * time.Microsecond,
				})
				return err
			},
		},
		{
			name: "metric export interval",
			normalize: func() error {
				_, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{
					Endpoint:       otelEndpoint,
					ExportInterval: 2*time.Millisecond + 500*time.Microsecond,
				})
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.normalize() == nil {
				t.Fatal("expected fractional-millisecond duration to fail")
			}
		})
	}
}

func TestOpenTelemetrySubscribersExposeRuntimeDiagnostics(t *testing.T) {
	endpoint := "http://127.0.0.1:4318/v1/traces"
	variable := "NEMO_RELAY_GO_SIGNAL_HEADER_" + time.Now().Format(otelTimeFormat)
	t.Setenv(variable, "signal-route")
	traceConfig := NewOpenTelemetryConfig(OpenTelemetryTypeFull, endpoint)
	traceConfig.HeaderEnv["x-relay-route"] = variable
	traceSubscriber, err := NewOpenTelemetrySubscriber(traceConfig)
	requireNoError(t, err, "NewOpenTelemetrySubscriber failed")
	defer traceSubscriber.Close()
	logConfig := NewOpenTelemetryLogConfig(endpoint)
	logConfig.HeaderEnv["x-relay-route"] = variable
	logSubscriber, err := NewOpenTelemetryLogSubscriber(logConfig)
	requireNoError(t, err, "NewOpenTelemetryLogSubscriber failed")
	defer logSubscriber.Close()
	metricConfig := NewOpenTelemetryMetricConfig(endpoint)
	metricConfig.HeaderEnv["x-relay-route"] = variable
	metricSubscriber, err := NewOpenTelemetryMetricSubscriber(metricConfig)
	requireNoError(t, err, "NewOpenTelemetryMetricSubscriber failed")
	defer metricSubscriber.Close()

	subscribers := []struct {
		name               string
		register           func(string) error
		deregister         func(string) error
		runtimeDiagnostics func() ([]OpenTelemetryRuntimeDiagnostic, error)
	}{
		{"trace", traceSubscriber.Register, traceSubscriber.Deregister, traceSubscriber.RuntimeDiagnostics},
		{"log", logSubscriber.Register, logSubscriber.Deregister, logSubscriber.RuntimeDiagnostics},
		{"metric", metricSubscriber.Register, metricSubscriber.Deregister, metricSubscriber.RuntimeDiagnostics},
	}
	for index := range subscribers {
		subscribers[index].name = "go_otel_" + subscribers[index].name + "_diagnostics_" + time.Now().Format(otelTimeFormat)
		requireNoError(t, subscribers[index].register(subscribers[index].name), "subscriber Register failed")
		defer func(subscriberName string, deregister func(string) error) {
			_ = deregister(subscriberName)
		}(subscribers[index].name, subscribers[index].deregister)
	}

	runWithTestScopeStack(t, func() {
		requireNoError(t, EmitEvent(
			"invalid_metric",
			WithEventData(json.RawMessage(`{"measurements":[]}`)),
			WithEventDataSchema(DataSchema{Name: "nemo.relay.metric_measurements", Version: "999"}),
		), emitEventFailedMsg)
	})
	requireNoError(t, FlushSubscribers(), "FlushSubscribers failed")

	for _, subscriber := range subscribers {
		diagnostics, err := subscriber.runtimeDiagnostics()
		requireNoError(t, err, "RuntimeDiagnostics failed")
		var invalidMetric *OpenTelemetryRuntimeDiagnostic
		for index := range diagnostics {
			if diagnostics[index].Code == "otel.metric_mark_invalid" {
				invalidMetric = &diagnostics[index]
				break
			}
		}
		if invalidMetric == nil {
			t.Fatalf("expected invalid metric diagnostic for %s subscriber, got %#v", subscriber.name, diagnostics)
		}
		if invalidMetric.Count != 1 || !bytes.Contains([]byte(invalidMetric.Message), []byte("unsupported metric schema version")) {
			t.Fatalf("unexpected invalid metric diagnostic for %s subscriber: %#v", subscriber.name, invalidMetric)
		}
	}
}

func TestOpenTelemetryLogSubscriberLifecycleAndDerivation(t *testing.T) {
	requests := make(chan otelRequest, 4)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	variable := "NEMO_RELAY_GO_LOG_HEADER_" + time.Now().Format(otelTimeFormat)
	secret := "go-log-activation-route"
	t.Setenv(variable, secret)
	config := NewOpenTelemetryLogConfig(server.URL + "/v1/traces")
	config.ServiceName = "go-log-test"
	config.HeaderEnv["x-relay-route"] = variable
	subscriber, err := NewOpenTelemetryLogSubscriber(config)
	requireNoError(t, err, "NewOpenTelemetryLogSubscriber failed")
	t.Setenv(variable, "go-log-changed-route")
	defer subscriber.Close()
	name := "go_otel_log_" + time.Now().Format(otelTimeFormat)
	requireNoError(t, subscriber.Register(name), "log Register failed")
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		requireNoError(t, EmitEvent("go_exported_log", WithEventSeverity(LogSeverityError)), emitEventFailedMsg)
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
		if request.RelayRoute != secret {
			t.Fatalf("expected activation-time log header, got %q", request.RelayRoute)
		}
		if bytes.Contains(request.Body, []byte(secret)) {
			t.Fatal("log export body exposed the environment-derived header value")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP log request")
	}
	diagnostics, err := subscriber.RuntimeDiagnostics()
	requireNoError(t, err, "log RuntimeDiagnostics failed")
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, secret) {
			t.Fatal("log runtime diagnostics exposed the environment-derived header value")
		}
	}
	requireNoError(t, subscriber.Deregister(name), "log Deregister failed")
	requireNoError(t, subscriber.Shutdown(), "log Shutdown failed")
}

func TestOpenTelemetryMetricSubscriberLifecycleAndDerivation(t *testing.T) {
	requests := make(chan otelRequest, 4)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	variable := "NEMO_RELAY_GO_METRIC_HEADER_" + time.Now().Format(otelTimeFormat)
	secret := "go-metric-activation-route"
	t.Setenv(variable, secret)
	config := NewOpenTelemetryMetricConfig(server.URL + "/v1/traces")
	config.ServiceName = "go-metric-test"
	config.HeaderEnv["x-relay-route"] = variable
	subscriber, err := NewOpenTelemetryMetricSubscriber(config)
	requireNoError(t, err, "NewOpenTelemetryMetricSubscriber failed")
	t.Setenv(variable, "go-metric-changed-route")
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
		if request.RelayRoute != secret {
			t.Fatalf("expected activation-time metric header, got %q", request.RelayRoute)
		}
		if bytes.Contains(request.Body, []byte(secret)) {
			t.Fatal("metric export body exposed the environment-derived header value")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP metric request")
	}
	diagnostics, err := subscriber.RuntimeDiagnostics()
	requireNoError(t, err, "metric RuntimeDiagnostics failed")
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, secret) {
			t.Fatal("metric runtime diagnostics exposed the environment-derived header value")
		}
	}
	requireNoError(t, subscriber.Deregister(name), "metric Deregister failed")
	requireNoError(t, subscriber.Shutdown(), "metric Shutdown failed")
}
