// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func assertOtlpStringAttribute(t *testing.T, body []byte, key string, value string) {
	t.Helper()
	encoded := append([]byte{0x0a}, binary.AppendUvarint(nil, uint64(len(key)))...)
	encoded = append(encoded, key...)
	attributeValue := append([]byte{0x0a}, binary.AppendUvarint(nil, uint64(len(value)))...)
	attributeValue = append(attributeValue, value...)
	encoded = append(encoded, 0x12)
	encoded = binary.AppendUvarint(encoded, uint64(len(attributeValue)))
	encoded = append(encoded, attributeValue...)
	if !bytes.Contains(body, encoded) {
		t.Fatalf("expected OTLP string attribute %s=%s", key, value)
	}
}

func TestNewOpenTelemetryConfigDefaults(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, "http://localhost:4318/v1/traces")

	if config.Transport != OpenTelemetryTransportHTTPBinary {
		t.Fatalf("expected default transport http_binary, got %q", config.Transport)
	}
	if config.ServiceName != "unknown_service" {
		t.Fatalf("expected default service name unknown_service, got %q", config.ServiceName)
	}
	if config.InstrumentationScope != "opentelemetry" {
		t.Fatalf("expected default instrumentation scope, got %q", config.InstrumentationScope)
	}
	if config.Timeout != 3*time.Second {
		t.Fatalf("expected default timeout 3s, got %v", config.Timeout)
	}
	if config.Headers == nil || len(config.Headers) != 0 {
		t.Fatalf("expected empty headers map, got %#v", config.Headers)
	}
	if config.ResourceAttributes == nil || len(config.ResourceAttributes) != 0 {
		t.Fatalf("expected empty resource attributes map, got %#v", config.ResourceAttributes)
	}
	if config.MarkProjection != MarkProjectionInherit {
		t.Fatalf("expected default mark projection inherit, got %q", config.MarkProjection)
	}
	if len(config.MarkExcludeNames) != 1 || config.MarkExcludeNames[0] != "llm.chunk" {
		t.Fatalf("expected default mark exclusion, got %#v", config.MarkExcludeNames)
	}
	if config.AttributeMappings == nil || len(config.AttributeMappings) != 0 {
		t.Fatalf("expected empty attribute mappings, got %#v", config.AttributeMappings)
	}
}

func TestOpenTelemetrySubscriberAcceptsProjectionControls(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, "http://localhost:4318/v1/traces")
	config.MarkProjection = MarkProjectionTool
	config.MarkExcludeNames = []string{"custom.mark"}
	config.AttributeMappings = []OtlpAttributeMapping{{
		Key:   "nemo_relay.model_name",
		Alias: "model.alias",
	}}

	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber with projection controls failed: %v", err)
	}
	defer subscriber.Close()
}

func TestOpenTelemetrySubscriberRejectsInvalidAttributeMappings(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, "http://localhost:4318/v1/traces")
	config.AttributeMappings = []OtlpAttributeMapping{{Key: "", Alias: "model.alias"}}

	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected invalid attribute mapping error")
	}
}

func TestOpenTelemetrySubscriberLifecycle(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, "http://localhost:4318/v1/traces")
	config.ServiceName = "go-agent"
	config.ServiceNamespace = "agents"
	config.ServiceVersion = "1.0.0"
	config.InstrumentationScope = "go-tests"
	config.Timeout = 1250 * time.Millisecond
	config.Headers["authorization"] = "Bearer token"
	config.ResourceAttributes["deployment.environment"] = "test"
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber failed: %v", err)
	}
	defer subscriber.Close()

	name := "go_otel_subscriber_" + time.Now().Format("150405.000000")
	if err := subscriber.Register(name); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	if err := subscriber.Deregister(name); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}
	if err := subscriber.Deregister(name); err != nil {
		t.Fatalf("repeated Deregister should be safe, got: %v", err)
	}
	if err := subscriber.ForceFlush(); err != nil {
		t.Fatalf("ForceFlush failed: %v", err)
	}
	if err := subscriber.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}

func TestOpenTelemetrySubscriberRejectsInvalidTransport(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, "http://localhost:4318/v1/traces")
	config.Transport = OpenTelemetryTransport("invalid")

	_, err := NewOpenTelemetrySubscriber(config)
	if err == nil {
		t.Fatal("expected invalid transport error")
	}
}

func TestOpenTelemetrySubscriberRejectsInvalidRequiredFields(t *testing.T) {
	testCases := []struct {
		name   string
		config OpenTelemetryConfig
	}{
		{
			name:   "missing type",
			config: NewOpenTelemetryConfig("", "http://localhost:4318/v1/traces"),
		},
		{
			name:   "unknown type",
			config: NewOpenTelemetryConfig(OpenTelemetryType("invalid"), "http://localhost:4318/v1/traces"),
		},
		{
			name:   "missing endpoint",
			config: NewOpenTelemetryConfig(OpenTelemetryTypeFull, ""),
		},
		{
			name:   "blank endpoint",
			config: NewOpenTelemetryConfig(OpenTelemetryTypeFull, " \t"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			subscriber, err := NewOpenTelemetrySubscriber(testCase.config)
			if err == nil {
				subscriber.Close()
				t.Fatal("expected required-field validation error")
			}
		})
	}
}

func TestOpenTelemetrySubscriberExportsScopeLifecycleAndMarks(t *testing.T) {
	type otelRequest struct {
		Path        string
		ContentType string
		Body        []byte
	}

	requests := make(chan otelRequest, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		requests <- otelRequest{
			Path:        r.URL.Path,
			ContentType: r.Header.Get("Content-Type"),
			Body:        body,
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, server.URL+"/v1/traces")
	config.ServiceName = "go-agent"
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber failed: %v", err)
	}
	defer subscriber.Close()

	name := "go_otel_e2e_" + time.Now().Format("150405.000000")
	if err := subscriber.Register(name); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope("otel_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		if err := EmitEvent(
			"otel_mark",
			WithEventParent(handle),
			WithEventData(json.RawMessage(`{"step":1}`)),
			WithEventMetadata(json.RawMessage(`{"source":"go"}`)),
		); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}
		if err := PopScope(handle); err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
	})
	if err := subscriber.ForceFlush(); err != nil {
		t.Fatalf("ForceFlush failed: %v", err)
	}

	select {
	case request := <-requests:
		if request.Path != "/v1/traces" {
			t.Fatalf("expected /v1/traces path, got %q", request.Path)
		}
		if request.ContentType != "application/x-protobuf" {
			t.Fatalf("expected protobuf content type, got %q", request.ContentType)
		}
		if len(request.Body) == 0 {
			t.Fatal("expected non-empty OTLP request body")
		}
		assertOtlpStringAttribute(t, request.Body, "nemo_relay.scope_type", "agent")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP request")
	}
}

func TestOpenTelemetrySubscriberExportsGenAIAgentProjection(t *testing.T) {
	requests := make(chan otelRequest, 1)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	config := NewOpenTelemetryConfig(OpenTelemetryTypeGenAI, server.URL+"/v1/traces")
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber failed: %v", err)
	}
	defer subscriber.Close()

	name := "go_gen_ai_e2e_" + time.Now().Format("150405.000000")
	if err := subscriber.Register(name); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope("research-agent", ScopeTypeAgent)
		requireNoError(t, err, "PushScope failed")
		requireNoError(t, PopScope(handle), "PopScope failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "ForceFlush failed")

	select {
	case request := <-requests:
		for _, needle := range [][]byte{
			[]byte("invoke_agent research-agent"),
			[]byte("gen_ai.operation.name"),
		} {
			if !bytes.Contains(request.Body, needle) {
				t.Fatalf("expected OTLP request body to contain %q", needle)
			}
		}
		if bytes.Contains(request.Body, []byte("nemo_relay.")) {
			t.Fatal("GenAI projection must not contain nemo_relay attributes")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP request")
	}
}
