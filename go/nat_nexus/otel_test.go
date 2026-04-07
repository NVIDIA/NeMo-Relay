// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewOpenTelemetryConfigDefaults(t *testing.T) {
	config := NewOpenTelemetryConfig()

	if config.Transport != OpenTelemetryTransportHTTPBinary {
		t.Fatalf("expected default transport http_binary, got %q", config.Transport)
	}
	if config.ServiceName != "nat-nexus" {
		t.Fatalf("expected default service name nat-nexus, got %q", config.ServiceName)
	}
	if config.InstrumentationScope != "nvidia-nat-nexus-otel" {
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
}

func TestOpenTelemetrySubscriberLifecycle(t *testing.T) {
	config := NewOpenTelemetryConfig()
	config.Endpoint = "http://localhost:4318/v1/traces"
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
	config := NewOpenTelemetryConfig()
	config.Transport = OpenTelemetryTransport("invalid")

	_, err := NewOpenTelemetrySubscriber(config)
	if err == nil {
		t.Fatal("expected invalid transport error")
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

	config := NewOpenTelemetryConfig()
	config.Endpoint = server.URL + "/v1/traces"
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
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP request")
	}
}
