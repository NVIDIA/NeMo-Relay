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
	"os"
	"strings"
	"testing"
	"time"
)

const (
	newOpenTelemetrySubscriberFailed = "NewOpenTelemetrySubscriber failed: %v"
	otelRegisterFailed               = "Register failed: %v"
	otelTestEndpoint                 = "http://localhost:4318/v1/traces"
	otelTestPath                     = "/v1/traces"
	otelTimeFormat                   = "150405.000000"
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
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)

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
	if config.HeaderEnv == nil || len(config.HeaderEnv) != 0 {
		t.Fatalf("expected empty header environment map, got %#v", config.HeaderEnv)
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
	if config.PromoteMetadataPrefixes == nil || len(config.PromoteMetadataPrefixes) != 0 {
		t.Fatalf("expected empty metadata promotion prefixes, got %#v", config.PromoteMetadataPrefixes)
	}
}

func TestOpenTelemetrySubscriberAcceptsProjectionControls(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
	config.MarkProjection = MarkProjectionTool
	config.MarkExcludeNames = []string{"custom.mark"}
	config.AttributeMappings = []OtlpAttributeMapping{{
		Key:   "nemo_relay.model_name",
		Alias: "model.alias",
	}}
	config.PromoteMetadataPrefixes = []string{"nv."}

	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber with projection controls failed: %v", err)
	}
	defer subscriber.Close()
}

func TestOpenTelemetrySubscriberRejectsInvalidMetadataPromotionPrefix(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
	config.PromoteMetadataPrefixes = []string{"nv.*"}

	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected invalid metadata promotion prefix error")
	}
}

func TestOpenTelemetrySubscriberRejectsInvalidAttributeMappings(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
	config.AttributeMappings = []OtlpAttributeMapping{{Key: "", Alias: "model.alias"}}

	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected invalid attribute mapping error")
	}
}

func TestOpenTelemetrySubscriberLifecycle(t *testing.T) {
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
	config.ServiceName = "go-agent"
	config.ServiceNamespace = "agents"
	config.ServiceVersion = "1.0.0"
	config.InstrumentationScope = "go-tests"
	config.Timeout = 1250 * time.Millisecond
	config.Headers["authorization"] = "Bearer token"
	config.ResourceAttributes["deployment.environment"] = "test"
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf(newOpenTelemetrySubscriberFailed, err)
	}
	defer subscriber.Close()

	name := "go_otel_subscriber_" + time.Now().Format(otelTimeFormat)
	if err := subscriber.Register(name); err != nil {
		t.Fatalf(otelRegisterFailed, err)
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
	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
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
			config: NewOpenTelemetryConfig("", otelTestEndpoint),
		},
		{
			name:   "unknown type",
			config: NewOpenTelemetryConfig(OpenTelemetryType("invalid"), otelTestEndpoint),
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
		Path          string
		ContentType   string
		Authorization string
		Body          []byte
	}

	requests := make(chan otelRequest, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		requests <- otelRequest{
			Path:          r.URL.Path,
			ContentType:   r.Header.Get("Content-Type"),
			Authorization: r.Header.Get("Authorization"),
			Body:          body,
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, server.URL+otelTestPath)
	config.ServiceName = "go-agent"
	config.PromoteMetadataPrefixes = []string{"nv."}
	variable := "NEMO_RELAY_GO_HEADER_" + time.Now().Format(otelTimeFormat)
	secret := "Bearer go-activation-secret"
	t.Setenv(variable, secret)
	config.HeaderEnv["authorization"] = variable
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf(newOpenTelemetrySubscriberFailed, err)
	}
	defer subscriber.Close()
	if err := os.Setenv(variable, "Bearer go-changed-secret"); err != nil {
		t.Fatalf("change header environment: %v", err)
	}

	name := "go_otel_e2e_" + time.Now().Format(otelTimeFormat)
	if err := subscriber.Register(name); err != nil {
		t.Fatalf(otelRegisterFailed, err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope(
			"otel_scope",
			ScopeTypeAgent,
			WithMetadata(json.RawMessage(`{"nv.binding":"go"}`)),
		)
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
		if err := PopScope(
			handle,
			WithScopeEndMetadata(json.RawMessage(`{"nv.binding":"go"}`)),
		); err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
	})
	if err := subscriber.ForceFlush(); err != nil {
		t.Fatalf("ForceFlush failed: %v", err)
	}

	select {
	case request := <-requests:
		if request.Path != otelTestPath {
			t.Fatalf("expected /v1/traces path, got %q", request.Path)
		}
		if request.ContentType != "application/x-protobuf" {
			t.Fatalf("expected protobuf content type, got %q", request.ContentType)
		}
		if request.Authorization != secret {
			t.Fatalf("expected activation-time authorization header, got %q", request.Authorization)
		}
		if len(request.Body) == 0 {
			t.Fatal("expected non-empty OTLP request body")
		}
		if bytes.Contains(request.Body, []byte(secret)) {
			t.Fatal("authorization value must not appear in the OTLP payload")
		}
		assertOtlpStringAttribute(t, request.Body, "nemo_relay.scope_type", "agent")
		assertOtlpStringAttribute(t, request.Body, "nv.binding", "go")
		diagnostics, err := subscriber.RuntimeDiagnostics()
		if err != nil {
			t.Fatalf("RuntimeDiagnostics failed: %v", err)
		}
		for _, diagnostic := range diagnostics {
			if strings.Contains(diagnostic.Message, secret) {
				t.Fatal("authorization value must not appear in runtime diagnostics")
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP request")
	}
}

func TestOpenTelemetrySubscriberRejectsInvalidHeaderEnvWithoutSecretValues(t *testing.T) {
	variable := "NEMO_RELAY_GO_INVALID_HEADER_" + time.Now().Format(otelTimeFormat)
	secret := "relay-go-secret"

	config := NewOpenTelemetryConfig(OpenTelemetryTypeFull, otelTestEndpoint)
	config.HeaderEnv["authorization"] = variable
	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected unset header environment variable to fail")
	}

	t.Setenv(variable, "  ")
	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected blank header environment variable to fail")
	}

	if err := os.Setenv(variable, secret+"\ninvalid"); err != nil {
		t.Fatalf("set invalid header value: %v", err)
	}
	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected invalid header value to fail")
	} else if strings.Contains(err.Error(), secret) {
		t.Fatal("invalid header error exposed the environment-derived value")
	}

	if err := os.Setenv(variable, "valid"); err != nil {
		t.Fatalf("set valid header value: %v", err)
	}
	config.Headers["Authorization"] = "static"
	if _, err := NewOpenTelemetrySubscriber(config); err == nil {
		t.Fatal("expected case-insensitive header collision to fail")
	}
}

func TestOpenTelemetrySubscriberExportsGenAIAgentProjection(t *testing.T) {
	requests := make(chan otelRequest, 1)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	config := NewOpenTelemetryConfig(OpenTelemetryTypeGenAI, server.URL+otelTestPath)
	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf(newOpenTelemetrySubscriberFailed, err)
	}
	defer subscriber.Close()

	name := "go_gen_ai_e2e_" + time.Now().Format(otelTimeFormat)
	if err := subscriber.Register(name); err != nil {
		t.Fatalf(otelRegisterFailed, err)
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
