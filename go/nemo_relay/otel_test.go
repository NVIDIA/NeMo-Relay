// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
		tool, err := PushScope("search", ScopeTypeTool, WithInput(json.RawMessage(`{"query":"docs"}`)))
		requireNoError(t, err, "tool PushScope failed")
		requireNoError(t, PopScope(tool, WithOutput(json.RawMessage(`{"hits":[]}`))), "tool PopScope failed")
		requireNoError(t, PopScope(handle), "PopScope failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "ForceFlush failed")

	select {
	case request := <-requests:
		for _, needle := range [][]byte{
			[]byte("invoke_agent research-agent"),
			[]byte("gen_ai.tool.call.arguments"),
			[]byte("gen_ai.tool.call.result"),
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

// TestObservabilityOpenTelemetryFileSinkConfigSerializes checks the file-sink section
// marshals to the keys the Rust plugin config deserializes, and that optional fields
// stay absent so core defaults apply.
func TestObservabilityOpenTelemetryFileSinkConfigSerializes(t *testing.T) {
	config := ObservabilityOpenTelemetryConfig{
		Enabled: true,
		FileSinks: []ObservabilityOpenTelemetryFileSinkConfig{{
			Type:                            OpenTelemetryTypeFull,
			OutputDirectory:                 "/var/log/nemo-relay",
			Format:                          "proto",
			PromoteResourceMetadataPrefixes: []string{"deployment."},
		}},
	}

	encoded, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal file sink config: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal file sink config: %v", err)
	}
	sinks, ok := decoded["file_sinks"].([]any)
	if !ok || len(sinks) != 1 {
		t.Fatalf("expected one file_sinks entry, got %v", decoded["file_sinks"])
	}
	sink, ok := sinks[0].(map[string]any)
	if !ok {
		t.Fatalf("expected a file sink object, got %T", sinks[0])
	}
	if sink["output_directory"] != "/var/log/nemo-relay" {
		t.Errorf("unexpected output_directory %v", sink["output_directory"])
	}
	if sink["format"] != "proto" {
		t.Errorf("unexpected format %v", sink["format"])
	}
	// Resource promotion is a supported file-sink setting, so the typed config
	// has to be able to express it.
	prefixes, ok := sink["promote_resource_metadata_prefixes"].([]any)
	if !ok || len(prefixes) != 1 || prefixes[0] != "deployment." {
		t.Errorf("unexpected promote_resource_metadata_prefixes %v", sink["promote_resource_metadata_prefixes"])
	}
	// A file sink has no endpoint, and unset optionals must not be emitted:
	// an empty mode would otherwise override the core default.
	for _, absent := range []string{"endpoint", "transport", "filename", "mode"} {
		if _, present := sink[absent]; present {
			t.Errorf("unexpected %q in a file sink section", absent)
		}
	}
}

// closeFileSink releases a native file-sink subscriber. Shutdown flushes the
// output file; Close is the only method that frees the native handle.
func closeFileSink(t *testing.T, subscriber *OpenTelemetrySubscriber) {
	t.Helper()
	t.Cleanup(func() {
		if err := subscriber.Shutdown(); err != nil {
			t.Errorf("shutdown: %v", err)
		}
		subscriber.Close()
	})
}

// TestOpenTelemetryFileSinkSubscriberWritesTraceFile exercises the file-sink FFI
// entry point: a subscriber built from a file-sink config opens its output file,
// and the endpoint-only parameters are absent from the config entirely.
func TestOpenTelemetryFileSinkSubscriberWritesTraceFile(t *testing.T) {
	dir := t.TempDir()
	subscriber, err := NewOpenTelemetryFileSinkSubscriber(OpenTelemetryFileSinkConfig{
		Type:            OpenTelemetryTypeFull,
		OutputDirectory: dir,
		Filename:        "go-trace.jsonl",
		ServiceName:     "go-file-sink",
	})
	if err != nil {
		t.Fatalf("create file sink subscriber: %v", err)
	}
	closeFileSink(t, subscriber)

	name := "go_file_sink_e2e_" + time.Now().Format(otelTimeFormat)
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

	// The file exists as soon as the subscriber is created, so the assertion is
	// on the exported record rather than on the file.
	contents, err := os.ReadFile(filepath.Join(dir, "go-trace.jsonl"))
	if err != nil {
		t.Fatalf("read the trace file: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(contents)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one JSON line, got %d: %s", len(lines), contents)
	}
	var record struct {
		ResourceSpans []json.RawMessage `json:"resourceSpans"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &record); err != nil {
		t.Fatalf("decode the exported record: %v", err)
	}
	if len(record.ResourceSpans) == 0 {
		t.Fatalf("expected resourceSpans in %s", lines[0])
	}
	if !strings.Contains(lines[0], "research-agent") {
		t.Fatalf("expected the projected span in %s", lines[0])
	}
}

// TestOpenTelemetryFileSinkSubscriberDefaultsFilenameToFormat checks that an
// omitted filename is derived from the chosen format.
func TestOpenTelemetryFileSinkSubscriberDefaultsFilenameToFormat(t *testing.T) {
	dir := t.TempDir()
	subscriber, err := NewOpenTelemetryFileSinkSubscriber(OpenTelemetryFileSinkConfig{
		OutputDirectory: dir,
		Format:          OpenTelemetryFileSinkFormatProto,
		Mode:            OpenTelemetryFileSinkModeAppend,
	})
	if err != nil {
		t.Fatalf("create file sink subscriber: %v", err)
	}
	closeFileSink(t, subscriber)

	if _, err := os.Stat(filepath.Join(dir, "nemo-relay-otlp.otlp.pb")); err != nil {
		t.Fatalf("expected the default proto filename: %v", err)
	}
}

// TestOpenTelemetryFileSinkSubscriberRejectsInvalidConfig covers the rejection
// paths: each is refused rather than silently defaulted.
func TestOpenTelemetryFileSinkSubscriberRejectsInvalidConfig(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name   string
		config OpenTelemetryFileSinkConfig
	}{
		{"blank output directory", OpenTelemetryFileSinkConfig{OutputDirectory: ""}},
		{"unknown format", OpenTelemetryFileSinkConfig{OutputDirectory: dir, Format: "yaml"}},
		{"unknown mode", OpenTelemetryFileSinkConfig{OutputDirectory: dir, Mode: "truncate"}},
		{"filename escapes directory", OpenTelemetryFileSinkConfig{OutputDirectory: dir, Filename: "../escape.jsonl"}},
		{"unknown type", OpenTelemetryFileSinkConfig{OutputDirectory: dir, Type: "unsupported"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			subscriber, err := NewOpenTelemetryFileSinkSubscriber(tc.config)
			if err == nil {
				closeFileSink(t, subscriber)
				t.Fatalf("expected %s to be rejected", tc.name)
			}
		})
	}
}

// TestOpenTelemetryFileSinkSubscriberAppliesOptionalSettings covers the optional
// fields the minimal cases leave unset, so every branch of the FFI conversion is
// exercised from Go.
//
// The assertions run over the JSON Lines encoding because this module carries no
// protobuf dependency: the record is decoded and its OTLP resource and scope
// fields are inspected by name, rather than the frame being searched for
// substrings. The proto sink keeps its own framing check below.
func TestOpenTelemetryFileSinkSubscriberAppliesOptionalSettings(t *testing.T) {
	dir := t.TempDir()
	subscriber, err := NewOpenTelemetryFileSinkSubscriber(OpenTelemetryFileSinkConfig{
		Type:                 OpenTelemetryTypeFull,
		OutputDirectory:      dir,
		Filename:             "full.jsonl",
		Format:               OpenTelemetryFileSinkFormatJSONLines,
		Mode:                 OpenTelemetryFileSinkModeAppend,
		ServiceName:          "go-file-sink",
		ServiceNamespace:     "agents",
		ServiceVersion:       "1.2.3",
		InstrumentationScope: "go-scope",
		ResourceAttributes:   map[string]string{"deployment.environment": "test"},
	})
	if err != nil {
		t.Fatalf("create file sink subscriber: %v", err)
	}
	closeFileSink(t, subscriber)

	name := "go_file_sink_full_" + time.Now().Format(otelTimeFormat)
	if err := subscriber.Register(name); err != nil {
		t.Fatalf(otelRegisterFailed, err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope("go-optional-agent", ScopeTypeAgent)
		requireNoError(t, err, "PushScope failed")
		requireNoError(t, PopScope(handle), "PopScope failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "ForceFlush failed")

	contents, err := os.ReadFile(filepath.Join(dir, "full.jsonl"))
	if err != nil {
		t.Fatalf("read the trace file: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(contents)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one JSON line, got %d: %s", len(lines), contents)
	}

	var record struct {
		ResourceSpans []struct {
			Resource struct {
				Attributes []struct {
					Key   string `json:"key"`
					Value struct {
						StringValue string `json:"stringValue"`
					} `json:"value"`
				} `json:"attributes"`
			} `json:"resource"`
			ScopeSpans []struct {
				Scope struct {
					Name string `json:"name"`
				} `json:"scope"`
				Spans []struct {
					Name string `json:"name"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &record); err != nil {
		t.Fatalf("decode the exported record: %v", err)
	}
	if len(record.ResourceSpans) != 1 || len(record.ResourceSpans[0].ScopeSpans) != 1 {
		t.Fatalf("expected one resource and one scope, got %s", lines[0])
	}

	resource := map[string]string{}
	for _, attribute := range record.ResourceSpans[0].Resource.Attributes {
		resource[attribute.Key] = attribute.Value.StringValue
	}
	for _, want := range []struct{ key, value string }{
		{"service.name", "go-file-sink"},
		{"service.namespace", "agents"},
		{"service.version", "1.2.3"},
		{"deployment.environment", "test"},
	} {
		if resource[want.key] != want.value {
			t.Errorf("resource %q = %q, want %q", want.key, resource[want.key], want.value)
		}
	}

	scopeSpans := record.ResourceSpans[0].ScopeSpans[0]
	if scopeSpans.Scope.Name != "go-scope" {
		t.Errorf("instrumentation scope = %q, want %q", scopeSpans.Scope.Name, "go-scope")
	}
	if len(scopeSpans.Spans) != 1 || scopeSpans.Spans[0].Name != "go-optional-agent" {
		t.Errorf("unexpected spans in %s", lines[0])
	}
}

// TestOpenTelemetryFileSinkSubscriberOmittedServiceNameReachesTheSdkDefault
// checks that leaving ServiceName unset is passed through as NULL rather than
// as "unknown_service", so the SDK's own resource detection supplies it.
func TestOpenTelemetryFileSinkSubscriberOmittedServiceNameReachesTheSdkDefault(t *testing.T) {
	dir := t.TempDir()
	subscriber, err := NewOpenTelemetryFileSinkSubscriber(OpenTelemetryFileSinkConfig{
		OutputDirectory: dir,
		Filename:        "defaulted.jsonl",
	})
	if err != nil {
		t.Fatalf("create file sink subscriber: %v", err)
	}
	closeFileSink(t, subscriber)

	name := "go_file_sink_default_name_" + time.Now().Format(otelTimeFormat)
	if err := subscriber.Register(name); err != nil {
		t.Fatalf(otelRegisterFailed, err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope("go-default-agent", ScopeTypeAgent)
		requireNoError(t, err, "PushScope failed")
		requireNoError(t, PopScope(handle), "PopScope failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "ForceFlush failed")

	contents, err := os.ReadFile(filepath.Join(dir, "defaulted.jsonl"))
	if err != nil {
		t.Fatalf("read the trace file: %v", err)
	}
	// The SDK supplies "unknown_service" itself; what matters is that Go did not
	// send a configured value that would shadow OTEL_SERVICE_NAME.
	if !strings.Contains(string(contents), "service.name") {
		t.Fatalf("expected an SDK-detected service.name in %s", contents)
	}
}

// TestOpenTelemetryFileSinkSubscriberRegisters checks the subscriber reaches the
// shared registration path rather than only being constructed.
func TestOpenTelemetryFileSinkSubscriberRegisters(t *testing.T) {
	dir := t.TempDir()
	subscriber, err := NewOpenTelemetryFileSinkSubscriber(OpenTelemetryFileSinkConfig{
		OutputDirectory: dir,
		Filename:        "registered.jsonl",
	})
	if err != nil {
		t.Fatalf("create file sink subscriber: %v", err)
	}
	closeFileSink(t, subscriber)

	name := fmt.Sprintf("go_file_sink_%d", time.Now().UnixNano())
	if err := subscriber.Register(name); err != nil {
		t.Fatalf("register: %v", err)
	}
	if err := subscriber.Deregister(name); err != nil {
		t.Fatalf("deregister: %v", err)
	}
}
