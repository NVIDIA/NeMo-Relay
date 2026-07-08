// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	ClearPluginConfigurationFailed = "ClearPluginConfiguration failed"
	InitializePluginsFailed        = "InitializePlugins failed"
	TrajectoryFilenamePrefix       = "trajectory-"
	FirstAgentName                 = "go-first-agent"
	NestedAgentName                = "go-nested-agent"
	SecondAgentName                = "go-second-agent"
	testAccessKeyID                = "test-access-key"
	testAtifEndpoint               = "https://example.com/atif"
	testRegion                     = "us-west-2"
	testStaticHeader               = "x-static"
	fatalErrorFormat               = "%s: %v"
	failedSuffix                   = " failed"
)

func TestObservabilityConfigHelpers(t *testing.T) {
	config := NewObservabilityConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}
	atof := NewObservabilityAtofConfig()
	if atof.Enabled || atof.Mode != "append" {
		t.Fatalf("unexpected ATOF defaults: %#v", atof)
	}
	atof.Endpoints = []ObservabilityAtofEndpoint{{
		URL:             "http://localhost:8080/events",
		Transport:       "http_post",
		Headers:         map[string]string{"X-Test": "yes"},
		TimeoutMillis:   1000,
		FieldNamePolicy: "replace_dots",
	}}
	atif := NewObservabilityAtifConfig()
	if atif.Enabled || atif.AgentName != "NeMo Relay" || atif.ModelName != "unknown" || atif.FilenameTemplate != "nemo-relay-atif-{session_id}.json" {
		t.Fatalf("unexpected ATIF defaults: %#v", atif)
	}
	allowHTTP := false
	s3Storage := NewObservabilityS3StorageConfig("archive")
	s3Storage.KeyPrefix = "runs/"
	s3Storage.AccessKeyID = testAccessKeyID
	s3Storage.SecretAccessKeyVar = "NEMO_RELAY_TEST_SECRET"
	s3Storage.Region = testRegion
	s3Storage.AllowHTTP = &allowHTTP
	httpStorage := NewObservabilityHttpStorageConfig(testAtifEndpoint)
	httpStorage.Headers = map[string]string{testStaticHeader: "value"}
	httpStorage.HeaderEnv = map[string]string{"authorization": "NEMO_RELAY_ATIF_HTTP_AUTH"}
	httpStorage.TimeoutMillis = 1500
	assertS3StorageConfig(t, s3Storage)
	assertHTTPStorageConfig(t, httpStorage)
	atif.Storage = []ObservabilityAtifStorageConfigurer{
		s3Storage,
		httpStorage,
	}
	otlp := NewObservabilityOtlpConfig()
	if otlp.Enabled || otlp.MarkProjection != ObservabilityMarkProjectionInherit || len(otlp.MarkExcludeNames) != 1 || otlp.MarkExcludeNames[0] != "llm.chunk" || otlp.Transport != "http_binary" || otlp.ServiceName != "nemo-relay" || otlp.TimeoutMillis != 3000 {
		t.Fatalf("unexpected OTLP defaults: %#v", otlp)
	}
	otlp.MarkProjection = ObservabilityMarkProjectionTool

	config.Atof = &atof
	config.Atif = &atif
	config.OpenTelemetry = &otlp
	wrapped := ObservabilityComponent(config)
	if wrapped.Kind != ObservabilityPluginKind || !wrapped.Enabled {
		t.Fatalf("unexpected component wrapper: %#v", wrapped)
	}
	if _, ok := wrapped.Config["atof"].(map[string]any); !ok {
		t.Fatalf("expected serialized ATOF config object, got %#v", wrapped.Config)
	}
	atofConfig := wrapped.Config["atof"].(map[string]any)
	endpoints, ok := atofConfig["endpoints"].([]any)
	if !ok {
		t.Fatalf("expected serialized ATOF endpoints, got %#v", atofConfig)
	}
	firstEndpoint, ok := endpoints[0].(map[string]any)
	if !ok || firstEndpoint["field_name_policy"] != "replace_dots" {
		t.Fatalf("expected serialized ATOF endpoint field name policy, got %#v", endpoints)
	}
	serialized, err := json.Marshal(wrapped)
	if err != nil {
		t.Fatalf("marshal observability component failed: %v", err)
	}
	if !strings.Contains(string(serialized), `"field_name_policy":"replace_dots"`) {
		t.Fatalf("expected field_name_policy in serialized component, got %s", serialized)
	}
	assertWrappedAtifStorageConfig(t, wrapped.Config["atif"].(map[string]any))
	if wrapped.Config["opentelemetry"].(map[string]any)["mark_projection"] != "tool" {
		t.Fatalf("expected tool mark projection in serialized config: %#v", wrapped.Config)
	}
}

func assertS3StorageConfig(t *testing.T, storage ObservabilityS3StorageConfig) {
	t.Helper()
	if storage.Bucket != "archive" ||
		storage.KeyPrefix != "runs/" ||
		storage.AccessKeyID != testAccessKeyID ||
		storage.SecretAccessKeyVar != "NEMO_RELAY_TEST_SECRET" ||
		storage.Region != testRegion ||
		storage.AllowHTTP == nil ||
		*storage.AllowHTTP {
		t.Fatalf("unexpected S3 constructor values: %#v", storage)
	}

	serialized := marshalStorageConfig(t, storage)
	if serialized["type"] != "s3" ||
		serialized["bucket"] != "archive" ||
		serialized["key_prefix"] != "runs/" ||
		serialized["access_key_id"] != testAccessKeyID ||
		serialized["secret_access_key_var"] != "NEMO_RELAY_TEST_SECRET" ||
		serialized["region"] != testRegion ||
		serialized["allow_http"] != false {
		t.Fatalf("unexpected serialized S3 storage config: %#v", serialized)
	}
}

func assertHTTPStorageConfig(t *testing.T, storage ObservabilityHttpStorageConfig) {
	t.Helper()
	if storage.Endpoint != testAtifEndpoint ||
		storage.Headers[testStaticHeader] != "value" ||
		storage.HeaderEnv["authorization"] != "NEMO_RELAY_ATIF_HTTP_AUTH" ||
		storage.TimeoutMillis != 1500 {
		t.Fatalf("unexpected HTTP constructor values: %#v", storage)
	}

	serialized := marshalStorageConfig(t, storage)
	headers := serialized["headers"].(map[string]any)
	headerEnv := serialized["header_env"].(map[string]any)
	if serialized["type"] != "http" ||
		serialized["endpoint"] != testAtifEndpoint ||
		serialized["timeout_millis"] != float64(1500) ||
		headers[testStaticHeader] != "value" ||
		headerEnv["authorization"] != "NEMO_RELAY_ATIF_HTTP_AUTH" {
		t.Fatalf("unexpected serialized HTTP storage config: %#v", serialized)
	}
}

func assertWrappedAtifStorageConfig(t *testing.T, atifConfig map[string]any) {
	t.Helper()
	storage := atifConfig["storage"].([]any)
	if len(storage) != 2 {
		t.Fatalf("expected two ATIF storage destinations, got %#v", storage)
	}
	s3 := storage[0].(map[string]any)
	if s3["type"] != "s3" || s3["bucket"] != "archive" || s3["key_prefix"] != "runs/" || s3["allow_http"] != false {
		t.Fatalf("unexpected S3 storage config: %#v", s3)
	}
	http := storage[1].(map[string]any)
	if http["type"] != "http" || http["endpoint"] != testAtifEndpoint || http["timeout_millis"] != float64(1500) {
		t.Fatalf("unexpected HTTP storage config: %#v", http)
	}
}

func marshalStorageConfig(t *testing.T, config ObservabilityAtifStorageConfigurer) map[string]any {
	t.Helper()
	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal storage config: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		t.Fatalf("unmarshal storage config: %v", err)
	}
	return parsed
}

func TestObservabilityPluginAtofAndAtifFiles(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf(fatalErrorFormat, ClearPluginConfigurationFailed, err)
	}
	t.Cleanup(func() {
		requireNoError(t, ClearPluginConfiguration(), ClearPluginConfigurationFailed)
	})
	dir := t.TempDir()
	config := NewAtofAndAtifTestConfig(dir)
	pluginConfig := PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}

	if report, err := ValidatePluginConfig(pluginConfig); err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	} else if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
	if _, err := InitializePlugins(pluginConfig); err != nil {
		t.Fatalf(fatalErrorFormat, InitializePluginsFailed, err)
	}

	handle := EmitObservabilityTestTrajectory(t)
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf(fatalErrorFormat, ClearPluginConfigurationFailed, err)
	}

	AssertAtofRecordCount(t, filepath.Join(dir, eventsJSONLFilename), 3)
	AssertAtifAgentMetadata(t, TrajectoryFilePath(dir, handle))
}

func NewAtofAndAtifTestConfig(dir string) ObservabilityConfig {
	config := NewObservabilityConfig()
	atof := NewObservabilityAtofConfig()
	atof.Enabled = true
	atof.OutputDirectory = dir
	atof.Filename = eventsJSONLFilename
	atof.Mode = "overwrite"
	config.Atof = &atof

	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.AgentName = "go-agent"
	atif.AgentVersion = "1.2.3"
	atif.ModelName = "go-model"
	atif.ToolDefinitions = []map[string]any{{"name": "search"}}
	atif.Extra = map[string]any{"binding": "go"}
	atif.OutputDirectory = dir
	atif.FilenameTemplate = TrajectoryFilenamePrefix + "{session_id}.json"
	config.Atif = &atif
	return config
}

func EmitObservabilityTestTrajectory(t *testing.T) *ScopeHandle {
	t.Helper()
	var handle *ScopeHandle
	runWithTestScopeStack(t, func() {
		var err error
		handle, err = PushScope("go-observability-agent", ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":true}`)))
		requireNoError(t, err, "PushScope failed")
		requireNoError(t, EmitEvent("go-mark", WithEventParent(handle), WithEventData(json.RawMessage(`{"step":1}`))), "EmitEvent failed")
		requireNoError(t, PopScope(handle, WithOutput(json.RawMessage(`{"done":true}`))), "PopScope failed")
	})
	return handle
}

func AssertAtofRecordCount(t *testing.T, path string, want int) {
	t.Helper()
	jsonl := string(mustReadFile(t, path))
	if got := strings.Count(strings.TrimSpace(jsonl), "\n") + 1; got != want {
		t.Fatalf("expected %d JSONL records, got %d: %s", want, got, jsonl)
	}
}

func AssertAtifAgentMetadata(t *testing.T, trajectoryPath string) {
	t.Helper()
	var trajectory map[string]any
	if err := json.Unmarshal(mustReadFile(t, trajectoryPath), &trajectory); err != nil {
		t.Fatalf("failed to read trajectory: %v", err)
	}
	agent := trajectory["agent"].(map[string]any)
	if agent["name"] != "go-agent" || agent["version"] != "1.2.3" || agent["model_name"] != "go-model" {
		t.Fatalf("unexpected ATIF agent metadata: %#v", agent)
	}
	if !strings.Contains(string(mustReadFile(t, trajectoryPath)), "go-observability-agent") {
		t.Fatalf("expected top-level agent event in ATIF file")
	}
}

func TestObservabilityPluginAtifSplitsMultipleTopLevelAgents(t *testing.T) {
	Dir := t.TempDir()
	InitializeAtifPlugin(t, Dir)
	var First, Nested, Second *ScopeHandle
	runWithTestScopeStack(t, func() {
		First = EmitAgentStart(t, "first", FirstAgentName)
		Nested = EmitAgentStart(t, "nested", NestedAgentName)
		EmitAgentEnd(t, "nested", Nested)
		EmitAgentEnd(t, "first", First)
		Second = EmitAgentTrajectory(t, "second", SecondAgentName)
	})
	requireNoError(t, ClearPluginConfiguration(), ClearPluginConfigurationFailed)

	Files, err := filepath.Glob(filepath.Join(Dir, TrajectoryFilenamePrefix+"*.json"))
	if err != nil {
		t.Fatalf("Glob failed: %v", err)
	}
	if len(Files) != 2 {
		t.Fatalf("expected 2 ATIF trajectory files, got %d: %#v", len(Files), Files)
	}

	FirstPayload := string(mustReadFile(t, TrajectoryFilePath(Dir, First)))
	SecondPayload := string(mustReadFile(t, TrajectoryFilePath(Dir, Second)))
	if !strings.Contains(FirstPayload, FirstAgentName) || !strings.Contains(FirstPayload, NestedAgentName) {
		t.Fatalf("expected first trajectory to include first and nested agents: %s", FirstPayload)
	}
	if strings.Contains(FirstPayload, SecondAgentName) {
		t.Fatalf("first trajectory leaked second agent events: %s", FirstPayload)
	}
	if !strings.Contains(SecondPayload, SecondAgentName) {
		t.Fatalf("expected second trajectory to include second agent: %s", SecondPayload)
	}
	if strings.Contains(SecondPayload, FirstAgentName) || strings.Contains(SecondPayload, NestedAgentName) {
		t.Fatalf("second trajectory leaked first trajectory events: %s", SecondPayload)
	}
}

func TestObservabilityPluginValidationRejectsBadValues(t *testing.T) {
	config := NewObservabilityConfig()
	atof := NewObservabilityAtofConfig()
	atof.Mode = "bad"
	config.Atof = &atof
	atif := NewObservabilityAtifConfig()
	atif.FilenameTemplate = "missing-placeholder.json"
	config.Atif = &atif

	report, err := ValidatePluginConfig(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}})
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(report.Diagnostics) < 2 {
		t.Fatalf("expected validation diagnostics, got %#v", report.Diagnostics)
	}
}

func TestObservabilityPluginListKindIsAutomatic(t *testing.T) {
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds failed: %v", err)
	}
	for _, kind := range kinds {
		if kind == ObservabilityPluginKind {
			return
		}
	}
	t.Fatalf("expected %q in registered kinds: %#v", ObservabilityPluginKind, kinds)
}

func TestObservabilityAtifOpenAgentFlushesOnClear(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf(fatalErrorFormat, ClearPluginConfigurationFailed, err)
	}
	t.Cleanup(func() {
		requireNoError(t, ClearPluginConfiguration(), ClearPluginConfigurationFailed)
	})
	dir := t.TempDir()
	config := NewObservabilityConfig()
	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.OutputDirectory = dir
	config.Atif = &atif
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf(fatalErrorFormat, InitializePluginsFailed, err)
	}
	var handle *ScopeHandle
	runWithTestScopeStack(t, func() {
		var err error
		handle, err = PushScope("go-open-agent", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		if err := ClearPluginConfiguration(); err != nil {
			t.Fatalf(fatalErrorFormat, ClearPluginConfigurationFailed, err)
		}
		if err := PopScope(handle); err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
	})
	path := filepath.Join(dir, "nemo-relay-atif-"+handle.UUID()+".json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected open-agent ATIF file at %s: %v", path, err)
	}
}

func InitializeAtifPlugin(t *testing.T, Dir string) {
	t.Helper()
	t.Cleanup(func() {
		requireNoError(t, ClearPluginConfiguration(), ClearPluginConfigurationFailed)
	})
	requireNoError(t, ClearPluginConfiguration(), ClearPluginConfigurationFailed)

	Config := NewObservabilityConfig()
	Atif := NewObservabilityAtifConfig()
	Atif.Enabled = true
	Atif.OutputDirectory = Dir
	Atif.FilenameTemplate = TrajectoryFilenamePrefix + "{session_id}.json"
	Config.Atif = &Atif

	_, Err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(Config)}})
	requireNoError(t, Err, InitializePluginsFailed)
}

func EmitAgentTrajectory(t *testing.T, Label string, Name string) *ScopeHandle {
	t.Helper()
	Handle := EmitAgentStart(t, Label, Name)
	EmitAgentEnd(t, Label, Handle)
	return Handle
}

func EmitAgentStart(t *testing.T, Label string, Name string) *ScopeHandle {
	t.Helper()
	Handle, Err := PushScope(Name, ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":"`+Label+`"}`)))
	requireNoError(t, Err, "PushScope "+Label+failedSuffix)
	requireNoError(t, EmitEvent("go-"+Label+"-mark", WithEventParent(Handle), WithEventData(json.RawMessage(`{"agent":"`+Label+`"}`))), "EmitEvent "+Label+failedSuffix)
	return Handle
}

func EmitAgentEnd(t *testing.T, Label string, Handle *ScopeHandle) {
	t.Helper()
	requireNoError(t, PopScope(Handle, WithOutput(json.RawMessage(`{"done":true}`))), "PopScope "+Label+failedSuffix)
}

func TrajectoryFilePath(Dir string, Handle *ScopeHandle) string {
	return filepath.Join(Dir, TrajectoryFilenamePrefix+Handle.UUID()+".json")
}
