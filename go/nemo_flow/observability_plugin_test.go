// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	clearPluginConfigurationFailed = "ClearPluginConfiguration failed"
	initializePluginsFailed        = "InitializePlugins failed"
	trajectoryFilenamePrefix       = "trajectory-"
	firstAgentName                 = "go-first-agent"
	nestedAgentName                = "go-nested-agent"
	secondAgentName                = "go-second-agent"
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
	atif := NewObservabilityAtifConfig()
	if atif.Enabled || atif.AgentName != "NeMo Flow" || atif.ModelName != "unknown" || atif.FilenameTemplate != "nemo-flow-atif-{session_id}.json" {
		t.Fatalf("unexpected ATIF defaults: %#v", atif)
	}
	otlp := NewObservabilityOtlpConfig()
	if otlp.Enabled || otlp.Transport != "http_binary" || otlp.ServiceName != "nemo-flow" || otlp.TimeoutMillis != 3000 {
		t.Fatalf("unexpected OTLP defaults: %#v", otlp)
	}

	config.Atof = &atof
	wrapped := ObservabilityComponent(config)
	if wrapped.Kind != ObservabilityPluginKind || !wrapped.Enabled {
		t.Fatalf("unexpected component wrapper: %#v", wrapped)
	}
	if _, ok := wrapped.Config["atof"].(map[string]any); !ok {
		t.Fatalf("expected serialized ATOF config object, got %#v", wrapped.Config)
	}
}

func TestObservabilityPluginAtofAndAtifFiles(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("%s: %v", clearPluginConfigurationFailed, err)
	}
	dir := t.TempDir()
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
	atif.FilenameTemplate = trajectoryFilenamePrefix + "{session_id}.json"
	config.Atif = &atif

	if report, err := ValidatePluginConfig(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	} else if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("%s: %v", initializePluginsFailed, err)
	}

	handle, err := PushScope("go-observability-agent", ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":true}`)))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := EmitEvent("go-mark", WithEventParent(handle), WithEventData(json.RawMessage(`{"step":1}`))); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if err := PopScope(handle, WithOutput(json.RawMessage(`{"done":true}`))); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("%s: %v", clearPluginConfigurationFailed, err)
	}

	jsonl := string(mustReadFile(t, filepath.Join(dir, eventsJSONLFilename)))
	if got := strings.Count(strings.TrimSpace(jsonl), "\n") + 1; got != 3 {
		t.Fatalf("expected 3 JSONL records, got %d: %s", got, jsonl)
	}

	trajectoryPath := trajectoryFilePath(dir, handle)
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
	dir := t.TempDir()
	initializeAtifPlugin(t, dir)
	first := emitAgentStart(t, "first", firstAgentName)
	nested := emitAgentStart(t, "nested", nestedAgentName)
	emitAgentEnd(t, "nested", nested)
	emitAgentEnd(t, "first", first)
	second := emitAgentTrajectory(t, "second", secondAgentName)
	requireNoError(t, ClearPluginConfiguration(), clearPluginConfigurationFailed)

	files, err := filepath.Glob(filepath.Join(dir, trajectoryFilenamePrefix+"*.json"))
	if err != nil {
		t.Fatalf("Glob failed: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 ATIF trajectory files, got %d: %#v", len(files), files)
	}

	firstPayload := string(mustReadFile(t, trajectoryFilePath(dir, first)))
	secondPayload := string(mustReadFile(t, trajectoryFilePath(dir, second)))
	if !strings.Contains(firstPayload, firstAgentName) || !strings.Contains(firstPayload, nestedAgentName) {
		t.Fatalf("expected first trajectory to include first and nested agents: %s", firstPayload)
	}
	if strings.Contains(firstPayload, secondAgentName) {
		t.Fatalf("first trajectory leaked second agent events: %s", firstPayload)
	}
	if !strings.Contains(secondPayload, secondAgentName) {
		t.Fatalf("expected second trajectory to include second agent: %s", secondPayload)
	}
	if strings.Contains(secondPayload, firstAgentName) || strings.Contains(secondPayload, nestedAgentName) {
		t.Fatalf("second trajectory leaked first trajectory events: %s", secondPayload)
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
		t.Fatalf("%s: %v", clearPluginConfigurationFailed, err)
	}
	dir := t.TempDir()
	config := NewObservabilityConfig()
	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.OutputDirectory = dir
	config.Atif = &atif
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("%s: %v", initializePluginsFailed, err)
	}
	handle, err := PushScope("go-open-agent", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("%s: %v", clearPluginConfigurationFailed, err)
	}
	path := filepath.Join(dir, "nemo-flow-atif-"+handle.UUID()+".json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected open-agent ATIF file at %s: %v", path, err)
	}
	if err := PopScope(handle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
}

func initializeAtifPlugin(t *testing.T, dir string) {
	t.Helper()
	t.Cleanup(func() {
		requireNoError(t, ClearPluginConfiguration(), clearPluginConfigurationFailed)
	})
	requireNoError(t, ClearPluginConfiguration(), clearPluginConfigurationFailed)

	config := NewObservabilityConfig()
	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.OutputDirectory = dir
	atif.FilenameTemplate = trajectoryFilenamePrefix + "{session_id}.json"
	config.Atif = &atif

	_, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}})
	requireNoError(t, err, initializePluginsFailed)
}

func emitAgentTrajectory(t *testing.T, label string, name string) *ScopeHandle {
	t.Helper()
	handle := emitAgentStart(t, label, name)
	emitAgentEnd(t, label, handle)
	return handle
}

func emitAgentStart(t *testing.T, label string, name string) *ScopeHandle {
	t.Helper()
	handle, err := PushScope(name, ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":"`+label+`"}`)))
	requireNoError(t, err, "PushScope "+label+" failed")
	requireNoError(t, EmitEvent("go-"+label+"-mark", WithEventParent(handle), WithEventData(json.RawMessage(`{"agent":"`+label+`"}`))), "EmitEvent "+label+" failed")
	return handle
}

func emitAgentEnd(t *testing.T, label string, handle *ScopeHandle) {
	t.Helper()
	requireNoError(t, PopScope(handle, WithOutput(json.RawMessage(`{"done":true}`))), "PopScope "+label+" failed")
}

func trajectoryFilePath(dir string, handle *ScopeHandle) string {
	return filepath.Join(dir, trajectoryFilenamePrefix+handle.UUID()+".json")
}
