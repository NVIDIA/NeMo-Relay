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
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	dir := t.TempDir()
	config := NewObservabilityConfig()
	atof := NewObservabilityAtofConfig()
	atof.Enabled = true
	atof.OutputDirectory = dir
	atof.Filename = "events.jsonl"
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
	atif.FilenameTemplate = "trajectory-{session_id}.json"
	config.Atif = &atif

	if report, err := ValidatePluginConfig(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	} else if len(report.Diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", report.Diagnostics)
	}
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
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
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}

	jsonl := string(mustReadFile(t, filepath.Join(dir, "events.jsonl")))
	if got := strings.Count(strings.TrimSpace(jsonl), "\n") + 1; got != 3 {
		t.Fatalf("expected 3 JSONL records, got %d: %s", got, jsonl)
	}

	trajectoryPath := filepath.Join(dir, "trajectory-"+handle.UUID()+".json")
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
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	dir := t.TempDir()
	config := NewObservabilityConfig()
	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.OutputDirectory = dir
	atif.FilenameTemplate = "trajectory-{session_id}.json"
	config.Atif = &atif

	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}

	first, err := PushScope("go-first-agent", ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":"first"}`)))
	if err != nil {
		t.Fatalf("PushScope first failed: %v", err)
	}
	if err := EmitEvent("go-first-mark", WithEventParent(first), WithEventData(json.RawMessage(`{"agent":"first"}`))); err != nil {
		t.Fatalf("EmitEvent first failed: %v", err)
	}
	nested, err := PushScope("go-nested-agent", ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":"nested"}`)))
	if err != nil {
		t.Fatalf("PushScope nested failed: %v", err)
	}
	if err := EmitEvent("go-nested-mark", WithEventParent(nested), WithEventData(json.RawMessage(`{"agent":"nested"}`))); err != nil {
		t.Fatalf("EmitEvent nested failed: %v", err)
	}
	if err := PopScope(nested, WithOutput(json.RawMessage(`{"done":true}`))); err != nil {
		t.Fatalf("PopScope nested failed: %v", err)
	}
	if err := PopScope(first, WithOutput(json.RawMessage(`{"done":true}`))); err != nil {
		t.Fatalf("PopScope first failed: %v", err)
	}

	second, err := PushScope("go-second-agent", ScopeTypeAgent, WithInput(json.RawMessage(`{"agent":"second"}`)))
	if err != nil {
		t.Fatalf("PushScope second failed: %v", err)
	}
	if err := EmitEvent("go-second-mark", WithEventParent(second), WithEventData(json.RawMessage(`{"agent":"second"}`))); err != nil {
		t.Fatalf("EmitEvent second failed: %v", err)
	}
	if err := PopScope(second, WithOutput(json.RawMessage(`{"done":true}`))); err != nil {
		t.Fatalf("PopScope second failed: %v", err)
	}
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}

	files, err := filepath.Glob(filepath.Join(dir, "trajectory-*.json"))
	if err != nil {
		t.Fatalf("Glob failed: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 ATIF trajectory files, got %d: %#v", len(files), files)
	}

	firstPayload := string(mustReadFile(t, filepath.Join(dir, "trajectory-"+first.UUID()+".json")))
	secondPayload := string(mustReadFile(t, filepath.Join(dir, "trajectory-"+second.UUID()+".json")))
	if !strings.Contains(firstPayload, "go-first-agent") || !strings.Contains(firstPayload, "go-nested-agent") {
		t.Fatalf("expected first trajectory to include first and nested agents: %s", firstPayload)
	}
	if strings.Contains(firstPayload, "go-second-agent") {
		t.Fatalf("first trajectory leaked second agent events: %s", firstPayload)
	}
	if !strings.Contains(secondPayload, "go-second-agent") {
		t.Fatalf("expected second trajectory to include second agent: %s", secondPayload)
	}
	if strings.Contains(secondPayload, "go-first-agent") || strings.Contains(secondPayload, "go-nested-agent") {
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
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	dir := t.TempDir()
	config := NewObservabilityConfig()
	atif := NewObservabilityAtifConfig()
	atif.Enabled = true
	atif.OutputDirectory = dir
	config.Atif = &atif
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{ObservabilityComponent(config)}}); err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}
	handle, err := PushScope("go-open-agent", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	path := filepath.Join(dir, "nemo-flow-atif-"+handle.UUID()+".json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected open-agent ATIF file at %s: %v", path, err)
	}
	if err := PopScope(handle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
}
