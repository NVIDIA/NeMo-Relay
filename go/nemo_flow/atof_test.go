// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewAtofExporterConfigDefaults(t *testing.T) {
	config := NewAtofExporterConfig()

	if config.Mode != AtofExporterModeAppend {
		t.Fatalf("expected append mode default, got %q", config.Mode)
	}
	if config.OutputDirectory != "" {
		t.Fatalf("expected empty output directory default override, got %q", config.OutputDirectory)
	}
	if config.Filename != "" {
		t.Fatalf("expected empty filename default override, got %q", config.Filename)
	}
}

func TestAtofExporterLifecycleWritesRawJSONL(t *testing.T) {
	dir := t.TempDir()
	exporter, err := NewAtofExporter(AtofExporterConfig{
		OutputDirectory: dir,
		Mode:            AtofExporterModeOverwrite,
		Filename:        "events.jsonl",
	})
	if err != nil {
		t.Fatalf("NewAtofExporter failed: %v", err)
	}
	defer exporter.Close()

	path, err := exporter.Path()
	if err != nil {
		t.Fatalf("Path failed: %v", err)
	}
	if filepath.Base(path) != "events.jsonl" {
		t.Fatalf("expected events.jsonl path, got %q", path)
	}

	name := "go_atof_" + time.Now().Format("150405.000000")
	if err := exporter.Register(name); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	handle, err := PushScope("atof_scope", ScopeTypeAgent, WithInput(json.RawMessage(`{"scope":true}`)))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := EmitEvent("atof_mark", WithEventParent(handle), WithEventData(json.RawMessage(`{"step":1}`))); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if err := PopScope(handle, WithOutput(json.RawMessage(`{"done":true}`))); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := exporter.Deregister(name); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}
	if err := exporter.Deregister(name); err != nil {
		t.Fatalf("repeated Deregister should be safe, got: %v", err)
	}
	if err := exporter.ForceFlush(); err != nil {
		t.Fatalf("ForceFlush failed: %v", err)
	}
	if err := exporter.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}

	records := readAtofRecords(t, path)
	if len(records) != 3 {
		t.Fatalf("expected 3 records, got %d", len(records))
	}
	if records[0]["kind"] != "scope" || records[0]["name"] != "atof_scope" {
		t.Fatalf("unexpected first record: %#v", records[0])
	}
	if records[1]["kind"] != "mark" || records[1]["name"] != "atof_mark" {
		t.Fatalf("unexpected mark record: %#v", records[1])
	}
	if records[2]["scope_category"] != "end" {
		t.Fatalf("expected end scope record, got %#v", records[2])
	}
}

func TestAtofExporterAppendAndOverwriteModes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "events.jsonl")
	if err := os.WriteFile(path, []byte("{\"existing\":true}\n"), 0o600); err != nil {
		t.Fatalf("write seed file: %v", err)
	}

	appendExporter, err := NewAtofExporter(AtofExporterConfig{
		OutputDirectory: dir,
		Filename:        "events.jsonl",
	})
	if err != nil {
		t.Fatalf("append NewAtofExporter failed: %v", err)
	}
	if err := appendExporter.Shutdown(); err != nil {
		t.Fatalf("append Shutdown failed: %v", err)
	}
	appendExporter.Close()
	if got := string(mustReadFile(t, path)); got != "{\"existing\":true}\n" {
		t.Fatalf("append mode changed file: %q", got)
	}

	overwriteExporter, err := NewAtofExporter(AtofExporterConfig{
		OutputDirectory: dir,
		Mode:            AtofExporterModeOverwrite,
		Filename:        "events.jsonl",
	})
	if err != nil {
		t.Fatalf("overwrite NewAtofExporter failed: %v", err)
	}
	if err := overwriteExporter.Shutdown(); err != nil {
		t.Fatalf("overwrite Shutdown failed: %v", err)
	}
	overwriteExporter.Close()
	if got := string(mustReadFile(t, path)); got != "" {
		t.Fatalf("overwrite mode did not truncate file: %q", got)
	}
}

func readAtofRecords(t *testing.T, path string) []map[string]interface{} {
	t.Helper()
	content := strings.TrimSpace(string(mustReadFile(t, path)))
	if content == "" {
		return nil
	}
	lines := strings.Split(content, "\n")
	records := make([]map[string]interface{}, 0, len(lines))
	for _, line := range lines {
		var record map[string]interface{}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("invalid JSONL record %q: %v", line, err)
		}
		records = append(records, record)
	}
	return records
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return content
}
