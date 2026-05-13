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

const eventsJSONLFilename = "events.jsonl"

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
		Filename:        eventsJSONLFilename,
	})
	requireNoError(t, err, "NewAtofExporter failed")
	defer exporter.Close()

	path, err := exporter.Path()
	requireNoError(t, err, "Path failed")
	requireEqual(t, filepath.Base(path), eventsJSONLFilename, "expected %s path", eventsJSONLFilename)

	name := "go_atof_" + time.Now().Format("150405.000000")
	requireNoError(t, exporter.Register(name), "Register failed")
	handle, err := PushScope("atof_scope", ScopeTypeAgent, WithInput(json.RawMessage(`{"scope":true}`)))
	requireNoError(t, err, "PushScope failed")
	requireNoError(t, EmitEvent("atof_mark", WithEventParent(handle), WithEventData(json.RawMessage(`{"step":1}`))), "EmitEvent failed")
	requireNoError(t, PopScope(handle, WithOutput(json.RawMessage(`{"done":true}`))), "PopScope failed")
	requireNoError(t, exporter.Deregister(name), "Deregister failed")
	requireNoError(t, exporter.Deregister(name), "repeated Deregister should be safe")
	requireNoError(t, exporter.ForceFlush(), "ForceFlush failed")
	requireNoError(t, exporter.Shutdown(), "Shutdown failed")

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
	path := filepath.Join(dir, eventsJSONLFilename)
	if err := os.WriteFile(path, []byte("{\"existing\":true}\n"), 0o600); err != nil {
		t.Fatalf("write seed file: %v", err)
	}

	appendExporter, err := NewAtofExporter(AtofExporterConfig{
		OutputDirectory: dir,
		Filename:        eventsJSONLFilename,
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
		Filename:        eventsJSONLFilename,
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

func requireNoError(t *testing.T, err error, message string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", message, err)
	}
}

func requireEqual[T comparable](t *testing.T, got T, want T, message string, args ...any) {
	t.Helper()
	if got != want {
		t.Fatalf(message+", got %q", append(args, got)...)
	}
}
