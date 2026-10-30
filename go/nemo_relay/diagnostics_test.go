// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestQueryByteLimitMatchesSharedContractFixture(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "crates", "diagnostics", "tests", "fixtures", "diagnostics_query_contract.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		MaxQueryBytes uint32 `json:"max_query_bytes"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if actual := uint32(maxDiagnosticsQueryBytes()); actual != fixture.MaxQueryBytes {
		t.Fatalf("query byte limit: got %d, want %d", actual, fixture.MaxQueryBytes)
	}
}

func TestDiagnosticNumberLossless(t *testing.T) {
	for _, input := range []string{`"9007199254740993"`, `"18446744073709551615"`} {
		var value DiagnosticNumber
		if err := json.Unmarshal([]byte(input), &value); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(value)
		if err != nil || string(encoded) != input {
			t.Fatalf("round trip: %s %v", encoded, err)
		}
	}
	for _, input := range []string{`9007199254740993`, `"01"`, `"-1"`, `"18446744073709551616"`, `"+1"`} {
		var value DiagnosticNumber
		if json.Unmarshal([]byte(input), &value) == nil {
			t.Fatalf("accepted %s", input)
		}
	}
	var count DiagnosticCount
	if err := json.Unmarshal([]byte(`{"value":"18446744073709551615","saturated":true}`), &count); err != nil || uint64(count.Value) != math.MaxUint64 || !count.Saturated {
		t.Fatalf("saturation: %+v %v", count, err)
	}
}

func TestDiagnosticsConfiguredService(t *testing.T) {
	if os.Getenv("RELAY_DIAGNOSTICS_TEST_CHILD") != "1" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		command := exec.Command(executable, "-test.run=^TestDiagnosticsConfiguredService$")
		command.Env = append(os.Environ(), "RELAY_DIAGNOSTICS_TEST_CHILD=1", "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG=1")
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("%v\n%s", err, output)
		}
		return
	}
	if _, err := ReadDiagnostics(); err == nil {
		t.Fatal("import selected diagnostics")
	}
	config := NewPluginConfig()
	config.Diagnostics = &DiagnosticsSettings{Enabled: true, Limits: map[string]uint64{"max_events": 3}}
	activation, _, err := Initialize(config, nil)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := ReadDiagnostics()
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	memory, err := reader.InspectMemory(DiagnosticQuery{})
	if err != nil {
		t.Fatal(err)
	}
	if memory.ComponentOccupiedBytes+memory.SharedOrUnassignedReservedBytes != memory.Aggregate.ReservedBytes {
		t.Fatal("memory conservation failed")
	}
	if memory.Resource.Availability != "disabled" {
		t.Fatal("inspection started resource collection")
	}

	first, err := reader.Snapshot(DiagnosticQuery{})
	if err != nil {
		t.Fatal(err)
	}
	if err := activation.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := reader.Snapshot(DiagnosticQuery{})
	if err != nil {
		t.Fatal(err)
	}
	if first.Source.InstanceID != second.Source.InstanceID {
		t.Fatal("plugin close replaced service")
	}
	version := uint32(2)
	_, versionErr := reader.Snapshot(DiagnosticQuery{SchemaVersion: &version})
	var unsupported *DiagnosticsError
	if !errors.As(versionErr, &unsupported) || unsupported.Reason != "unsupported_schema" {
		t.Fatalf("unsupported version: %v", versionErr)
	}
	reader.Close()
	reader.Close()
	_, err = reader.Snapshot(DiagnosticQuery{})
	var unavailable *DiagnosticsError
	if !errors.As(err, &unavailable) || unavailable.Reason != "closed" {
		t.Fatalf("closed: %v", err)
	}
}

func TestDiagnosticCanonicalFixture(t *testing.T) {
	data, err := os.ReadFile("../../crates/types/tests/fixtures/diagnostics/diagnostic-snapshot-schema-1.json")
	if err != nil {
		t.Fatal(err)
	}
	var page DiagnosticSnapshot
	if err := json.Unmarshal(data, &page); err != nil {
		t.Fatal(err)
	}
	if uint64(page.Sequence) != math.MaxUint64 || uint64(page.Events[0].Count) != 9007199254740993 {
		t.Fatal("canonical fixture lost precision")
	}
	if !page.Lifetime.Totals.OccurrencesTotal.Saturated {
		t.Fatal("lost saturation")
	}
}

func TestDiagnosticSnapshotRejectsUnsupportedEnvelope(t *testing.T) {
	for _, data := range []string{`{"schema_version":2}`, `{}`} {
		var page DiagnosticSnapshot
		var reason *DiagnosticsError
		err := json.Unmarshal([]byte(data), &page)
		if !errors.As(err, &reason) || reason.Reason != "unsupported_schema" {
			t.Fatalf("unsupported snapshot: %v", err)
		}
	}
}
