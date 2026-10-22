// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestDiagnosticMemoryCanonicalFixture(t *testing.T) {
	data, err := os.ReadFile("../../crates/types/tests/fixtures/diagnostics/diagnostic-memory-schema-1.json")
	if err != nil {
		t.Fatal(err)
	}
	var page DiagnosticMemoryPage
	if err := json.Unmarshal(data, &page); err != nil {
		t.Fatal(err)
	}
	if uint64(page.Sequence) != math.MaxUint64 || uint64(page.Aggregate.ReservedBytes) != 9007199254740993 {
		t.Fatal("memory quantities lost precision")
	}
	if page.Components[0].Component.Key != "9007199254740993" {
		t.Fatal("component text changed")
	}
	if page.ComponentOccupiedBytes+page.SharedOrUnassignedReservedBytes != page.Aggregate.ReservedBytes {
		t.Fatal("memory accounting changed")
	}
	sample := page.Resource.Sample
	if sample == nil || uint64(sample.Ordinal) != math.MaxUint64 {
		t.Fatal("resource identity lost precision")
	}
	var integer DiagnosticNumber
	if err := json.Unmarshal(sample.Memory.SystemUsed.Value.Value, &integer); err != nil || uint64(integer) != math.MaxUint64 {
		t.Fatalf("integer resource: %v %v", integer, err)
	}
	var fractional float64
	if err := json.Unmarshal(sample.Memory.Private.Value.Value, &fractional); err != nil || fractional != 1.25 {
		t.Fatalf("fractional resource: %v %v", fractional, err)
	}
	if sample.Memory.VirtualMemory.Value.Kind != "non_finite" || sample.Memory.SystemAvailable != nil {
		t.Fatal("resource availability changed")
	}
	if sample.Memory.SomePressureStallTime.MeasurementScope != "environment" || sample.Memory.SystemUsed.MeasurementScope != "system" {
		t.Fatal("field scopes changed")
	}
	encoded, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}
	var originalJSON, roundTripJSON any
	if err := json.Unmarshal(data, &originalJSON); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &roundTripJSON); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(originalJSON, roundTripJSON) {
		t.Fatal("memory fixture fields changed during round trip")
	}
	var again DiagnosticMemoryPage
	if err := json.Unmarshal(encoded, &again); err != nil {
		t.Fatal(err)
	}
	if again.Sequence != page.Sequence || again.Resource.Sample.Ordinal != sample.Ordinal {
		t.Fatal("round trip lost identity")
	}
}

func TestDiagnosticMemoryRejectsInvalidEnvelopes(t *testing.T) {
	for _, test := range []struct{ name, input, reason string }{
		{"malformed", "{", ""},
		{"missing_version", "{}", "unsupported_schema"},
		{"future_version", `{"schema_version":2}`, "unsupported_schema"},
		{"noncanonical_integer", `{"schema_version":1,"aggregate":{"reserved_bytes":9007199254740993}}`, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			var page DiagnosticMemoryPage
			err := json.Unmarshal([]byte(test.input), &page)
			if err == nil {
				t.Fatal("invalid memory envelope accepted")
			}
			if test.reason != "" {
				var diagnostic *DiagnosticsError
				if !errors.As(err, &diagnostic) || diagnostic.Reason != test.reason {
					t.Fatalf("reason: %v", err)
				}
			}
		})
	}
}

func TestMemoryReaderReportsClosedWithoutStartingService(t *testing.T) {
	for _, reader := range []*DiagnosticsReader{nil, {}, {state: &diagnosticsReaderState{}}} {
		_, rawErr := reader.InspectMemoryJSON(DiagnosticQuery{})
		_, typedErr := reader.InspectMemory(DiagnosticQuery{})
		for _, err := range []error{rawErr, typedErr} {
			var diagnostic *DiagnosticsError
			if !errors.As(err, &diagnostic) || diagnostic.Reason != "closed" {
				t.Fatalf("closed reader: %v", err)
			}
		}
	}
}
