// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestCacheMissTaggedJSONRoundTrips(t *testing.T) {
	for _, payload := range []string{
		`{"kind":"prefix_mismatch","first_mismatch_span_id":"span-1","sequence_index":0,"expected_hash_prefix":"expected","actual_hash_prefix":"actual"}`,
		`{"kind":"below_minimum_threshold","observed_prefix_tokens":0,"required_min_tokens":1024,"estimation_source":"provider"}`,
		`{"kind":"retention_expired","observed_gap_secs":0,"retention_window_secs":300,"provider_semantics":"test"}`,
		`{"kind":"unknown","missing_facts":[]}`,
	} {
		original := json.RawMessage(`{"summary":"miss","recommendation":"retry","evidence":` + payload + `}`)
		var diagnosis CacheMissDiagnosis
		if err := json.Unmarshal(original, &diagnosis); err != nil {
			t.Fatal(err)
		}
		actual, err := json.Marshal(diagnosis)
		if err != nil {
			t.Fatal(err)
		}
		assertSemanticJSONEqual(t, original, actual)
		if reflect.TypeOf(diagnosis.Evidence).Kind() != reflect.Pointer {
			t.Fatalf("expected concrete decoded evidence: %T", diagnosis.Evidence)
		}
	}
	for _, reason := range []CacheMissReasonKind{CacheMissReasonPrefixMismatch, CacheMissReasonBelowMinimumThreshold, CacheMissReasonRetentionExpired, CacheMissReasonRoutingMismatch, CacheMissReasonEvicted, CacheMissReasonUnsupportedFeature, CacheMissReasonColdStart, CacheMissReasonUnknown, CacheMissReasonOther} {
		original := CacheMissReason{Reason: reason}
		payload, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var decoded CacheMissReason
		if err := json.Unmarshal(payload, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded != original {
			t.Fatalf("reason changed: %+v -> %+v", original, decoded)
		}
	}
	for _, payload := range []string{`{"reason":"future"}`, `{"reason":"other"}`} {
		var reason CacheMissReason
		if err := json.Unmarshal([]byte(payload), &reason); err == nil {
			t.Fatalf("invalid reason accepted: %s", payload)
		}
	}
	var diagnosis CacheMissDiagnosis
	if err := json.Unmarshal([]byte(`{"evidence":{"kind":"future"}}`), &diagnosis); err == nil {
		t.Fatal("invalid evidence kind accepted")
	}
}

func TestNativeCacheTelemetryDecodesConcreteDiagnosis(t *testing.T) {
	index := uint32(3)
	span, expected, actual := "stable-span-3", "sha256:1234567890abcdef", "sha256:fedcba0987654321"
	event, err := BuildCacheTelemetryEvent(CacheTelemetryEventInput{
		Provider: "openai", RequestID: "00000000-0000-0000-0000-000000000401",
		Usage:        &CacheUsage{PromptTokens: uint64Ptr(1000), CacheReadTokens: uint64Ptr(0)},
		RequestFacts: &CacheRequestFacts{Provider: "openai", StablePrefixLength: 3, StablePrefixTokens: uint32PtrForCacheTest(1400), RequiredMinTokens: uint32PtrForCacheTest(1024), FirstMismatchSpanID: &span, FirstMismatchSequenceIndex: &index, ExpectedHashPrefix: &expected, ActualHashPrefix: &actual},
		AgentID:      testAgentID, TemplateVersion: "v1", ToolsetHash: "tools", ModelFamily: "gpt", TenantScope: "tenant",
	})
	if err != nil {
		t.Fatal(err)
	}
	if event == nil || event.MissReason == nil || event.MissDiagnosis == nil {
		t.Fatalf("missing typed diagnosis: %+v", event)
	}
	if event.MissReason.Reason != CacheMissReasonPrefixMismatch {
		t.Fatalf("reason: %+v", event.MissReason)
	}
	evidence, ok := event.MissDiagnosis.Evidence.(*CacheMissPrefixMismatchEvidence)
	if !ok || evidence.FirstMismatchSpanID != span || evidence.SequenceIndex != index {
		t.Fatalf("unexpected evidence: %+v", event.MissDiagnosis.Evidence)
	}
	if event.MissDiagnosis.Summary == "" || event.MissDiagnosis.Recommendation == "" {
		t.Fatal("missing explanation")
	}
	payload, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	var decoded CacheTelemetryEvent
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(event.MissDiagnosis, decoded.MissDiagnosis) {
		t.Fatal("native evidence did not survive round trip")
	}
}

func uint32PtrForCacheTest(value uint32) *uint32 { return &value }
