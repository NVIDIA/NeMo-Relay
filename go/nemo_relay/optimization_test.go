// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func optimizationContributionFixture(t *testing.T) ([]byte, LLMOptimizationContribution) {
	t.Helper()
	fixture, err := os.ReadFile("../../crates/types/tests/fixtures/llm_optimization_contribution_v1.json")
	if err != nil {
		t.Fatalf("read optimization fixture: %v", err)
	}
	var contribution LLMOptimizationContribution
	if err := json.Unmarshal(fixture, &contribution); err != nil {
		t.Fatalf("decode optimization fixture: %v", err)
	}
	return fixture, contribution
}

func assertSemanticJSONEqual(t *testing.T, expected, actual []byte) {
	t.Helper()
	var expectedValue any
	var actualValue any
	if err := json.Unmarshal(expected, &expectedValue); err != nil {
		t.Fatalf("decode expected JSON: %v", err)
	}
	if err := json.Unmarshal(actual, &actualValue); err != nil {
		t.Fatalf("decode actual JSON: %v", err)
	}
	if !jsonValuesEqual(expectedValue, actualValue) {
		t.Fatalf("JSON mismatch\nexpected: %s\nactual:   %s", expected, actual)
	}
}

func jsonValuesEqual(left, right any) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func TestLLMOptimizationContributionFixtureRoundTrip(t *testing.T) {
	fixture, contribution := optimizationContributionFixture(t)
	if contribution.Kind == LLMOptimizationKindInputCompression || contribution.Kind == LLMOptimizationKindModelRouting {
		t.Fatalf("fixture kind must exercise the open-string contract: %q", contribution.Kind)
	}
	if _, ok := contribution.Extra["future_top_level_field"]; !ok {
		t.Fatal("expected future_top_level_field to be retained in Extra")
	}
	wire, err := json.Marshal(contribution)
	if err != nil {
		t.Fatalf("encode optimization fixture: %v", err)
	}
	assertSemanticJSONEqual(t, fixture, wire)
}

func TestLLMOptimizationContributionRequiresPayloadSchema(t *testing.T) {
	contribution := LLMOptimizationContribution{
		Producer: "test",
		Kind:     "custom",
		Payload:  json.RawMessage(`{"value":1}`),
	}
	if _, err := json.Marshal(contribution); err == nil || !strings.Contains(err.Error(), "payload_schema") {
		t.Fatalf("expected marshal payload_schema error, got %v", err)
	}

	var decoded LLMOptimizationContribution
	if err := json.Unmarshal(
		[]byte(`{"producer":"test","kind":"custom","payload":{"value":1}}`),
		&decoded,
	); err == nil || !strings.Contains(err.Error(), "payload_schema") {
		t.Fatalf("expected unmarshal payload_schema error, got %v", err)
	}

	if err := json.Unmarshal(
		[]byte(`{"producer":"test","kind":"custom","payload":null}`),
		&decoded,
	); err != nil {
		t.Fatalf("null payload should behave like an absent payload: %v", err)
	}
}

func TestLLMOptimizationContributionOmittedAppliedIsNonApplied(t *testing.T) {
	var contribution LLMOptimizationContribution
	if err := json.Unmarshal(
		[]byte(`{"producer":"test","kind":"custom"}`),
		&contribution,
	); err != nil {
		t.Fatalf("decode optimization contribution: %v", err)
	}
	if contribution.Applied {
		t.Fatal("omitted applied must decode as false")
	}
}

func TestLLMRequestInterceptOptimizationContributionsRoundTrip(t *testing.T) {
	fixture, contribution := optimizationContributionFixture(t)
	const interceptName = "go_optimization_fixture"
	if err := RegisterLlmRequestIntercept(interceptName, 1, false,
		func(_ string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
			return LLMRequestInterceptOutcome{
				Request:                   request,
				AnnotatedRequest:          annotated,
				OptimizationContributions: []LLMOptimizationContribution{contribution},
			}, nil
		}); err != nil {
		t.Fatalf("register request intercept: %v", err)
	}
	t.Cleanup(func() { _ = DeregisterLlmRequestIntercept(interceptName) })

	outcome, err := LlmRequestIntercepts(interceptName, json.RawMessage(`{"headers":{},"content":{}}`))
	if err != nil {
		t.Fatalf("run request intercepts: %v", err)
	}
	if len(outcome.OptimizationContributions) != 1 {
		t.Fatalf("expected one optimization contribution, got %d", len(outcome.OptimizationContributions))
	}
	wire, err := json.Marshal(outcome.OptimizationContributions[0])
	if err != nil {
		t.Fatalf("encode returned contribution: %v", err)
	}
	assertSemanticJSONEqual(t, fixture, wire)
}
