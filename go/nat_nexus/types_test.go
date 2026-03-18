// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"testing"
)

// ============================================================================
// Types and Constants
// ============================================================================

func TestScopeTypeConstants(t *testing.T) {
	types := []ScopeType{
		ScopeTypeAgent, ScopeTypeFunction, ScopeTypeTool, ScopeTypeLlm,
		ScopeTypeRetriever, ScopeTypeEmbedder, ScopeTypeReranker,
		ScopeTypeGuardrail, ScopeTypeEvaluator, ScopeTypeCustom, ScopeTypeUnknown,
	}
	if len(types) != 11 {
		t.Fatalf("expected 11 scope types, got %d", len(types))
	}
	// Verify sequential values
	for i, st := range types {
		if int(st) != i {
			t.Fatalf("ScopeType at index %d has value %d", i, int(st))
		}
	}
}

func TestEventTypeConstants(t *testing.T) {
	if EventTypeStart != 0 || EventTypeEnd != 1 || EventTypeMark != 2 {
		t.Fatal("unexpected EventType values")
	}
}

func TestScopeAttributeConstants(t *testing.T) {
	if ScopeAttrParallel != 0b01 || ScopeAttrRelocatable != 0b10 {
		t.Fatal("unexpected ScopeAttr values")
	}
	combined := ScopeAttrParallel | ScopeAttrRelocatable
	if combined != 0b11 {
		t.Fatal("combined scope attributes incorrect")
	}
}

func TestToolAttributeConstants(t *testing.T) {
	if ToolAttrLocal != 0b01 {
		t.Fatal("unexpected ToolAttr value")
	}
}

func TestLLMAttributeConstants(t *testing.T) {
	if LLMAttrStateless != 0b01 || LLMAttrStreaming != 0b10 {
		t.Fatal("unexpected LLMAttr values")
	}
}

// ============================================================================
// LLMRequest
// ============================================================================

func TestNewLLMRequest(t *testing.T) {
	req := NewLLMRequest(
		map[string]interface{}{"Authorization": "Bearer token"},
		map[string]interface{}{"messages": []string{}},
	)
	if req == nil {
		t.Fatal("NewLLMRequest returned nil")
	}
	if req.Headers() == nil {
		t.Fatal("headers is nil")
	}
	if req.Content() == nil {
		t.Fatal("content is nil")
	}
}

func TestNewLLMRequestEmptyHeaders(t *testing.T) {
	req := NewLLMRequest(map[string]interface{}{}, map[string]interface{}{})
	if req == nil {
		t.Fatal("returned nil")
	}
	if req.Headers() == nil {
		t.Fatal("headers is nil")
	}
}
