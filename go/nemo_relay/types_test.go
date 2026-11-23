// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"errors"
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
	if ToolAttrRemote != 0b01 {
		t.Fatal("unexpected ToolAttr value")
	}
}

func TestLLMAttributeConstants(t *testing.T) {
	if LLMAttrStateful != 0b01 || LLMAttrStreaming != 0b10 {
		t.Fatal("unexpected LLMAttr values")
	}
}

// ============================================================================
// LLMRequest
// ============================================================================

func TestNewLLMRequest(t *testing.T) {
	req, err := NewLLMRequest(jsonForTest(map[string]interface{}{"Authorization": "Bearer token"}), jsonForTest(map[string]interface{}{"messages": []string{}}))
	if err != nil {
		t.Fatal(err)
	}
	if req.Headers == nil {
		t.Fatal("NewLLMRequest returned nil")
	}
	if req.Headers == nil {
		t.Fatal("headers is nil")
	}
	if req.Content == nil {
		t.Fatal("content is nil")
	}
}

func TestNewLLMRequestEmptyHeaders(t *testing.T) {
	req, err := NewLLMRequest(jsonForTest(map[string]interface{}{}), jsonForTest(map[string]interface{}{}))
	if err != nil {
		t.Fatal(err)
	}
	if req.Headers == nil {
		t.Fatal("returned nil")
	}
	if req.Headers == nil {
		t.Fatal("headers is nil")
	}
}

func TestNewLLMRequestReportsSerializationAndNativeErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		headers json.RawMessage
		content json.RawMessage
	}{
		{"headers", json.RawMessage(`{`), nil},
		{"content", nil, json.RawMessage(`{`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req, err := NewLLMRequest(tc.headers, tc.content)
			var malformed *json.MarshalerError
			if req.Headers != nil || !errors.As(err, &malformed) {
				t.Fatalf("expected JSON serialization error and no handle, got %v, %v", req, err)
			}
		})
	}

	req, err := newLLMRequestFunc(`{}`, `{`)
	var native *RelayError
	if req.Headers != nil || !errors.As(err, &native) || native.Code != StatusUnknown || native.Message == "" {
		t.Fatalf("expected native diagnostic and no handle, got %v, %v", req, err)
	}
	req, err = NewLLMRequest(jsonForTest(nil), jsonForTest(nil))
	if err != nil {
		t.Fatal(err)
	}
	if string(req.Headers) != `{}` || string(req.Content) != `null` {
		t.Fatalf("nil defaults changed: %s, %s", req.Headers, req.Content)
	}
}

func TestConstructedLLMRequestWorksAcrossPublicAPIs(t *testing.T) {
	runWithTestScopeStack(t, func() {
		request, err := NewLLMRequest(json.RawMessage(`{"x-test":"request-value"}`), json.RawMessage(`{"model":"request-model","messages":[]}`))
		if err != nil {
			t.Fatal(err)
		}
		codec := NewOpenAIChatCodec()
		defer codec.Close()
		annotation, err := codec.Decode(request)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(annotation, []byte(`"request-model"`)) {
			t.Fatalf("codec lost request model: %s", annotation)
		}
		response, err := LlmCallExecute("constructed-request", request, func(payload json.RawMessage) (json.RawMessage, error) {
			var observed LLMRequest
			if err := json.Unmarshal(payload, &observed); err != nil {
				return nil, err
			}
			assertSemanticJSONEqual(t, request.Content, observed.Content)
			return json.RawMessage(`{"ok":true}`), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		assertSemanticJSONEqual(t, json.RawMessage(`{"ok":true}`), response)
	})
}
