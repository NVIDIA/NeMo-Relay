// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"runtime"
	"testing"
	"time"
)

func TestEventBaseNilPointerFallbacks(t *testing.T) {
	event := eventBase{}

	if got := event.UUID(); got != "" {
		t.Fatalf("expected empty UUID, got %q", got)
	}
	if got := event.Name(); got != "" {
		t.Fatalf("expected empty Name, got %q", got)
	}
	if got := event.Kind(); got != "" {
		t.Fatalf("expected empty Kind, got %q", got)
	}
	if got := event.ScopeType(); got != "" {
		t.Fatalf("expected empty ScopeType, got %q", got)
	}
	if got := event.Attributes(); got != 0 {
		t.Fatalf("expected zero Attributes, got %d", got)
	}
	if got := event.Data(); got != nil {
		t.Fatalf("expected nil Data, got %s", got)
	}
	if got := event.Metadata(); got != nil {
		t.Fatalf("expected nil Metadata, got %s", got)
	}
	if got := event.Timestamp(); got != "" {
		t.Fatalf("expected empty Timestamp, got %q", got)
	}
	if got := event.Input(); got != nil {
		t.Fatalf("expected nil Input, got %s", got)
	}
	if got := event.Output(); got != nil {
		t.Fatalf("expected nil Output, got %s", got)
	}
	if got := event.ModelName(); got != "" {
		t.Fatalf("expected empty ModelName, got %q", got)
	}
	if got := event.ToolCallID(); got != "" {
		t.Fatalf("expected empty ToolCallID, got %q", got)
	}
	if got := event.ParentUUID(); got != "" {
		t.Fatalf("expected empty ParentUUID, got %q", got)
	}
	if got := event.AnnotatedRequest(); got != nil {
		t.Fatalf("expected nil AnnotatedRequest, got %s", got)
	}
	if got := event.AnnotatedResponse(); got != nil {
		t.Fatalf("expected nil AnnotatedResponse, got %s", got)
	}
}

func TestPublicAPIErrorAndDefaultCoverage(t *testing.T) {
	if _, err := PushScope("invalid_scope_json", ScopeTypeAgent, WithData(json.RawMessage("{"))); err == nil {
		t.Fatal("expected PushScope to fail on invalid JSON data")
	}

	if _, err := ToolCall("invalid_tool_json", json.RawMessage("{")); err == nil {
		t.Fatal("expected ToolCall to fail on invalid JSON args")
	}

	badMarshal := map[string]interface{}{"ch": make(chan int)}
	if _, err := LlmCall("llm_marshal_error", badMarshal); err == nil {
		t.Fatal("expected LlmCall marshal error")
	}
	if _, err := LlmCallExecute("llm_execute_marshal_error", badMarshal, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}); err == nil {
		t.Fatal("expected LlmCallExecute marshal error")
	}
	if _, err := LlmStreamCallExecute("llm_stream_marshal_error", badMarshal, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}, nil, nil); err == nil {
		t.Fatal("expected LlmStreamCallExecute marshal error")
	}

	malformedRequest := map[string]interface{}{"not": "an LLMRequest"}
	if _, err := LlmCall("llm_invalid_request", malformedRequest); err == nil {
		t.Fatal("expected LlmCall request-shape error")
	}
	if _, err := LlmCallExecute("llm_execute_invalid_request", malformedRequest, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}); err == nil {
		t.Fatal("expected LlmCallExecute request-shape error")
	}
	if _, err := LlmStreamCallExecute("llm_stream_invalid_request", malformedRequest, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}, nil, nil); err == nil {
		t.Fatal("expected LlmStreamCallExecute request-shape error")
	}

	if _, err := ToolRequestIntercepts("invalid_tool_request_intercepts", json.RawMessage("{")); err == nil {
		t.Fatal("expected ToolRequestIntercepts to fail on invalid JSON")
	}
	if _, err := LlmRequestIntercepts("invalid_llm_request_intercepts", json.RawMessage("{")); err == nil {
		t.Fatal("expected LlmRequestIntercepts to fail on invalid JSON")
	}

	exporter, err := NewAtifExporter("session-gap", "agent-gap", "1.0.0", "")
	if err != nil {
		t.Fatalf("NewAtifExporter failed: %v", err)
	}
	exporter.Close()
	if _, err := exporter.ExportJSON(); err == nil {
		t.Fatal("expected ExportJSON to fail after Close")
	}

	otel, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{})
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber with zero config failed: %v", err)
	}
	otel.Close()

	openInference, err := NewOpenInferenceSubscriber(OpenInferenceConfig{})
	if err != nil {
		t.Fatalf("NewOpenInferenceSubscriber with zero config failed: %v", err)
	}
	openInference.Close()

	if got := mustConfigMap(nil); len(got) != 0 {
		t.Fatalf("expected empty map for nil config payload, got %#v", got)
	}
}

func TestWrapperAndCodecFinalizersRun(t *testing.T) {
	scopeHandle, err := PushScope("finalizer_scope", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := PopScope(scopeHandle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}

	toolHandle, err := ToolCall("finalizer_tool", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if err := ToolCallEnd(toolHandle, json.RawMessage(`{}`)); err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}

	llmHandle, err := LlmCall("finalizer_llm", map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"model": "test-model"},
	})
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if err := LlmCallEnd(llmHandle, json.RawMessage(`{"content":"ok"}`)); err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}

	request := NewLLMRequest(
		map[string]interface{}{"x-test": "finalizer"},
		map[string]interface{}{"model": "test-model"},
	)
	if request == nil {
		t.Fatal("expected non-nil LLMRequest")
	}

	chatCodec := NewOpenAIChatCodec()
	responsesCodec := NewOpenAIResponsesCodec()
	anthropicCodec := NewAnthropicMessagesCodec()
	if chatCodec == nil || responsesCodec == nil || anthropicCodec == nil {
		t.Fatal("expected non-nil codec handles")
	}

	scopeHandle = nil
	toolHandle = nil
	llmHandle = nil
	request = nil
	chatCodec = nil
	responsesCodec = nil
	anthropicCodec = nil

	for i := 0; i < 8; i++ {
		runtime.GC()
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
	}
}
