// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"runtime"
	"testing"
	"time"
)

func TestEventBaseNilPointerFallbacks(t *testing.T) {
	event := eventBase{}

	assertEmptyEventString(t, "UUID", event.UUID())
	assertEmptyEventString(t, "Name", event.Name())
	assertEmptyEventString(t, "Kind", event.Kind())
	assertEmptyEventString(t, "ATOFVersion", event.ATOFVersion())
	assertEmptyEventString(t, "ScopeType", event.ScopeType())
	assertZeroEventAttributes(t, event.Attributes())
	assertNilEventJSON(t, "AttributesJSON", event.AttributesJSON())
	assertNilEventJSON(t, "Data", event.Data())
	assertNilEventJSON(t, "DataSchema", event.DataSchema())
	assertNilEventJSON(t, "Metadata", event.Metadata())
	assertEmptyEventString(t, "Timestamp", event.Timestamp())
	assertNilEventJSON(t, "Input", event.Input())
	assertNilEventJSON(t, "Output", event.Output())
	assertEmptyEventString(t, "ModelName", event.ModelName())
	assertEmptyEventString(t, "ToolCallID", event.ToolCallID())
	assertEmptyEventString(t, "ParentUUID", event.ParentUUID())
	assertNilEventJSON(t, "AnnotatedRequest", event.AnnotatedRequest())
	assertNilEventJSON(t, "AnnotatedResponse", event.AnnotatedResponse())
}

func assertEmptyEventString(t *testing.T, name string, got string) {
	t.Helper()
	if got != "" {
		t.Fatalf("expected empty %s, got %q", name, got)
	}
}

func assertZeroEventAttributes(t *testing.T, got uint32) {
	t.Helper()
	if got != 0 {
		t.Fatalf("expected zero Attributes, got %d", got)
	}
}

func assertNilEventJSON(t *testing.T, name string, got []byte) {
	t.Helper()
	if got != nil {
		t.Fatalf("expected nil %s, got %s", name, got)
	}
}

func TestPublicAPIErrorAndDefaultCoverage(t *testing.T) {
	runTestWithScopeStack(t, testPublicAPIErrorAndDefaultCoverage)
}

func testPublicAPIErrorAndDefaultCoverage(t *testing.T) {
	assertInvalidScopePayloads(t)
	assertInvalidCallPayloads(t)
	assertClosedExporterFails(t)
	assertZeroSubscriberConfigs(t)
	if got := mustConfigMap(nil); len(got) != 0 {
		t.Fatalf("expected empty map for nil config payload, got %#v", got)
	}
}

func assertInvalidScopePayloads(t *testing.T) {
	t.Helper()
	for _, tc := range []struct {
		name string
		opt  ScopeOption
	}{
		{name: "data", opt: WithData(json.RawMessage("{"))},
		{name: "metadata", opt: WithMetadata(json.RawMessage("{"))},
		{name: "input", opt: WithInput(json.RawMessage("{"))},
	} {
		if _, err := PushScope("invalid_scope_json_"+tc.name, ScopeTypeAgent, tc.opt); err == nil {
			t.Fatalf("expected PushScope to fail on invalid JSON %s", tc.name)
		}
	}

	handle, err := PushScope("invalid_scope_end_metadata", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if PopScope(handle, WithScopeEndMetadata(json.RawMessage("{"))) == nil {
		t.Fatal("expected PopScope to fail on invalid end metadata JSON")
	}
	if err := PopScope(handle); err != nil {
		t.Fatalf("cleanup PopScope failed: %v", err)
	}
}

func assertInvalidCallPayloads(t *testing.T) {
	t.Helper()
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
}

func assertClosedExporterFails(t *testing.T) {
	t.Helper()
	exporter, err := NewAtifExporter("session-gap", "agent-gap", "1.0.0", "")
	if err != nil {
		t.Fatalf("NewAtifExporter failed: %v", err)
	}
	exporter.Close()
	if _, err := exporter.ExportJSON(); err == nil {
		t.Fatal("expected ExportJSON to fail after Close")
	}
}

func assertZeroSubscriberConfigs(t *testing.T) {
	t.Helper()
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
}

func TestWrapperAndCodecFinalizersRun(t *testing.T) {
	runTestWithScopeStack(t, testWrapperAndCodecFinalizersRun)
}

func testWrapperAndCodecFinalizersRun(t *testing.T) {
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
