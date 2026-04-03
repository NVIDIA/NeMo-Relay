// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
	"testing"
)

func TestStandaloneMiddlewareHelpers(t *testing.T) {
	if err := RegisterToolRequestIntercept("go_standalone_tool_req", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(args, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterToolRequestIntercept failed: %v", err)
	}
	defer DeregisterToolRequestIntercept("go_standalone_tool_req")

	args, err := ToolRequestIntercepts("standalone_tool", json.RawMessage(`{"value": 1}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts failed: %v", err)
	}
	var toolPayload map[string]interface{}
	if err := json.Unmarshal(args, &toolPayload); err != nil {
		t.Fatalf("unmarshal tool args: %v", err)
	}
	if toolPayload["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", toolPayload)
	}

	if err := RegisterToolConditionalExecutionGuardrail("go_standalone_tool_cond", 1,
		func(name string, args json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterToolConditionalExecutionGuardrail failed: %v", err)
	}
	defer DeregisterToolConditionalExecutionGuardrail("go_standalone_tool_cond")

	if err := ToolConditionalExecution("standalone_tool", json.RawMessage(`{"value": 1}`)); err != nil {
		t.Fatalf("ToolConditionalExecution failed: %v", err)
	}

	if err := RegisterLlmRequestIntercept("go_standalone_llm_req", 1, false,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			var payload map[string]interface{}
			_ = json.Unmarshal(content, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return headers, out
		},
	); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept failed: %v", err)
	}
	defer DeregisterLlmRequestIntercept("go_standalone_llm_req")

	request, err := LlmRequestIntercepts("standalone_llm", json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`))
	if err != nil {
		t.Fatalf("LlmRequestIntercepts failed: %v", err)
	}
	var llmPayload struct {
		Content map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(request, &llmPayload); err != nil {
		t.Fatalf("unmarshal llm request: %v", err)
	}
	if llmPayload.Content["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", llmPayload.Content)
	}

	if err := RegisterLlmConditionalExecutionGuardrail("go_standalone_llm_cond", 1,
		func(headers, content json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterLlmConditionalExecutionGuardrail failed: %v", err)
	}
	defer DeregisterLlmConditionalExecutionGuardrail("go_standalone_llm_cond")

	if err := LlmConditionalExecution(json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`)); err != nil {
		t.Fatalf("LlmConditionalExecution failed: %v", err)
	}
}

func TestNilWrapperConstructors(t *testing.T) {
	if got := newScopeHandle(nil); got != nil {
		t.Fatalf("expected nil scope handle, got %#v", got)
	}
	if got := newToolHandle(nil); got != nil {
		t.Fatalf("expected nil tool handle, got %#v", got)
	}
	if got := newLLMHandle(nil); got != nil {
		t.Fatalf("expected nil llm handle, got %#v", got)
	}
	if got := newLlmStream(nil, nil, nil); got != nil {
		t.Fatalf("expected nil llm stream, got %#v", got)
	}
}

func TestNewLLMRequestRoundTrip(t *testing.T) {
	req := NewLLMRequest(
		map[string]interface{}{"Authorization": "Bearer token"},
		map[string]interface{}{"model": "test-model", "messages": []interface{}{}},
	)
	if req == nil {
		t.Fatal("expected non-nil LLM request")
	}

	var headers map[string]interface{}
	if err := json.Unmarshal(req.Headers(), &headers); err != nil {
		t.Fatalf("unmarshal headers: %v", err)
	}
	if headers["Authorization"] != "Bearer token" {
		t.Fatalf("expected Authorization header, got %v", headers)
	}

	var content map[string]interface{}
	if err := json.Unmarshal(req.Content(), &content); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if content["model"] != "test-model" {
		t.Fatalf("expected model=test-model, got %v", content)
	}
}

func TestAtifExporterLifecycleAndFiltering(t *testing.T) {
	exporter, err := NewAtifExporter("session-go", "go-agent", "1.0.0", "test-model")
	if err != nil {
		t.Fatalf("NewAtifExporter failed: %v", err)
	}
	defer exporter.Close()

	if err := exporter.Register("go_atif_exporter"); err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	stack1, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack stack1 failed: %v", err)
	}
	defer stack1.Close()

	stack2, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack stack2 failed: %v", err)
	}
	defer stack2.Close()

	var root1 string
	var root2 string

	stack1.Run(func() {
		handle, err := GetHandle()
		if err != nil {
			t.Fatalf("GetHandle stack1 failed: %v", err)
		}
		root1 = handle.UUID()

		_, err = LlmCallExecute("atif_llm_1", map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"messages": []map[string]interface{}{{"role": "user", "content": "agent one"}},
				"model":    "test-model",
			},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"content":"response one","role":"assistant","token_usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"tool_calls":[]}`), nil
		}, WithLLMModelName("test-model"))
		if err != nil {
			t.Fatalf("LlmCallExecute stack1 failed: %v", err)
		}
	})

	stack2.Run(func() {
		handle, err := GetHandle()
		if err != nil {
			t.Fatalf("GetHandle stack2 failed: %v", err)
		}
		root2 = handle.UUID()

		_, err = LlmCallExecute("atif_llm_2", map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"messages": []map[string]interface{}{{"role": "user", "content": "agent two"}},
				"model":    "test-model",
			},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"content":"response two","role":"assistant","token_usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9},"tool_calls":[]}`), nil
		}, WithLLMModelName("test-model"))
		if err != nil {
			t.Fatalf("LlmCallExecute stack2 failed: %v", err)
		}
	})

	allJSON, err := exporter.ExportJSON("")
	if err != nil {
		t.Fatalf("ExportJSON all failed: %v", err)
	}

	var all struct {
		SchemaVersion string `json:"schema_version"`
		Steps         []struct {
			Message json.RawMessage `json:"message"`
		} `json:"steps"`
		FinalMetrics map[string]interface{} `json:"final_metrics"`
	}
	if err := json.Unmarshal(allJSON, &all); err != nil {
		t.Fatalf("unmarshal all trajectory: %v", err)
	}
	if all.SchemaVersion == "" {
		t.Fatal("expected schema_version to be present")
	}
	if len(all.Steps) < 4 {
		t.Fatalf("expected at least four ATIF steps, got %d", len(all.Steps))
	}
	if all.FinalMetrics == nil {
		t.Fatal("expected aggregated final metrics")
	}

	filteredJSON, err := exporter.ExportJSON(root1)
	if err != nil {
		t.Fatalf("ExportJSON filtered failed: %v", err)
	}

	var filtered struct {
		Steps []struct {
			Message json.RawMessage `json:"message"`
		} `json:"steps"`
	}
	if err := json.Unmarshal(filteredJSON, &filtered); err != nil {
		t.Fatalf("unmarshal filtered trajectory: %v", err)
	}
	if len(filtered.Steps) != 2 {
		t.Fatalf("expected two filtered steps, got %d", len(filtered.Steps))
	}
	if string(filtered.Steps[1].Message) != `"response one"` {
		t.Fatalf("expected filtered response one, got %s", filtered.Steps[1].Message)
	}

	if err := exporter.Deregister("go_atif_exporter"); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}

	exporter.Clear()
	emptyJSON, err := exporter.ExportJSON("")
	if err != nil {
		t.Fatalf("ExportJSON after clear failed: %v", err)
	}
	var empty struct {
		Steps []json.RawMessage `json:"steps"`
	}
	if err := json.Unmarshal(emptyJSON, &empty); err != nil {
		t.Fatalf("unmarshal empty trajectory: %v", err)
	}
	if len(empty.Steps) != 0 {
		t.Fatalf("expected zero steps after clear, got %d", len(empty.Steps))
	}

	stack1.Run(func() {
		_, err := LlmCallExecute("atif_llm_after_deregister", map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"messages": []map[string]interface{}{{"role": "user", "content": "ignored"}},
				"model":    "test-model",
			},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"content":"ignored","role":"assistant","tool_calls":[]}`), nil
		})
		if err != nil {
			t.Fatalf("LlmCallExecute after deregister failed: %v", err)
		}
	})

	afterDeregisterJSON, err := exporter.ExportJSON(root2)
	if err != nil {
		t.Fatalf("ExportJSON after deregister failed: %v", err)
	}
	var afterDeregister struct {
		Steps []json.RawMessage `json:"steps"`
	}
	if err := json.Unmarshal(afterDeregisterJSON, &afterDeregister); err != nil {
		t.Fatalf("unmarshal after deregister trajectory: %v", err)
	}
	if len(afterDeregister.Steps) != 0 {
		t.Fatalf("expected no captured steps after deregister, got %d", len(afterDeregister.Steps))
	}

	exporter.Close()
	exporter.Close()
}
