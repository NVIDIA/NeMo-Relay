// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"strings"
	"testing"
)

const wrapperTestModel = "test-model"

func assertTrajectoryStepCount(t *testing.T, raw json.RawMessage, minimum int) {
	t.Helper()

	var decoded struct {
		SchemaVersion string `json:"schema_version"`
		Steps         []struct {
			Message json.RawMessage `json:"message"`
		} `json:"steps"`
		FinalMetrics map[string]interface{} `json:"final_metrics"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal trajectory: %v", err)
	}
	if decoded.SchemaVersion == "" {
		t.Fatal("expected schema_version to be present")
	}
	if len(decoded.Steps) < minimum {
		t.Fatalf("expected at least %d ATIF steps, got %d", minimum, len(decoded.Steps))
	}
	if decoded.FinalMetrics == nil {
		t.Fatal("expected aggregated final metrics")
	}
}

func assertNoTrajectorySteps(t *testing.T, raw json.RawMessage, context string) {
	t.Helper()

	var decoded struct {
		Steps []json.RawMessage `json:"steps"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal %s trajectory: %v", context, err)
	}
	if len(decoded.Steps) != 0 {
		t.Fatalf("expected no captured steps after %s, got %d", context, len(decoded.Steps))
	}
}

func runAtifLLMCall(t *testing.T, stack *ScopeStack, name, prompt, response string) {
	t.Helper()

	stack.Run(func() {
		handle, err := GetHandle()
		if err != nil {
			t.Fatalf("GetHandle failed for %s: %v", name, err)
		}
		_ = handle.UUID()

		_, err = LlmCallExecute(name, map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"messages": []map[string]interface{}{{"role": "user", "content": prompt}},
				"model":    wrapperTestModel,
			},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(response), nil
		}, WithLLMModelName(wrapperTestModel))
		if err != nil {
			t.Fatalf("LlmCallExecute %s failed: %v", name, err)
		}
	})
}

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
		func(name string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
			var payload map[string]interface{}
			_ = json.Unmarshal(request.Content, &payload)
			payload["intercepted"] = true
			request.Content, _ = json.Marshal(payload)
			return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotated}, nil
		},
	); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept failed: %v", err)
	}
	defer DeregisterLlmRequestIntercept("go_standalone_llm_req")

	request, err := LlmRequestIntercepts("standalone_llm", json.RawMessage("{\"headers\":{},\"content\":{\"model\":\""+wrapperTestModel+"\"}}"))
	if err != nil {
		t.Fatalf("LlmRequestIntercepts failed: %v", err)
	}
	var llmPayload struct {
		Content map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(request.Request.Content, &llmPayload.Content); err != nil {
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

	if err := LlmConditionalExecution(json.RawMessage("{\"headers\":{},\"content\":{\"model\":\"" + wrapperTestModel + "\"}}")); err != nil {
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
		map[string]interface{}{"model": wrapperTestModel, "messages": []interface{}{}},
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
	if content["model"] != wrapperTestModel {
		t.Fatalf("expected model=test-model, got %v", content)
	}
}

func TestWrapperHelpersCoverNilAndErrorPaths(t *testing.T) {
	if got := goString(nil); got != "" {
		t.Fatalf("expected empty string for nil goString, got %q", got)
	}
	if got := goStringOpt(nil); got != "" {
		t.Fatalf("expected empty string for nil goStringOpt, got %q", got)
	}
	if got := goJSONOpt(nil); got != nil {
		t.Fatalf("expected nil json for nil goJSONOpt, got %v", got)
	}
	if err := lastError(); err == nil || !strings.Contains(err.Error(), "unknown nemo_relay error") {
		t.Fatalf("expected unknown nemo_relay error fallback, got %v", err)
	}
}

func TestAtifExporterLifecycleAndFiltering(t *testing.T) {
	exporter, err := NewAtifExporter("session-go", "go-agent", "1.0.0", wrapperTestModel)
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

	runAtifLLMCall(t, stack1, "atif_llm_1", "agent one", `{"content":"response one","role":"assistant","token_usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"tool_calls":[]}`)
	runAtifLLMCall(t, stack2, "atif_llm_2", "agent two", `{"content":"response two","role":"assistant","token_usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9},"tool_calls":[]}`)

	allJSON, err := exporter.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON all failed: %v", err)
	}
	assertTrajectoryStepCount(t, allJSON, 4)

	if err := exporter.Deregister("go_atif_exporter"); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}

	exporter.Clear()
	emptyJSON, err := exporter.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON after clear failed: %v", err)
	}
	assertNoTrajectorySteps(t, emptyJSON, "clear")

	stack1.Run(func() {
		_, err := LlmCallExecute("atif_llm_after_deregister", map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"messages": []map[string]interface{}{{"role": "user", "content": "ignored"}},
				"model":    wrapperTestModel,
			},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"content":"ignored","role":"assistant","tool_calls":[]}`), nil
		})
		if err != nil {
			t.Fatalf("LlmCallExecute after deregister failed: %v", err)
		}
	})

	afterDeregisterJSON, err := exporter.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON after deregister failed: %v", err)
	}
	assertNoTrajectorySteps(t, afterDeregisterJSON, "deregister")

	exporter.Close()
	exporter.Close()
}
