// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package guardrails_test

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/guardrails"
)

func makeRequest() map[string]interface{} {
	return map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"messages": []interface{}{}, "model": "test-model"},
	}
}

func captureEndEventOutput(t *testing.T, subscriberName, eventName string) (func() json.RawMessage, func()) {
	t.Helper()

	var output json.RawMessage
	var mu sync.Mutex
	if err := nemo_relay.RegisterSubscriber(subscriberName, func(event nemo_relay.Event) {
		mu.Lock()
		defer mu.Unlock()
		if event.Kind() == "scope" && event.ScopeCategory() == "end" && event.Name() == eventName {
			output = append(json.RawMessage(nil), event.Output()...)
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}

	getOutput := func() json.RawMessage {
		mu.Lock()
		defer mu.Unlock()
		return append(json.RawMessage(nil), output...)
	}
	cleanup := func() {
		_ = nemo_relay.DeregisterSubscriber(subscriberName)
	}
	return getOutput, cleanup
}

func assertJSONFieldEquals(t *testing.T, raw json.RawMessage, field string, want interface{}) {
	t.Helper()

	var decoded map[string]interface{}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal JSON field %s: %v", field, err)
	}
	if decoded[field] != want {
		t.Fatalf("expected %s=%v, got %v", field, want, decoded)
	}
}

func runGlobalToolGuardrailShorthandChecks(t *testing.T, output func() json.RawMessage) {
	t.Helper()

	if err := guardrails.RegisterToolSanitizeRequest("guardrails_tool_req", 1,
		func(name string, args json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(args, &payload)
			payload["sanitized"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterToolSanitizeRequest failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterToolSanitizeRequest("guardrails_tool_req")
	})

	if err := guardrails.RegisterToolSanitizeResponse("guardrails_tool_resp", 1,
		func(name string, result json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(result, &payload)
			payload["guarded"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterToolSanitizeResponse failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterToolSanitizeResponse("guardrails_tool_resp")
	})

	if err := guardrails.RegisterToolConditionalExecution("guardrails_tool_cond", 1,
		func(name string, args json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterToolConditionalExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterToolConditionalExecution("guardrails_tool_cond")
	})

	if _, err := nemo_relay.ToolCallExecute("guardrails_tool", json.RawMessage(`{"value": 1}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	); err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}
	assertJSONFieldEquals(t, output(), "guarded", true)

	if err := guardrails.ToolConditionalExecution("guardrails_tool", json.RawMessage(`{"value": 1}`)); err != nil {
		t.Fatalf("ToolConditionalExecution failed: %v", err)
	}
}

func runGlobalLLMGuardrailShorthandChecks(t *testing.T, output func() json.RawMessage) {
	t.Helper()

	if err := guardrails.RegisterLlmSanitizeRequest("guardrails_llm_req", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			var payload map[string]interface{}
			_ = json.Unmarshal(content, &payload)
			payload["request_sanitized"] = true
			out, _ := json.Marshal(payload)
			return headers, out
		},
	); err != nil {
		t.Fatalf("RegisterLlmSanitizeRequest failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterLlmSanitizeRequest("guardrails_llm_req")
	})

	if err := guardrails.RegisterLlmSanitizeResponse("guardrails_llm_resp", 1,
		func(response json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(response, &payload)
			payload["guarded"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterLlmSanitizeResponse failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterLlmSanitizeResponse("guardrails_llm_resp")
	})

	if err := guardrails.RegisterLlmConditionalExecution("guardrails_llm_cond", 1,
		func(headers, content json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterLlmConditionalExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterLlmConditionalExecution("guardrails_llm_cond")
	})

	if _, err := nemo_relay.LlmCallExecute("guardrails_llm", makeRequest(),
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	); err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}
	assertJSONFieldEquals(t, output(), "guarded", true)

	if err := guardrails.LlmConditionalExecution(json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`)); err != nil {
		t.Fatalf("LlmConditionalExecution failed: %v", err)
	}
}

func runScopeLocalToolGuardrailShorthandChecks(t *testing.T, scopeUUID string) {
	t.Helper()

	if err := guardrails.ScopeRegisterToolSanitizeRequest(scopeUUID, "guardrails_scope_tool_req", 1,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	); err != nil {
		t.Fatalf("ScopeRegisterToolSanitizeRequest failed: %v", err)
	}
	if err := guardrails.ScopeRegisterToolSanitizeResponse(scopeUUID, "guardrails_scope_tool_resp", 1,
		func(name string, result json.RawMessage) json.RawMessage { return result },
	); err != nil {
		t.Fatalf("ScopeRegisterToolSanitizeResponse failed: %v", err)
	}
	if err := guardrails.ScopeRegisterToolConditionalExecution(scopeUUID, "guardrails_scope_tool_cond", 1,
		func(name string, args json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("ScopeRegisterToolConditionalExecution failed: %v", err)
	}

	if _, err := nemo_relay.ToolCallExecute("guardrails_scope_tool", json.RawMessage(`{"ok": true}`),
		func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
	); err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	if err := guardrails.ScopeDeregisterToolSanitizeRequest(scopeUUID, "guardrails_scope_tool_req"); err != nil {
		t.Fatalf("ScopeDeregisterToolSanitizeRequest failed: %v", err)
	}
	if err := guardrails.ScopeDeregisterToolSanitizeResponse(scopeUUID, "guardrails_scope_tool_resp"); err != nil {
		t.Fatalf("ScopeDeregisterToolSanitizeResponse failed: %v", err)
	}
	if err := guardrails.ScopeDeregisterToolConditionalExecution(scopeUUID, "guardrails_scope_tool_cond"); err != nil {
		t.Fatalf("ScopeDeregisterToolConditionalExecution failed: %v", err)
	}
}

func runScopeLocalLLMGuardrailShorthandChecks(t *testing.T, scopeUUID string) {
	t.Helper()

	if err := guardrails.ScopeRegisterLlmSanitizeRequest(scopeUUID, "guardrails_scope_llm_req", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			return headers, content
		},
	); err != nil {
		t.Fatalf("ScopeRegisterLlmSanitizeRequest failed: %v", err)
	}
	if err := guardrails.ScopeRegisterLlmSanitizeResponse(scopeUUID, "guardrails_scope_llm_resp", 1,
		func(response json.RawMessage) json.RawMessage { return response },
	); err != nil {
		t.Fatalf("ScopeRegisterLlmSanitizeResponse failed: %v", err)
	}
	if err := guardrails.ScopeRegisterLlmConditionalExecution(scopeUUID, "guardrails_scope_llm_cond", 1,
		func(headers, content json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("ScopeRegisterLlmConditionalExecution failed: %v", err)
	}

	if _, err := nemo_relay.LlmCallExecute("guardrails_scope_llm", makeRequest(),
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	); err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	if err := guardrails.ScopeDeregisterLlmSanitizeRequest(scopeUUID, "guardrails_scope_llm_req"); err != nil {
		t.Fatalf("ScopeDeregisterLlmSanitizeRequest failed: %v", err)
	}
	if err := guardrails.ScopeDeregisterLlmSanitizeResponse(scopeUUID, "guardrails_scope_llm_resp"); err != nil {
		t.Fatalf("ScopeDeregisterLlmSanitizeResponse failed: %v", err)
	}
	if err := guardrails.ScopeDeregisterLlmConditionalExecution(scopeUUID, "guardrails_scope_llm_cond"); err != nil {
		t.Fatalf("ScopeDeregisterLlmConditionalExecution failed: %v", err)
	}
}

func TestGuardrailShorthandsGlobal(t *testing.T) {
	toolEventOutput, cleanupTool := captureEndEventOutput(t, "guardrails_tool_events", "guardrails_tool")
	defer cleanupTool()
	runGlobalToolGuardrailShorthandChecks(t, toolEventOutput)

	llmEventOutput, cleanupLLM := captureEndEventOutput(t, "guardrails_llm_events", "guardrails_llm")
	defer cleanupLLM()
	runGlobalLLMGuardrailShorthandChecks(t, llmEventOutput)
}

func TestGuardrailShorthandsScopeLocal(t *testing.T) {
	stack, err := nemo_relay.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nemo_relay.PushScope("guardrails_scope", nemo_relay.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nemo_relay.PopScope(handle)

		scopeUUID := handle.UUID()
		runScopeLocalToolGuardrailShorthandChecks(t, scopeUUID)
		runScopeLocalLLMGuardrailShorthandChecks(t, scopeUUID)
	})
}
