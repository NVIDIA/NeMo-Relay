// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package guardrails_test

import (
	"encoding/json"
	"sync"
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/guardrails"
)

func makeRequest() map[string]interface{} {
	return map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"messages": []interface{}{}, "model": "test-model"},
	}
}

func TestGuardrailShorthandsGlobal(t *testing.T) {
	var toolEventOutput json.RawMessage
	var llmEventOutput json.RawMessage
	var mu sync.Mutex

	if err := nat_nexus.RegisterSubscriber("guardrails_events", func(event nat_nexus.Event) {
		mu.Lock()
		defer mu.Unlock()
		if event.Kind() != "ToolEnd" && event.Kind() != "LLMEnd" {
			return
		}
		switch event.Name() {
		case "guardrails_tool":
			toolEventOutput = append(json.RawMessage(nil), event.Output()...)
		case "guardrails_llm":
			llmEventOutput = append(json.RawMessage(nil), event.Output()...)
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	t.Cleanup(func() {
		_ = nat_nexus.DeregisterSubscriber("guardrails_events")
	})

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
		func(name string, args json.RawMessage) *string {
			return nil
		},
	); err != nil {
		t.Fatalf("RegisterToolConditionalExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterToolConditionalExecution("guardrails_tool_cond")
	})

	_, err := nat_nexus.ToolCallExecute("guardrails_tool", json.RawMessage(`{"value": 1}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var toolOut map[string]interface{}
	if err := json.Unmarshal(toolEventOutput, &toolOut); err != nil {
		t.Fatalf("unmarshal tool event output: %v", err)
	}
	if toolOut["guarded"] != true {
		t.Fatalf("expected guarded=true, got %v", toolOut)
	}

	if err := guardrails.ToolConditionalExecution("guardrails_tool", json.RawMessage(`{"value": 1}`)); err != nil {
		t.Fatalf("ToolConditionalExecution failed: %v", err)
	}

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
		func(headers, content json.RawMessage) *string {
			return nil
		},
	); err != nil {
		t.Fatalf("RegisterLlmConditionalExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = guardrails.DeregisterLlmConditionalExecution("guardrails_llm_cond")
	})

	_, err = nat_nexus.LlmCallExecute("guardrails_llm", makeRequest(),
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	var llmOut map[string]interface{}
	if err := json.Unmarshal(llmEventOutput, &llmOut); err != nil {
		t.Fatalf("unmarshal llm event output: %v", err)
	}
	if llmOut["guarded"] != true {
		t.Fatalf("expected guarded=true, got %v", llmOut)
	}

	if err := guardrails.LlmConditionalExecution(json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`)); err != nil {
		t.Fatalf("LlmConditionalExecution failed: %v", err)
	}
}

func TestGuardrailShorthandsScopeLocal(t *testing.T) {
	stack, err := nat_nexus.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nat_nexus.PushScope("guardrails_scope", nat_nexus.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nat_nexus.PopScope(handle)

		scopeUUID := handle.UUID()

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

		if _, err := nat_nexus.ToolCallExecute("guardrails_scope_tool", json.RawMessage(`{"ok": true}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
		}

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

		if _, err := nat_nexus.LlmCallExecute("guardrails_scope_llm", makeRequest(),
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok": true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute failed: %v", err)
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
		if err := guardrails.ScopeDeregisterLlmSanitizeRequest(scopeUUID, "guardrails_scope_llm_req"); err != nil {
			t.Fatalf("ScopeDeregisterLlmSanitizeRequest failed: %v", err)
		}
		if err := guardrails.ScopeDeregisterLlmSanitizeResponse(scopeUUID, "guardrails_scope_llm_resp"); err != nil {
			t.Fatalf("ScopeDeregisterLlmSanitizeResponse failed: %v", err)
		}
		if err := guardrails.ScopeDeregisterLlmConditionalExecution(scopeUUID, "guardrails_scope_llm_cond"); err != nil {
			t.Fatalf("ScopeDeregisterLlmConditionalExecution failed: %v", err)
		}
	})
}
