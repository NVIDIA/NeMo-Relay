// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package intercepts_test

import (
	"encoding/json"
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/intercepts"
)

func makeRequest() json.RawMessage {
	return json.RawMessage(`{"headers":{},"content":{"messages":[],"model":"test-model"}}`)
}

func TestInterceptShorthandsGlobal(t *testing.T) {
	if err := intercepts.RegisterToolRequest("intercepts_tool_req", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(args, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterToolRequest failed: %v", err)
	}
	t.Cleanup(func() {
		_ = intercepts.DeregisterToolRequest("intercepts_tool_req")
	})

	transformedArgs, err := intercepts.ToolRequestIntercepts("tool", json.RawMessage(`{"value": 1}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts failed: %v", err)
	}

	var toolArgs map[string]interface{}
	if err := json.Unmarshal(transformedArgs, &toolArgs); err != nil {
		t.Fatalf("unmarshal tool args: %v", err)
	}
	if toolArgs["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", toolArgs)
	}

	if err := intercepts.RegisterToolExecution("intercepts_tool_exec", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(args)
			if err != nil {
				return nil, err
			}
			var payload map[string]interface{}
			_ = json.Unmarshal(result, &payload)
			payload["wrapped"] = true
			out, _ := json.Marshal(payload)
			return out, nil
		},
	); err != nil {
		t.Fatalf("RegisterToolExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = intercepts.DeregisterToolExecution("intercepts_tool_exec")
	})

	result, err := nat_nexus.ToolCallExecute("intercepts_tool", json.RawMessage(`{"value": 1}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var toolResult map[string]interface{}
	if err := json.Unmarshal(result, &toolResult); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if toolResult["wrapped"] != true {
		t.Fatalf("expected wrapped=true, got %v", toolResult)
	}

	if err := intercepts.RegisterLlmRequest("intercepts_llm_req", 1, false,
		func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
			var payload map[string]interface{}
			_ = json.Unmarshal(content, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return headers, out, annotated, nil
		},
	); err != nil {
		t.Fatalf("RegisterLlmRequest failed: %v", err)
	}
	t.Cleanup(func() {
		_ = intercepts.DeregisterLlmRequest("intercepts_llm_req")
	})

	transformedRequest, err := intercepts.LlmRequestIntercepts("llm", makeRequest())
	if err != nil {
		t.Fatalf("LlmRequestIntercepts failed: %v", err)
	}

	var llmReq struct {
		Content map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(transformedRequest, &llmReq); err != nil {
		t.Fatalf("unmarshal llm request: %v", err)
	}
	if llmReq.Content["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", llmReq.Content)
	}

	if err := intercepts.RegisterLlmExecution("intercepts_llm_exec", 1,
		func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(request)
			if err != nil {
				return nil, err
			}
			var payload map[string]interface{}
			_ = json.Unmarshal(result, &payload)
			payload["wrapped"] = true
			out, _ := json.Marshal(payload)
			return out, nil
		},
	); err != nil {
		t.Fatalf("RegisterLlmExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = intercepts.DeregisterLlmExecution("intercepts_llm_exec")
	})

	response, err := nat_nexus.LlmCallExecute("intercepts_llm", map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"messages": []interface{}{}, "model": "test-model"},
	}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"ok": true}`), nil
	})
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	var llmResult map[string]interface{}
	if err := json.Unmarshal(response, &llmResult); err != nil {
		t.Fatalf("unmarshal llm result: %v", err)
	}
	if llmResult["wrapped"] != true {
		t.Fatalf("expected wrapped=true, got %v", llmResult)
	}

	if err := intercepts.RegisterLlmStreamExecution("intercepts_llm_stream", 1,
		func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(request)
		},
	); err != nil {
		t.Fatalf("RegisterLlmStreamExecution failed: %v", err)
	}
	t.Cleanup(func() {
		_ = intercepts.DeregisterLlmStreamExecution("intercepts_llm_stream")
	})
}

func TestInterceptShorthandsScopeLocal(t *testing.T) {
	stack, err := nat_nexus.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nat_nexus.PushScope("intercepts_scope", nat_nexus.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nat_nexus.PopScope(handle)

		scopeUUID := handle.UUID()

		if err := intercepts.ScopeRegisterToolRequest(scopeUUID, "intercepts_scope_tool_req", 1, false,
			func(name string, args json.RawMessage) json.RawMessage { return args },
		); err != nil {
			t.Fatalf("ScopeRegisterToolRequest failed: %v", err)
		}
		if err := intercepts.ScopeRegisterToolExecution(scopeUUID, "intercepts_scope_tool_exec", 1,
			func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				return next(args)
			},
		); err != nil {
			t.Fatalf("ScopeRegisterToolExecution failed: %v", err)
		}
		if _, err := nat_nexus.ToolCallExecute("intercepts_scope_tool", json.RawMessage(`{"ok": true}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
		}

		if err := intercepts.ScopeRegisterLlmRequest(scopeUUID, "intercepts_scope_llm_req", 1, false,
			func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
				return headers, content, annotated, nil
			},
		); err != nil {
			t.Fatalf("ScopeRegisterLlmRequest failed: %v", err)
		}
		if err := intercepts.ScopeRegisterLlmExecution(scopeUUID, "intercepts_scope_llm_exec", 1,
			func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				return next(request)
			},
		); err != nil {
			t.Fatalf("ScopeRegisterLlmExecution failed: %v", err)
		}
		if err := intercepts.ScopeRegisterLlmStreamExecution(scopeUUID, "intercepts_scope_llm_stream", 1,
			func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				return next(request)
			},
		); err != nil {
			t.Fatalf("ScopeRegisterLlmStreamExecution failed: %v", err)
		}
		if _, err := nat_nexus.LlmCallExecute("intercepts_scope_llm", map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{"messages": []interface{}{}, "model": "test-model"},
		}, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		}); err != nil {
			t.Fatalf("LlmCallExecute failed: %v", err)
		}

		if err := intercepts.ScopeDeregisterToolRequest(scopeUUID, "intercepts_scope_tool_req"); err != nil {
			t.Fatalf("ScopeDeregisterToolRequest failed: %v", err)
		}
		if err := intercepts.ScopeDeregisterToolExecution(scopeUUID, "intercepts_scope_tool_exec"); err != nil {
			t.Fatalf("ScopeDeregisterToolExecution failed: %v", err)
		}
		if err := intercepts.ScopeDeregisterLlmRequest(scopeUUID, "intercepts_scope_llm_req"); err != nil {
			t.Fatalf("ScopeDeregisterLlmRequest failed: %v", err)
		}
		if err := intercepts.ScopeDeregisterLlmExecution(scopeUUID, "intercepts_scope_llm_exec"); err != nil {
			t.Fatalf("ScopeDeregisterLlmExecution failed: %v", err)
		}
		if err := intercepts.ScopeDeregisterLlmStreamExecution(scopeUUID, "intercepts_scope_llm_stream"); err != nil {
			t.Fatalf("ScopeDeregisterLlmStreamExecution failed: %v", err)
		}
	})
}
