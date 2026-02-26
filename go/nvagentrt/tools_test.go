// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nvagentrt

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// Tool lifecycle
// ============================================================================

func TestToolCallAndEnd(t *testing.T) {
	handle, err := ToolCall("my_tool", json.RawMessage(`{"input": "data"}`))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle == nil {
		t.Fatal("returned nil handle")
	}
	if handle.Name() != "my_tool" {
		t.Fatalf("expected 'my_tool', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}

	err = ToolCallEnd(handle, json.RawMessage(`{"output": "result"}`))
	if err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}
}

func TestToolCallWithAttributes(t *testing.T) {
	handle, err := ToolCall("local_tool", json.RawMessage(`{}`), WithToolAttributes(ToolAttrLocal))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle.Attributes()&ToolAttrLocal == 0 {
		t.Fatal("expected LOCAL attribute")
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolCallWithDataMetadata(t *testing.T) {
	handle, err := ToolCall("tool_dm", json.RawMessage(`{"arg": 1}`),
		WithToolData(json.RawMessage(`{"custom": "info"}`)),
		WithToolMetadata(json.RawMessage(`{"trace_id": "abc123"}`)),
	)
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	ToolCallEnd(handle, json.RawMessage(`{}`),
		WithToolData(json.RawMessage(`{"end_data": true}`)),
		WithToolMetadata(json.RawMessage(`{"end_meta": true}`)),
	)
}

func TestToolCallWithParent(t *testing.T) {
	parent, _ := PushScope("tool_parent", ScopeTypeAgent)
	defer PopScope(parent)

	handle, err := ToolCall("child_tool", json.RawMessage(`{}`), WithToolParent(parent))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_tool_evt", func(event *Event) {
		mu.Lock()
		if event.Type() == EventTypeStart {
			startSeen = true
		}
		if event.Type() == EventTypeEnd {
			endSeen = true
		}
		mu.Unlock()
	})

	handle, _ := ToolCall("evt_tool", json.RawMessage(`{}`))
	ToolCallEnd(handle, json.RawMessage(`{}`))
	DeregisterSubscriber("go_tool_evt")

	mu.Lock()
	if !startSeen {
		t.Fatal("start event not seen")
	}
	if !endSeen {
		t.Fatal("end event not seen")
	}
	mu.Unlock()
}

// ============================================================================
// Tool execute
// ============================================================================

func TestToolCallExecuteBasic(t *testing.T) {
	fn := func(args json.RawMessage) (json.RawMessage, error) {
		var input map[string]interface{}
		json.Unmarshal(args, &input)
		x := input["x"].(float64)
		result, _ := json.Marshal(map[string]interface{}{"result": x * 2})
		return result, nil
	}

	result, err := ToolCallExecute("double", json.RawMessage(`{"x": 5}`), fn)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["result"].(float64) != 10 {
		t.Fatalf("expected 10, got %v", output["result"])
	}
}

func TestToolCallExecuteWithAttributes(t *testing.T) {
	fn := func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	}

	result, err := ToolCallExecute("attr_tool", json.RawMessage(`{"test": true}`), fn,
		WithToolAttributes(ToolAttrLocal),
	)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["test"] != true {
		t.Fatalf("expected test=true, got %v", output["test"])
	}
}

// ============================================================================
// Tool guardrails
// ============================================================================

func TestToolSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterToolSanitizeRequestGuardrail("go_san_req", 1,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["sanitized"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolSanitizeRequestGuardrail("go_san_req")
}

func TestToolSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterToolSanitizeResponseGuardrail("go_san_resp", 1,
		func(name string, result json.RawMessage) json.RawMessage {
			return result
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolSanitizeResponseGuardrail("go_san_resp")
}

func TestToolConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterToolConditionalExecutionGuardrail("go_cond", 1,
		func(name string, args json.RawMessage) *string {
			return nil // pass
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolConditionalExecutionGuardrail("go_cond")
}

func TestDuplicateGuardrailFails(t *testing.T) {
	RegisterToolSanitizeRequestGuardrail("go_dup_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	err := RegisterToolSanitizeRequestGuardrail("go_dup_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err == nil {
		t.Fatal("expected error for duplicate guardrail")
	}
	DeregisterToolSanitizeRequestGuardrail("go_dup_guard")
}

func TestToolConditionalBlocksExecution(t *testing.T) {
	msg := "blocked by policy"
	RegisterToolConditionalExecutionGuardrail("go_blocker", 1,
		func(name string, args json.RawMessage) *string {
			return &msg
		},
	)

	_, err := ToolCallExecute("blocked_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"should": "not reach"}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected error from guardrail rejection")
	}
	if !strings.Contains(err.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' error, got: %v", err)
	}

	DeregisterToolConditionalExecutionGuardrail("go_blocker")
}

// ============================================================================
// Tool intercepts
// ============================================================================

func TestToolRequestInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolRequestIntercept("go_req_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolRequestIntercept("go_req_int")
}

func TestToolResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolResponseIntercept("go_resp_int", 1, false,
		func(name string, result json.RawMessage) json.RawMessage { return result },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolResponseIntercept("go_resp_int")
}

func TestToolExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolExecutionIntercept("go_exec_int", 1,
		func(name string, args json.RawMessage) bool { return false },
		func(args json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{}`), nil },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolExecutionIntercept("go_exec_int")
}

func TestDuplicateInterceptFails(t *testing.T) {
	RegisterToolRequestIntercept("go_dup_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	err := RegisterToolRequestIntercept("go_dup_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err == nil {
		t.Fatal("expected error for duplicate intercept")
	}
	DeregisterToolRequestIntercept("go_dup_int")
}

func TestToolRequestInterceptModifiesArgs(t *testing.T) {
	RegisterToolRequestIntercept("go_req_mod", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["intercepted"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)

	result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"original": true}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true || output["intercepted"] != true {
		t.Fatalf("expected both original and intercepted, got %v", output)
	}

	DeregisterToolRequestIntercept("go_req_mod")
}

func TestToolResponseInterceptModifiesResult(t *testing.T) {
	RegisterToolResponseIntercept("go_resp_mod", 1, false,
		func(name string, result json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["post_processed"] = true
			out, _ := json.Marshal(m)
			return out
		},
	)

	result, err := ToolCallExecute("resp_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"output": "raw"}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["output"] != "raw" || output["post_processed"] != true {
		t.Fatalf("expected output + post_processed, got %v", output)
	}

	DeregisterToolResponseIntercept("go_resp_mod")
}

func TestToolExecutionInterceptReplacesFunc(t *testing.T) {
	RegisterToolExecutionIntercept("go_exec_replace", 1,
		func(name string, args json.RawMessage) bool { return true },
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	result, err := ToolCallExecute("replaced_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["from_intercept"] != true {
		t.Fatalf("expected from_intercept, got %v", output)
	}
	if _, ok := output["from_original"]; ok {
		t.Fatal("should not contain from_original")
	}

	DeregisterToolExecutionIntercept("go_exec_replace")
}

func TestToolRequestInterceptBreakChain(t *testing.T) {
	RegisterToolRequestIntercept("go_chain1", 1, true, // break_chain=true
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["from_first"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	RegisterToolRequestIntercept("go_chain2", 2, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["from_second"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)

	result, err := ToolCallExecute("chain_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["from_first"] != true {
		t.Fatal("expected from_first")
	}
	if _, ok := output["from_second"]; ok {
		t.Fatal("should not contain from_second (chain was broken)")
	}

	DeregisterToolRequestIntercept("go_chain1")
	DeregisterToolRequestIntercept("go_chain2")
}
