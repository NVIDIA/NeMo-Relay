// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

const (
	toolCallFailed        = "ToolCall failed: %v"
	toolCallExecuteFailed = "ToolCallExecute failed: %v"
	registerFailed        = "register failed: %v"
	executeFailed         = "execute failed: %v"
)

// ============================================================================
// Tool lifecycle
// ============================================================================

func TestToolCallAndEnd(t *testing.T) {
	handle, err := ToolCall("my_tool", json.RawMessage(`{"input": "data"}`))
	if err != nil {
		t.Fatalf(toolCallFailed, err)
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
	handle, err := ToolCall("remote_tool", json.RawMessage(`{}`), WithToolAttributes(ToolAttrRemote))
	if err != nil {
		t.Fatalf(toolCallFailed, err)
	}
	if handle.Attributes()&ToolAttrRemote == 0 {
		t.Fatal("expected REMOTE attribute")
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolCallWithDataMetadata(t *testing.T) {
	handle, err := ToolCall("tool_dm", json.RawMessage(`{"arg": 1}`),
		WithToolData(json.RawMessage(`{"custom": "info"}`)),
		WithToolMetadata(json.RawMessage(`{"trace_id": "abc123"}`)),
	)
	if err != nil {
		t.Fatalf(toolCallFailed, err)
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
		t.Fatalf(toolCallFailed, err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_tool_evt", func(event Event) {
		mu.Lock()
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "start" {
			startSeen = true
		}
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "end" {
			endSeen = true
		}
		mu.Unlock()
	})
	defer func() { _ = DeregisterSubscriber("go_tool_evt") }()

	handle, _ := ToolCall("evt_tool", json.RawMessage(`{}`))
	ToolCallEnd(handle, json.RawMessage(`{}`))
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

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
		t.Fatalf(toolCallExecuteFailed, err)
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
		WithToolAttributes(ToolAttrRemote),
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
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
		t.Fatalf(registerFailed, err)
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
		t.Fatalf(registerFailed, err)
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
		t.Fatalf(registerFailed, err)
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
		t.Fatalf(registerFailed, err)
	}
	DeregisterToolRequestIntercept("go_req_int")
}

func TestToolExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolExecutionIntercept("go_exec_int", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(args)
		},
	)
	if err != nil {
		t.Fatalf(registerFailed, err)
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
		t.Fatalf(executeFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true || output["intercepted"] != true {
		t.Fatalf("expected both original and intercepted, got %v", output)
	}

	DeregisterToolRequestIntercept("go_req_mod")
}

func TestToolExecutionInterceptReplacesFunc(t *testing.T) {
	RegisterToolExecutionIntercept("go_exec_replace", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			// Short-circuit: don't call next, return directly
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	result, err := ToolCallExecute("replaced_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf(executeFailed, err)
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
		t.Fatalf(executeFailed, err)
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

// ============================================================================
// Full tool pipeline tests (intercepts + execute)
// ============================================================================

func TestToolFullPipelineInterceptsAndExecute(t *testing.T) {
	// Register a request intercept that adds a flag
	RegisterToolRequestIntercept("go_pipe_req_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["request_intercepted"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	defer DeregisterToolRequestIntercept("go_pipe_req_int")

	// Register an execution intercept that wraps the callable
	RegisterToolExecutionIntercept("go_pipe_exec_int", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(args)
			if err != nil {
				return nil, err
			}
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["exec_intercepted"] = true
			out, _ := json.Marshal(m)
			return out, nil
		},
	)
	defer DeregisterToolExecutionIntercept("go_pipe_exec_int")

	// Execute a tool call through the full pipeline
	result, err := ToolCallExecute("pipeline_tool", json.RawMessage(`{"input": "value"}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			// The tool callable receives args after request intercepts
			return args, nil
		},
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)

	// Verify all pipeline stages affected the result
	if output["input"] != "value" {
		t.Fatalf("expected input=value, got %v", output["input"])
	}
	if output["request_intercepted"] != true {
		t.Fatal("expected request_intercepted=true from request intercept")
	}
	if output["exec_intercepted"] != true {
		t.Fatal("expected exec_intercepted=true from execution intercept")
	}
}

func TestToolSanitizeRequestGuardrailModifiesEventInput(t *testing.T) {
	// Sanitize-request guardrails modify the event input (observable through subscribers),
	// not the actual args passed to the tool callable.
	var capturedInput json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_redact_sub", func(event Event) {
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "start" {
			mu.Lock()
			capturedInput = append(json.RawMessage(nil), event.Input()...)
			mu.Unlock()
		}
	})
	defer DeregisterSubscriber("go_redact_sub")

	RegisterToolSanitizeRequestGuardrail("go_redact_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			if _, ok := m["password"]; ok {
				m["password"] = "REDACTED"
			}
			result, _ := json.Marshal(m)
			return result
		},
	)
	defer DeregisterToolSanitizeRequestGuardrail("go_redact_guard")

	_, err := ToolCallExecute("redact_tool", json.RawMessage(`{"user": "alice", "password": "secret123"}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"done": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if capturedInput == nil {
		t.Fatal("expected non-nil captured input from event")
	}
	var input map[string]interface{}
	json.Unmarshal(capturedInput, &input)
	if input["password"] != "REDACTED" {
		t.Fatalf("expected password=REDACTED in event input, got %v", input["password"])
	}
	if input["user"] != "alice" {
		t.Fatalf("expected user=alice in event input, got %v", input["user"])
	}
}

func TestToolConditionalGuardrailSelectiveReject(t *testing.T) {
	// Register a guardrail that blocks based on tool name
	RegisterToolConditionalExecutionGuardrail("go_selective_cond", 1,
		func(name string, args json.RawMessage) *string {
			if name == "dangerous_tool" {
				msg := "tool 'dangerous_tool' is not allowed"
				return &msg
			}
			return nil
		},
	)
	defer DeregisterToolConditionalExecutionGuardrail("go_selective_cond")

	// The dangerous tool should be blocked
	_, err := ToolCallExecute("dangerous_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ran": true}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected dangerous_tool to be blocked")
	}
	if !strings.Contains(err.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected', got: %v", err)
	}

	// The safe tool should succeed
	result, err := ToolCallExecute("safe_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ran": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("safe_tool should succeed: %v", err)
	}
	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["ran"] != true {
		t.Fatalf("expected ran=true, got %v", output)
	}
}

func TestToolMultipleGuardrailsPriorityOrder(t *testing.T) {
	var order []string
	var mu sync.Mutex

	RegisterToolSanitizeRequestGuardrail("go_prio_guard_1", 10,
		func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "p10")
			mu.Unlock()
			return args
		},
	)
	defer DeregisterToolSanitizeRequestGuardrail("go_prio_guard_1")

	RegisterToolSanitizeRequestGuardrail("go_prio_guard_2", 5,
		func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "p5")
			mu.Unlock()
			return args
		},
	)
	defer DeregisterToolSanitizeRequestGuardrail("go_prio_guard_2")

	RegisterToolSanitizeRequestGuardrail("go_prio_guard_3", 20,
		func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "p20")
			mu.Unlock()
			return args
		},
	)
	defer DeregisterToolSanitizeRequestGuardrail("go_prio_guard_3")

	_, err := ToolCallExecute("prio_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(order) != 3 {
		t.Fatalf("expected 3 guardrail executions, got %d", len(order))
	}
	if order[0] != "p5" {
		t.Fatalf("expected p5 first, got %s", order[0])
	}
	if order[1] != "p10" {
		t.Fatalf("expected p10 second, got %s", order[1])
	}
	if order[2] != "p20" {
		t.Fatalf("expected p20 third, got %s", order[2])
	}
}

func TestToolCallableErrorPropagation(t *testing.T) {
	_, err := ToolCallExecute("error_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("tool internal failure")
		},
	)
	if err == nil {
		t.Fatal("expected tool callable error to propagate")
	}
	if !strings.Contains(err.Error(), "tool internal failure") {
		t.Fatalf("expected propagated tool error message, got %v", err)
	}
}

func TestToolExecutionInterceptWrapsCallable(t *testing.T) {
	// Register an execution intercept that modifies args and result
	RegisterToolExecutionIntercept("go_wrap_exec_int", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			// Before: modify args
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["before_exec"] = true
			modifiedArgs, _ := json.Marshal(m)

			// Call the next function in the chain
			result, err := next(modifiedArgs)
			if err != nil {
				return nil, err
			}

			// After: modify result
			var out map[string]interface{}
			json.Unmarshal(result, &out)
			out["after_exec"] = true
			final, _ := json.Marshal(out)
			return final, nil
		},
	)
	defer DeregisterToolExecutionIntercept("go_wrap_exec_int")

	result, err := ToolCallExecute("wrap_tool", json.RawMessage(`{"input": 1}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			// The tool callable should see the modified args
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["tool_ran"] = true
			out, _ := json.Marshal(m)
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["before_exec"] != true {
		t.Fatal("expected before_exec=true")
	}
	if output["tool_ran"] != true {
		t.Fatal("expected tool_ran=true")
	}
	if output["after_exec"] != true {
		t.Fatal("expected after_exec=true")
	}
}

func TestToolExecutionInterceptSeesNextError(t *testing.T) {
	RegisterToolExecutionIntercept("go_wrap_exec_err", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(args)
		},
	)
	defer DeregisterToolExecutionIntercept("go_wrap_exec_err")

	_, err := ToolCallExecute("wrap_tool_err", json.RawMessage(`{"input": 1}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("tool next failure")
		},
	)
	if err == nil {
		t.Fatal("expected tool next error to propagate through intercept")
	}
	if !strings.Contains(err.Error(), "tool next failure") {
		t.Fatalf("expected propagated tool next error message, got %v", err)
	}
}

func TestToolCallWithToolCallID(t *testing.T) {
	var capturedToolCallID string
	var mu sync.Mutex

	RegisterSubscriber("go_tool_call_id_sub", func(event Event) {
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "start" {
			mu.Lock()
			capturedToolCallID = event.ToolCallID()
			mu.Unlock()
		}
	})
	defer func() { _ = DeregisterSubscriber("go_tool_call_id_sub") }()

	handle, err := ToolCall("id_tool", json.RawMessage(`{}`), WithToolCallID("call_abc_123"))
	if err != nil {
		t.Fatalf(toolCallFailed, err)
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if capturedToolCallID != "call_abc_123" {
		t.Fatalf("expected tool_call_id='call_abc_123', got '%s'", capturedToolCallID)
	}
}

func TestToolEventInputOutput(t *testing.T) {
	var capturedInput, capturedOutput json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_tool_io_sub", func(event Event) {
		mu.Lock()
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "start" {
			capturedInput = append(json.RawMessage(nil), event.Input()...)
		}
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "end" {
			capturedOutput = append(json.RawMessage(nil), event.Output()...)
		}
		mu.Unlock()
	})
	defer func() { _ = DeregisterSubscriber("go_tool_io_sub") }()

	_, err := ToolCallExecute("io_tool", json.RawMessage(`{"query": "test"}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"answer": "result"}`), nil
		},
	)
	if err != nil {
		t.Fatalf(toolCallExecuteFailed, err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if capturedInput == nil {
		t.Fatal("expected non-nil input on Start event")
	}
	var input map[string]interface{}
	json.Unmarshal(capturedInput, &input)
	if input["query"] != "test" {
		t.Fatalf("expected query=test in input, got %v", input)
	}

	if capturedOutput == nil {
		t.Fatal("expected non-nil output on End event")
	}
	var output map[string]interface{}
	json.Unmarshal(capturedOutput, &output)
	if output["answer"] != "result" {
		t.Fatalf("expected answer=result in output, got %v", output)
	}
}
