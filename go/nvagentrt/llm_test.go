// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nvagentrt

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func makeNative() map[string]interface{} {
	return map[string]interface{}{"messages": []string{}}
}

// ============================================================================
// LLM lifecycle
// ============================================================================

func TestLlmCallAndEnd(t *testing.T) {
	native := makeNative()
	handle, err := LlmCall("my_llm", native)
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle == nil {
		t.Fatal("returned nil handle")
	}
	if handle.Name() != "my_llm" {
		t.Fatalf("expected 'my_llm', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}

	err = LlmCallEnd(handle, json.RawMessage(`{"response": "ok"}`))
	if err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}
}

func TestLlmCallWithAttributes(t *testing.T) {
	native := makeNative()
	handle, err := LlmCall("streaming_llm", native, WithLLMAttributes(LLMAttrStreaming))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle.Attributes()&LLMAttrStreaming == 0 {
		t.Fatal("expected STREAMING attribute")
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmCallWithDataMetadata(t *testing.T) {
	native := makeNative()
	handle, err := LlmCall("llm_dm", native,
		WithLLMData(json.RawMessage(`{"custom": "data"}`)),
		WithLLMMetadata(json.RawMessage(`{"trace": "xyz"}`)),
	)
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	LlmCallEnd(handle, json.RawMessage(`{}`),
		WithLLMData(json.RawMessage(`{"end": true}`)),
	)
}

func TestLlmCallWithParent(t *testing.T) {
	parent, _ := PushScope("llm_parent", ScopeTypeAgent)
	defer PopScope(parent)

	native := makeNative()
	handle, err := LlmCall("child_llm", native, WithLLMParent(parent))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_llm_evt", func(event *Event) {
		mu.Lock()
		if event.Type() == EventTypeStart {
			startSeen = true
		}
		if event.Type() == EventTypeEnd {
			endSeen = true
		}
		mu.Unlock()
	})

	native := makeNative()
	handle, _ := LlmCall("evt_llm", native)
	LlmCallEnd(handle, json.RawMessage(`{}`))
	DeregisterSubscriber("go_llm_evt")

	mu.Lock()
	if !startSeen || !endSeen {
		t.Fatal("expected both start and end events")
	}
	mu.Unlock()
}

// ============================================================================
// LLM execute
// ============================================================================

func TestLlmCallExecuteBasic(t *testing.T) {
	native := makeNative()
	result, err := LlmCallExecute("exec_llm", native,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			var input map[string]interface{}
			json.Unmarshal(nativeJSON, &input)
			out, _ := json.Marshal(map[string]interface{}{"received": true})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["received"] != true {
		t.Fatalf("expected received=true, got %v", output)
	}
}

// ============================================================================
// LLM guardrails
// ============================================================================

func TestLlmSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_san_req", 1,
		func(value json.RawMessage) json.RawMessage {
			return value
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_san_req")
}

func TestLlmSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeResponseGuardrail("go_llm_san_resp", 1,
		func(responseJSON json.RawMessage) json.RawMessage { return responseJSON },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeResponseGuardrail("go_llm_san_resp")
}

func TestLlmConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterLlmConditionalExecutionGuardrail("go_llm_cond", 1,
		func(headers, content json.RawMessage) *string {
			return nil // pass
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmConditionalExecutionGuardrail("go_llm_cond")
}

func TestLlmDuplicateGuardrailFails(t *testing.T) {
	RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(value json.RawMessage) json.RawMessage {
			return value
		},
	)
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(value json.RawMessage) json.RawMessage {
			return value
		},
	)
	if err == nil {
		t.Fatal("expected error for duplicate")
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_dup")
}

func TestLlmConditionalBlocksExecution(t *testing.T) {
	msg := "LLM blocked"
	RegisterLlmConditionalExecutionGuardrail("go_llm_blocker", 1,
		func(headers, content json.RawMessage) *string {
			return &msg
		},
	)

	native := makeNative()
	_, err := LlmCallExecute("blocked_llm", native,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"should": "not reach"}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected error from guardrail rejection")
	}
	if !strings.Contains(err.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' error, got: %v", err)
	}

	DeregisterLlmConditionalExecutionGuardrail("go_llm_blocker")
}

// ============================================================================
// LLM intercepts
// ============================================================================

func TestLlmRequestInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmRequestIntercept("go_llm_req", 1, false,
		func(value json.RawMessage) json.RawMessage {
			return value
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmRequestIntercept("go_llm_req")
}

func TestLlmResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmResponseIntercept("go_llm_resp", 1, false,
		func(responseJSON json.RawMessage) json.RawMessage { return responseJSON },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmResponseIntercept("go_llm_resp")
}

func TestLlmStreamResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamResponseIntercept("go_llm_sr", 1, false,
		func(chunkJSON json.RawMessage) json.RawMessage { return chunkJSON },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamResponseIntercept("go_llm_sr")
}

func TestLlmExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmExecutionIntercept("go_llm_exec", 1,
		func(nativeJSON json.RawMessage) bool { return false },
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(nativeJSON)
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmExecutionIntercept("go_llm_exec")
}

func TestLlmStreamExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamExecutionIntercept("go_llm_sexec", 1,
		func(nativeJSON json.RawMessage) bool { return false },
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(nativeJSON)
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamExecutionIntercept("go_llm_sexec")
}

func TestLlmRequestInterceptModifies(t *testing.T) {
	RegisterLlmRequestIntercept("go_llm_req_mod", 1, false,
		func(value json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(value, &m)
			m["intercepted"] = true
			out, _ := json.Marshal(m)
			return out
		},
	)

	native := makeNative()
	result, err := LlmCallExecute("int_llm", native,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			// The native JSON should now contain the intercepted field
			var m map[string]interface{}
			json.Unmarshal(nativeJSON, &m)
			out, _ := json.Marshal(map[string]interface{}{"saw_intercepted": m["intercepted"]})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["saw_intercepted"] != true {
		t.Fatalf("expected saw_intercepted=true, got %v", output)
	}

	DeregisterLlmRequestIntercept("go_llm_req_mod")
}

func TestLlmResponseInterceptModifies(t *testing.T) {
	RegisterLlmResponseIntercept("go_llm_resp_mod", 1, false,
		func(responseJSON json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(responseJSON, &m)
			// responseJSON is {"data": ...}, modify the data
			if data, ok := m["data"].(map[string]interface{}); ok {
				data["modified"] = true
				m["data"] = data
			}
			out, _ := json.Marshal(m)
			return out
		},
	)

	native := makeNative()
	result, err := LlmCallExecute("resp_llm", native,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true || output["modified"] != true {
		t.Fatalf("expected both original and modified, got %v", output)
	}

	DeregisterLlmResponseIntercept("go_llm_resp_mod")
}

func TestLlmExecutionInterceptReplaces(t *testing.T) {
	RegisterLlmExecutionIntercept("go_llm_exec_rep", 1,
		func(nativeJSON json.RawMessage) bool { return true },
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			// Short-circuit: don't call next, return directly
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	native := makeNative()
	result, err := LlmCallExecute("exec_llm_rep", native,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
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

	DeregisterLlmExecutionIntercept("go_llm_exec_rep")
}
