// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nvagentrt

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func makeRequest() *LLMRequest {
	return NewLLMRequest("POST", "https://api.example.com",
		map[string]interface{}{}, map[string]interface{}{"messages": []string{}},
	)
}

// ============================================================================
// LLM lifecycle
// ============================================================================

func TestLlmCallAndEnd(t *testing.T) {
	req := makeRequest()
	handle, err := LlmCall("my_llm", req)
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
	req := makeRequest()
	handle, err := LlmCall("streaming_llm", req, WithLLMAttributes(LLMAttrStreaming))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle.Attributes()&LLMAttrStreaming == 0 {
		t.Fatal("expected STREAMING attribute")
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmCallWithDataMetadata(t *testing.T) {
	req := makeRequest()
	handle, err := LlmCall("llm_dm", req,
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

	req := makeRequest()
	handle, err := LlmCall("child_llm", req, WithLLMParent(parent))
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

	req := makeRequest()
	handle, _ := LlmCall("evt_llm", req)
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
	req := makeRequest()
	result, err := LlmCallExecute("exec_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			out, _ := json.Marshal(map[string]interface{}{"model": url})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["model"] != "https://api.example.com" {
		t.Fatalf("expected url, got %v", output["model"])
	}
}

// ============================================================================
// LLM guardrails
// ============================================================================

func TestLlmSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_san_req", 1,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_san_req")
}

func TestLlmSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeResponseGuardrail("go_llm_san_resp", 1,
		func(value json.RawMessage) json.RawMessage { return value },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeResponseGuardrail("go_llm_san_resp")
}

func TestLlmConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterLlmConditionalExecutionGuardrail("go_llm_cond", 1,
		func(method, url string, headers, body json.RawMessage) *string {
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
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
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
		func(method, url string, headers, body json.RawMessage) *string {
			return &msg
		},
	)

	req := makeRequest()
	_, err := LlmCallExecute("blocked_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
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
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmRequestIntercept("go_llm_req")
}

func TestLlmResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmResponseIntercept("go_llm_resp", 1, false,
		func(value json.RawMessage) json.RawMessage { return value },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmResponseIntercept("go_llm_resp")
}

func TestLlmStreamResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamResponseIntercept("go_llm_sr", 1, false,
		func(chunk string) string { return chunk },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamResponseIntercept("go_llm_sr")
}

func TestLlmExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmExecutionIntercept("go_llm_exec", 1,
		func(method, url string, headers, body json.RawMessage) bool { return false },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmExecutionIntercept("go_llm_exec")
}

func TestLlmStreamExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamExecutionIntercept("go_llm_sexec", 1,
		func(method, url string, headers, body json.RawMessage) bool { return false },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamExecutionIntercept("go_llm_sexec")
}

func TestLlmRequestInterceptModifies(t *testing.T) {
	RegisterLlmRequestIntercept("go_llm_req_mod", 1, false,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, "https://intercepted.com", headers, body
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("int_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			out, _ := json.Marshal(map[string]interface{}{"called_url": url})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["called_url"] != "https://intercepted.com" {
		t.Fatalf("expected intercepted URL, got %v", output["called_url"])
	}

	DeregisterLlmRequestIntercept("go_llm_req_mod")
}

func TestLlmResponseInterceptModifies(t *testing.T) {
	RegisterLlmResponseIntercept("go_llm_resp_mod", 1, false,
		func(value json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(value, &m)
			m["modified"] = true
			out, _ := json.Marshal(m)
			return out
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("resp_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
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
		func(method, url string, headers, body json.RawMessage) bool { return true },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("exec_llm_rep", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
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
