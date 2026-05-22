// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

const (
	firstRegistrationFailed = "first register failed: %v"
	duplicateRegistration   = "expected error for duplicate registration"
	duplicateError          = "expected error for duplicate"
)

// ============================================================================
// AlreadyExists errors on duplicate registration
// ============================================================================

func TestAlreadyExistsErrorOnDuplicateToolSanitizeRequest(t *testing.T) {
	name := "go_err_dup_san_req"
	fn := func(n string, args json.RawMessage) json.RawMessage { return args }

	err := RegisterToolSanitizeRequestGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterToolSanitizeRequestGuardrail(name)

	err = RegisterToolSanitizeRequestGuardrail(name, 1, fn)
	if err == nil {
		t.Fatal("expected AlreadyExists error for duplicate registration")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "already") {
		t.Logf("error message: %v (checking it is an AlreadyExists-type error)", err)
	}
}

func TestAlreadyExistsErrorOnDuplicateToolSanitizeResponse(t *testing.T) {
	name := "go_err_dup_san_resp"
	fn := func(n string, args json.RawMessage) json.RawMessage { return args }

	err := RegisterToolSanitizeResponseGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterToolSanitizeResponseGuardrail(name)

	err = RegisterToolSanitizeResponseGuardrail(name, 1, fn)
	if err == nil {
		t.Fatal(duplicateRegistration)
	}
}

func TestAlreadyExistsErrorOnDuplicateToolConditional(t *testing.T) {
	name := "go_err_dup_cond"
	fn := func(n string, args json.RawMessage) *string { return nil }

	err := RegisterToolConditionalExecutionGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterToolConditionalExecutionGuardrail(name)

	err = RegisterToolConditionalExecutionGuardrail(name, 1, fn)
	if err == nil {
		t.Fatal(duplicateRegistration)
	}
}

func TestAlreadyExistsErrorOnDuplicateToolRequestIntercept(t *testing.T) {
	name := "go_err_dup_req_int"
	fn := func(n string, args json.RawMessage) json.RawMessage { return args }

	err := RegisterToolRequestIntercept(name, 1, false, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterToolRequestIntercept(name)

	err = RegisterToolRequestIntercept(name, 1, false, fn)
	if err == nil {
		t.Fatal(duplicateRegistration)
	}
}

func TestAlreadyExistsErrorOnDuplicateToolExecutionIntercept(t *testing.T) {
	name := "go_err_dup_exec_int"
	fn := func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		return next(args)
	}

	err := RegisterToolExecutionIntercept(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterToolExecutionIntercept(name)

	err = RegisterToolExecutionIntercept(name, 1, fn)
	if err == nil {
		t.Fatal(duplicateRegistration)
	}
}

func TestAlreadyExistsErrorOnDuplicateSubscriber(t *testing.T) {
	name := "go_err_dup_sub"
	fn := func(event Event) {
		// Subscriber is intentionally empty for duplicate-registration coverage.
	}

	err := RegisterSubscriber(name, fn)
	if err != nil {
		t.Fatalf(firstRegistrationFailed, err)
	}
	defer DeregisterSubscriber(name)

	err = RegisterSubscriber(name, fn)
	if err == nil {
		t.Fatal("expected error for duplicate subscriber")
	}
}

func TestAlreadyExistsErrorOnDuplicateLlmGuardrails(t *testing.T) {
	t.Run("LlmSanitizeRequest", func(t *testing.T) {
		name := "go_err_dup_llm_san_req"
		fn := func(h, c json.RawMessage) (json.RawMessage, json.RawMessage) { return h, c }

		err := RegisterLlmSanitizeRequestGuardrail(name, 1, fn)
		if err != nil {
			t.Fatalf(firstRegistrationFailed, err)
		}
		defer DeregisterLlmSanitizeRequestGuardrail(name)

		err = RegisterLlmSanitizeRequestGuardrail(name, 1, fn)
		if err == nil {
			t.Fatal(duplicateError)
		}
	})

	t.Run("LlmSanitizeResponse", func(t *testing.T) {
		name := "go_err_dup_llm_san_resp"
		fn := func(r json.RawMessage) json.RawMessage { return r }

		err := RegisterLlmSanitizeResponseGuardrail(name, 1, fn)
		if err != nil {
			t.Fatalf(firstRegistrationFailed, err)
		}
		defer DeregisterLlmSanitizeResponseGuardrail(name)

		err = RegisterLlmSanitizeResponseGuardrail(name, 1, fn)
		if err == nil {
			t.Fatal(duplicateError)
		}
	})

	t.Run("LlmConditional", func(t *testing.T) {
		name := "go_err_dup_llm_cond"
		fn := func(h, c json.RawMessage) *string { return nil }

		err := RegisterLlmConditionalExecutionGuardrail(name, 1, fn)
		if err != nil {
			t.Fatalf(firstRegistrationFailed, err)
		}
		defer DeregisterLlmConditionalExecutionGuardrail(name)

		err = RegisterLlmConditionalExecutionGuardrail(name, 1, fn)
		if err == nil {
			t.Fatal(duplicateError)
		}
	})
}

// ============================================================================
// GuardrailRejected error format
// ============================================================================

func TestGuardrailRejectedErrorMessage(t *testing.T) {
	msg := "custom rejection reason"
	RegisterToolConditionalExecutionGuardrail("go_err_reject_msg", 1,
		func(name string, args json.RawMessage) *string {
			return &msg
		},
	)
	defer DeregisterToolConditionalExecutionGuardrail("go_err_reject_msg")

	_, err := ToolCallExecute("rejected_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected guardrail rejection error")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' in error, got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "custom rejection reason") {
		t.Fatalf("expected custom rejection reason in error, got: %s", errMsg)
	}
}

func TestLlmGuardrailRejectedErrorMessage(t *testing.T) {
	msg := "LLM policy violation"
	RegisterLlmConditionalExecutionGuardrail("go_err_llm_reject_msg", 1,
		func(headers, content json.RawMessage) *string {
			return &msg
		},
	)
	defer DeregisterLlmConditionalExecutionGuardrail("go_err_llm_reject_msg")

	request := map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"model": "test"},
	}
	_, err := LlmCallExecute("rejected_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected guardrail rejection error")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' in error, got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "LLM policy violation") {
		t.Fatalf("expected 'LLM policy violation' in error, got: %s", errMsg)
	}
}

// ============================================================================
// Scope-local errors
// ============================================================================

func TestScopeLocalDuplicateRegistrationError(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("err_dup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()
		fn := func(name string, args json.RawMessage) json.RawMessage { return args }

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "err_dup_scope_guard", 1, fn)
		if err != nil {
			t.Fatalf("first registration should succeed: %v", err)
		}

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "err_dup_scope_guard", 1, fn)
		if err == nil {
			t.Fatal("expected error for duplicate scope-local registration")
		}
	})
}

// ============================================================================
// Event subscriber captures sanitize guardrail effects on event input
// ============================================================================

func TestSanitizeGuardrailAffectsEventInput(t *testing.T) {
	// Sanitize guardrails modify what appears in the event's input field,
	// not the args passed to the tool callable
	var capturedInput json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_err_san_evt_sub", func(event Event) {
		if event.Kind() == "scope" && event.Category() == "tool" && event.ScopeCategory() == "start" {
			mu.Lock()
			capturedInput = append(json.RawMessage(nil), event.Input()...)
			mu.Unlock()
		}
	})
	defer DeregisterSubscriber("go_err_san_evt_sub")

	RegisterToolSanitizeRequestGuardrail("go_err_san_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["sanitized_in_event"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	defer DeregisterToolSanitizeRequestGuardrail("go_err_san_guard")

	_, err := ToolCallExecute("san_evt_tool", json.RawMessage(`{"input": "test"}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"done": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if capturedInput == nil {
		t.Fatal("expected non-nil captured input")
	}
	var input map[string]interface{}
	json.Unmarshal(capturedInput, &input)
	if input["sanitized_in_event"] != true {
		t.Fatalf("expected sanitized_in_event=true in event input, got %v", input)
	}
}
