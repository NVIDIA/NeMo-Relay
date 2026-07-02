// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

const (
	firstRegisterFailed = "first register failed: %v"
	deregisterFailed    = "deregister failed: %v"
	reregisterFailed    = "re-register failed: %v"
)

// ============================================================================
// Deregister nonexistent (no panic, no error -- silent success)
// ============================================================================

func TestDeregisterNonexistentGuardrails(t *testing.T) {
	// These should not panic, but they may return errors
	DeregisterToolSanitizeRequestGuardrail("nonexistent")
	DeregisterToolSanitizeResponseGuardrail("nonexistent")
	DeregisterToolConditionalExecutionGuardrail("nonexistent")
	DeregisterLlmSanitizeRequestGuardrail("nonexistent")
	DeregisterLlmSanitizeResponseGuardrail("nonexistent")
	DeregisterLlmConditionalExecutionGuardrail("nonexistent")
}

func TestDeregisterNonexistentIntercepts(t *testing.T) {
	DeregisterToolRequestIntercept("nonexistent")
	DeregisterToolExecutionIntercept("nonexistent")
	DeregisterLlmRequestIntercept("nonexistent")
	DeregisterLlmExecutionIntercept("nonexistent")
	DeregisterLlmStreamExecutionIntercept("nonexistent")
}

// ============================================================================
// Deregister nonexistent is safe (does not panic)
// The FFI layer silently succeeds for nonexistent names.
// ============================================================================

func TestDeregisterNonexistentGuardrailIsSafe(t *testing.T) {
	// Deregistering something that was never registered should not panic
	// The FFI layer returns success (no error) for nonexistent names
	_ = DeregisterToolSanitizeRequestGuardrail("go_dereg_safe_tsr")
	_ = DeregisterToolSanitizeResponseGuardrail("go_dereg_safe_tsresp")
	_ = DeregisterToolConditionalExecutionGuardrail("go_dereg_safe_tc")
}

func TestDeregisterNonexistentSubscriberIsSafe(t *testing.T) {
	_ = DeregisterSubscriber("go_dereg_safe_sub")
}

func TestDeregisterNonexistentInterceptsIsSafe(t *testing.T) {
	_ = DeregisterToolRequestIntercept("go_dereg_safe_tri")
	_ = DeregisterToolExecutionIntercept("go_dereg_safe_tei")
	_ = DeregisterLlmRequestIntercept("go_dereg_safe_lri")
	_ = DeregisterLlmExecutionIntercept("go_dereg_safe_lei")
	_ = DeregisterLlmStreamExecutionIntercept("go_dereg_safe_lsei")
}

// ============================================================================
// Register, deregister, re-register same name
// ============================================================================

func TestRegisterDeregisterReregisterToolSanitizeRequestGuardrail(t *testing.T) {
	name := "go_reregister_san_req"
	fn := func(n string, args json.RawMessage) json.RawMessage { return args }

	err := RegisterToolSanitizeRequestGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterToolSanitizeRequestGuardrail(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterToolSanitizeRequestGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterToolSanitizeRequestGuardrail(name)
}

func TestRegisterDeregisterReregisterSubscriber(t *testing.T) {
	name := "go_reregister_sub"
	fn := func(event Event) {
		// Subscriber is intentionally empty for register/deregister coverage.
	}

	err := RegisterSubscriber(name, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterSubscriber(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterSubscriber(name, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterSubscriber(name)
}

func TestRegisterDeregisterReregisterToolRequestIntercept(t *testing.T) {
	name := "go_reregister_req_int"
	fn := func(n string, args json.RawMessage) json.RawMessage { return args }

	err := RegisterToolRequestIntercept(name, 1, false, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterToolRequestIntercept(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterToolRequestIntercept(name, 1, false, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterToolRequestIntercept(name)
}

func TestRegisterDeregisterReregisterToolExecutionIntercept(t *testing.T) {
	name := "go_reregister_exec_int"
	fn := func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
		return toolExecutionOutcome(next(args))
	}

	err := RegisterToolExecutionIntercept(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterToolExecutionIntercept(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterToolExecutionIntercept(name, 1, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterToolExecutionIntercept(name)
}

func TestRegisterDeregisterReregisterLlmSanitizeRequestGuardrail(t *testing.T) {
	name := "go_reregister_llm_san_req"
	fn := func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
		return headers, content
	}

	err := RegisterLlmSanitizeRequestGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterLlmSanitizeRequestGuardrail(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterLlmSanitizeRequestGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterLlmSanitizeRequestGuardrail(name)
}

func TestRegisterDeregisterReregisterToolConditionalGuardrail(t *testing.T) {
	name := "go_reregister_cond"
	fn := func(n string, args json.RawMessage) *string { return nil }

	err := RegisterToolConditionalExecutionGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(firstRegisterFailed, err)
	}

	err = DeregisterToolConditionalExecutionGuardrail(name)
	if err != nil {
		t.Fatalf(deregisterFailed, err)
	}

	err = RegisterToolConditionalExecutionGuardrail(name, 1, fn)
	if err != nil {
		t.Fatalf(reregisterFailed, err)
	}

	DeregisterToolConditionalExecutionGuardrail(name)
}

// ============================================================================
// Deregister all types
// ============================================================================

func TestDeregisterAllToolGuardrailTypes(t *testing.T) {
	// Register one of each type, then deregister all
	RegisterToolSanitizeRequestGuardrail("go_dereg_all_san_req", 1,
		func(n string, args json.RawMessage) json.RawMessage { return args },
	)
	RegisterToolSanitizeResponseGuardrail("go_dereg_all_san_resp", 1,
		func(n string, args json.RawMessage) json.RawMessage { return args },
	)
	RegisterToolConditionalExecutionGuardrail("go_dereg_all_cond", 1,
		func(n string, args json.RawMessage) *string { return nil },
	)

	if err := DeregisterToolSanitizeRequestGuardrail("go_dereg_all_san_req"); err != nil {
		t.Fatalf("deregister san req failed: %v", err)
	}
	if err := DeregisterToolSanitizeResponseGuardrail("go_dereg_all_san_resp"); err != nil {
		t.Fatalf("deregister san resp failed: %v", err)
	}
	if err := DeregisterToolConditionalExecutionGuardrail("go_dereg_all_cond"); err != nil {
		t.Fatalf("deregister cond failed: %v", err)
	}
}

func TestDeregisterAllInterceptTypes(t *testing.T) {
	RegisterToolRequestIntercept("go_dereg_all_req_int", 1, false,
		func(n string, args json.RawMessage) json.RawMessage { return args },
	)
	RegisterToolExecutionIntercept("go_dereg_all_exec_int", 1,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
			return toolExecutionOutcome(next(args))
		},
	)

	if err := DeregisterToolRequestIntercept("go_dereg_all_req_int"); err != nil {
		t.Fatalf("deregister req intercept failed: %v", err)
	}
	if err := DeregisterToolExecutionIntercept("go_dereg_all_exec_int"); err != nil {
		t.Fatalf("deregister exec intercept failed: %v", err)
	}
}
