// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package guardrails provides shorthand access to NeMo Flow guardrail registration.
//
// Guardrails are priority-ordered middleware that sanitize or gate tool and LLM
// calls. They run in priority order (lower values first). Function names drop
// the "Guardrail" suffix found in the parent nemo_flow package.
//
// Three guardrail categories are supported for both tools and LLMs:
//   - SanitizeRequest: modifies outgoing request arguments/parameters.
//   - SanitizeResponse: modifies incoming response data.
//   - ConditionalExecution: gates whether the call should proceed at all.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/guardrails"
//
//	// Register a tool request sanitizer that redacts sensitive fields.
//	err := guardrails.RegisterToolSanitizeRequest("redact-pii", 10,
//	    func(name string, args json.RawMessage) json.RawMessage {
//	        // ... redact PII from args ...
//	        return args
//	    },
//	)
//
//	// Later, remove it.
//	_ = guardrails.DeregisterToolSanitizeRequest("redact-pii")
package guardrails

import (
	"encoding/json"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

// --- Tool Sanitize Request ---

// RegisterToolSanitizeRequest registers a guardrail that sanitizes tool request
// arguments before they are passed to the tool. The callback receives the tool
// name and arguments JSON and must return the (possibly modified) arguments.
// Guardrails run in priority order (lower values first). This is a shorthand
// for [nemo_flow.RegisterToolSanitizeRequestGuardrail].
func RegisterToolSanitizeRequest(name string, priority int32, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.RegisterToolSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeRequest removes a tool sanitize-request guardrail by
// name. This is a shorthand for [nemo_flow.DeregisterToolSanitizeRequestGuardrail].
func DeregisterToolSanitizeRequest(name string) error {
	return nemo_flow.DeregisterToolSanitizeRequestGuardrail(name)
}

// --- Tool Sanitize Response ---

// RegisterToolSanitizeResponse registers a guardrail that sanitizes tool
// response data before it is returned to the caller. The callback receives the
// tool name and response JSON and must return the (possibly modified) response.
// This is a shorthand for [nemo_flow.RegisterToolSanitizeResponseGuardrail].
func RegisterToolSanitizeResponse(name string, priority int32, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.RegisterToolSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeResponse removes a tool sanitize-response guardrail by
// name. This is a shorthand for [nemo_flow.DeregisterToolSanitizeResponseGuardrail].
func DeregisterToolSanitizeResponse(name string) error {
	return nemo_flow.DeregisterToolSanitizeResponseGuardrail(name)
}

// --- Tool Conditional Execution ---

// RegisterToolConditionalExecution registers a guardrail that conditionally
// gates tool execution. The callback returns nil to allow execution or a
// non-nil pointer to an error message string to reject it. This is a shorthand
// for [nemo_flow.RegisterToolConditionalExecutionGuardrail].
func RegisterToolConditionalExecution(name string, priority int32, fn nemo_flow.ToolConditionalFunc) error {
	return nemo_flow.RegisterToolConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterToolConditionalExecution removes a tool conditional-execution
// guardrail by name. This is a shorthand for
// [nemo_flow.DeregisterToolConditionalExecutionGuardrail].
func DeregisterToolConditionalExecution(name string) error {
	return nemo_flow.DeregisterToolConditionalExecutionGuardrail(name)
}

// --- LLM Sanitize Request ---

// RegisterLlmSanitizeRequest registers a guardrail that sanitizes the LLM
// request data (headers and content) before the call is made. This is a
// shorthand for [nemo_flow.RegisterLlmSanitizeRequestGuardrail].
func RegisterLlmSanitizeRequest(name string, priority int32, fn nemo_flow.LLMRequestFunc) error {
	return nemo_flow.RegisterLlmSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeRequest removes an LLM sanitize-request guardrail by
// name. This is a shorthand for [nemo_flow.DeregisterLlmSanitizeRequestGuardrail].
func DeregisterLlmSanitizeRequest(name string) error {
	return nemo_flow.DeregisterLlmSanitizeRequestGuardrail(name)
}

// --- LLM Sanitize Response ---

// RegisterLlmSanitizeResponse registers a guardrail that sanitizes LLM response
// data before it is returned to the caller. The callback receives the response
// as plain JSON. This is a shorthand for
// [nemo_flow.RegisterLlmSanitizeResponseGuardrail].
func RegisterLlmSanitizeResponse(name string, priority int32, fn nemo_flow.LLMResponseFunc) error {
	return nemo_flow.RegisterLlmSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeResponse removes an LLM sanitize-response guardrail by
// name. This is a shorthand for [nemo_flow.DeregisterLlmSanitizeResponseGuardrail].
func DeregisterLlmSanitizeResponse(name string) error {
	return nemo_flow.DeregisterLlmSanitizeResponseGuardrail(name)
}

// --- LLM Conditional Execution ---

// RegisterLlmConditionalExecution registers a guardrail that conditionally
// gates LLM execution. The callback receives LLM request parameters and returns
// nil to allow execution or a non-nil pointer to an error message string to
// reject it. This is a shorthand for
// [nemo_flow.RegisterLlmConditionalExecutionGuardrail].
func RegisterLlmConditionalExecution(name string, priority int32, fn nemo_flow.LLMConditionalFunc) error {
	return nemo_flow.RegisterLlmConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterLlmConditionalExecution removes an LLM conditional-execution
// guardrail by name. This is a shorthand for
// [nemo_flow.DeregisterLlmConditionalExecutionGuardrail].
func DeregisterLlmConditionalExecution(name string) error {
	return nemo_flow.DeregisterLlmConditionalExecutionGuardrail(name)
}

// --- Scope-local Tool Sanitize Request ---

// ScopeRegisterToolSanitizeRequest registers a scope-local guardrail that
// sanitizes tool request arguments. This is a shorthand for
// [nemo_flow.ScopeRegisterToolSanitizeRequestGuardrail].
func ScopeRegisterToolSanitizeRequest(scopeUUID string, name string, priority int32, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeRequest removes a scope-local tool sanitize-request
// guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterToolSanitizeRequestGuardrail].
func ScopeDeregisterToolSanitizeRequest(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Sanitize Response ---

// ScopeRegisterToolSanitizeResponse registers a scope-local guardrail that
// sanitizes tool response data. This is a shorthand for
// [nemo_flow.ScopeRegisterToolSanitizeResponseGuardrail].
func ScopeRegisterToolSanitizeResponse(scopeUUID string, name string, priority int32, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeResponse removes a scope-local tool
// sanitize-response guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterToolSanitizeResponseGuardrail].
func ScopeDeregisterToolSanitizeResponse(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Conditional Execution ---

// ScopeRegisterToolConditionalExecution registers a scope-local guardrail that
// conditionally gates tool execution. This is a shorthand for
// [nemo_flow.ScopeRegisterToolConditionalExecutionGuardrail].
func ScopeRegisterToolConditionalExecution(scopeUUID string, name string, priority int32, fn nemo_flow.ToolConditionalFunc) error {
	return nemo_flow.ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolConditionalExecution removes a scope-local tool
// conditional-execution guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterToolConditionalExecutionGuardrail].
func ScopeDeregisterToolConditionalExecution(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Request ---

// ScopeRegisterLlmSanitizeRequest registers a scope-local guardrail that
// sanitizes the LLM request data. This is a shorthand for
// [nemo_flow.ScopeRegisterLlmSanitizeRequestGuardrail].
func ScopeRegisterLlmSanitizeRequest(scopeUUID string, name string, priority int32, fn nemo_flow.LLMRequestFunc) error {
	return nemo_flow.ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeRequest removes a scope-local LLM sanitize-request
// guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterLlmSanitizeRequestGuardrail].
func ScopeDeregisterLlmSanitizeRequest(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Response ---

// ScopeRegisterLlmSanitizeResponse registers a scope-local guardrail that
// sanitizes LLM response data. This is a shorthand for
// [nemo_flow.ScopeRegisterLlmSanitizeResponseGuardrail].
func ScopeRegisterLlmSanitizeResponse(scopeUUID string, name string, priority int32, fn nemo_flow.LLMResponseFunc) error {
	return nemo_flow.ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeResponse removes a scope-local LLM
// sanitize-response guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterLlmSanitizeResponseGuardrail].
func ScopeDeregisterLlmSanitizeResponse(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Conditional Execution ---

// ScopeRegisterLlmConditionalExecution registers a scope-local guardrail that
// conditionally gates LLM execution. This is a shorthand for
// [nemo_flow.ScopeRegisterLlmConditionalExecutionGuardrail].
func ScopeRegisterLlmConditionalExecution(scopeUUID string, name string, priority int32, fn nemo_flow.LLMConditionalFunc) error {
	return nemo_flow.ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmConditionalExecution removes a scope-local LLM
// conditional-execution guardrail by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterLlmConditionalExecutionGuardrail].
func ScopeDeregisterLlmConditionalExecution(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Tool Conditional Execution (standalone) ---

// ToolConditionalExecution runs the registered tool conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nemo_flow.ToolConditionalExecution].
func ToolConditionalExecution(name string, args json.RawMessage) error {
	return nemo_flow.ToolConditionalExecution(name, args)
}

// --- LLM Conditional Execution (standalone) ---

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nemo_flow.LlmConditionalExecution].
func LlmConditionalExecution(request json.RawMessage) error {
	return nemo_flow.LlmConditionalExecution(request)
}
