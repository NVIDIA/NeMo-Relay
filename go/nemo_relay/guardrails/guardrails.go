// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package guardrails provides shorthand access to NeMo Relay guardrail registration.
//
// Guardrails are priority-ordered middleware that sanitize or gate tool and LLM
// calls. They run in priority order (lower values first). Function names drop
// the "Guardrail" suffix found in the parent nemo_relay package.
//
// Three guardrail categories are supported for both tools and LLMs:
//   - SanitizeRequest: modifies outgoing request arguments/parameters.
//   - SanitizeResponse: modifies incoming response data.
//   - ConditionalExecution: gates whether the call should proceed at all.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/guardrails"
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

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

// RegisterMarkSanitize registers a global mark event sanitizer.
func RegisterMarkSanitize(name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.RegisterMarkSanitizeGuardrail(name, priority, fn)
}

// DeregisterMarkSanitize removes a global mark event sanitizer.
func DeregisterMarkSanitize(name string) error {
	return nemo_relay.DeregisterMarkSanitizeGuardrail(name)
}

// RegisterScopeSanitizeStart registers a global scope-start event sanitizer.
func RegisterScopeSanitizeStart(name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.RegisterScopeSanitizeStartGuardrail(name, priority, fn)
}

// DeregisterScopeSanitizeStart removes a global scope-start event sanitizer.
func DeregisterScopeSanitizeStart(name string) error {
	return nemo_relay.DeregisterScopeSanitizeStartGuardrail(name)
}

// RegisterScopeSanitizeEnd registers a global scope-end event sanitizer.
func RegisterScopeSanitizeEnd(name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.RegisterScopeSanitizeEndGuardrail(name, priority, fn)
}

// DeregisterScopeSanitizeEnd removes a global scope-end event sanitizer.
func DeregisterScopeSanitizeEnd(name string) error {
	return nemo_relay.DeregisterScopeSanitizeEndGuardrail(name)
}

// ScopeRegisterMarkSanitize registers a scope-local mark event sanitizer.
func ScopeRegisterMarkSanitize(scopeUUID, name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.ScopeRegisterMarkSanitizeGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterMarkSanitize removes a scope-local mark event sanitizer.
func ScopeDeregisterMarkSanitize(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterMarkSanitizeGuardrail(scopeUUID, name)
}

// ScopeRegisterScopeSanitizeStart registers a scope-local scope-start sanitizer.
func ScopeRegisterScopeSanitizeStart(scopeUUID, name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.ScopeRegisterScopeSanitizeStartGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterScopeSanitizeStart removes a scope-local scope-start sanitizer.
func ScopeDeregisterScopeSanitizeStart(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterScopeSanitizeStartGuardrail(scopeUUID, name)
}

// ScopeRegisterScopeSanitizeEnd registers a scope-local scope-end sanitizer.
func ScopeRegisterScopeSanitizeEnd(scopeUUID, name string, priority int32, fn nemo_relay.EventSanitizeFunc) error {
	return nemo_relay.ScopeRegisterScopeSanitizeEndGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterScopeSanitizeEnd removes a scope-local scope-end sanitizer.
func ScopeDeregisterScopeSanitizeEnd(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterScopeSanitizeEndGuardrail(scopeUUID, name)
}

// --- Tool Sanitize Request ---

// RegisterToolSanitizeRequest registers a guardrail that sanitizes tool request
// arguments before they are passed to the tool. The callback receives the tool
// name and arguments JSON and must return the (possibly modified) arguments.
// Guardrails run in priority order (lower values first). This is a shorthand
// for [nemo_relay.RegisterToolSanitizeRequestGuardrail].
func RegisterToolSanitizeRequest(name string, priority int32, fn nemo_relay.ToolSanitizeFunc) error {
	return nemo_relay.RegisterToolSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeRequest removes a tool sanitize-request guardrail by
// name. This is a shorthand for [nemo_relay.DeregisterToolSanitizeRequestGuardrail].
func DeregisterToolSanitizeRequest(name string) error {
	return nemo_relay.DeregisterToolSanitizeRequestGuardrail(name)
}

// --- Tool Sanitize Response ---

// RegisterToolSanitizeResponse registers a guardrail that sanitizes tool
// response data before it is returned to the caller. The callback receives the
// tool name and response JSON and must return the (possibly modified) response.
// This is a shorthand for [nemo_relay.RegisterToolSanitizeResponseGuardrail].
func RegisterToolSanitizeResponse(name string, priority int32, fn nemo_relay.ToolSanitizeFunc) error {
	return nemo_relay.RegisterToolSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeResponse removes a tool sanitize-response guardrail by
// name. This is a shorthand for [nemo_relay.DeregisterToolSanitizeResponseGuardrail].
func DeregisterToolSanitizeResponse(name string) error {
	return nemo_relay.DeregisterToolSanitizeResponseGuardrail(name)
}

// --- Tool Conditional Execution ---

// RegisterToolConditionalExecution registers a guardrail that conditionally
// gates tool execution. The callback returns nil to allow execution or a
// non-nil pointer to an error message string to reject it. This is a shorthand
// for [nemo_relay.RegisterToolConditionalExecutionGuardrail].
func RegisterToolConditionalExecution(name string, priority int32, fn nemo_relay.ToolConditionalFunc) error {
	return nemo_relay.RegisterToolConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterToolConditionalExecution removes a tool conditional-execution
// guardrail by name. This is a shorthand for
// [nemo_relay.DeregisterToolConditionalExecutionGuardrail].
func DeregisterToolConditionalExecution(name string) error {
	return nemo_relay.DeregisterToolConditionalExecutionGuardrail(name)
}

// --- LLM Sanitize Request ---

// RegisterLlmSanitizeRequest registers a guardrail that sanitizes the LLM
// request data (headers and content) before the call is made. This is a
// shorthand for [nemo_relay.RegisterLlmSanitizeRequestGuardrail].
func RegisterLlmSanitizeRequest(name string, priority int32, fn nemo_relay.LLMRequestFunc) error {
	return nemo_relay.RegisterLlmSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeRequest removes an LLM sanitize-request guardrail by
// name. This is a shorthand for [nemo_relay.DeregisterLlmSanitizeRequestGuardrail].
func DeregisterLlmSanitizeRequest(name string) error {
	return nemo_relay.DeregisterLlmSanitizeRequestGuardrail(name)
}

// RegisterContextualLlmSanitizeRequest registers a codec-aware LLM request
// sanitizer invoked as “fn(request, context)“.
func RegisterContextualLlmSanitizeRequest(name string, priority int32, fn nemo_relay.ContextualLLMRequestFunc) error {
	return nemo_relay.RegisterContextualLlmSanitizeRequestGuardrail(name, priority, fn)
}

// --- LLM Sanitize Response ---

// RegisterLlmSanitizeResponse registers a guardrail that sanitizes LLM response
// data before it is returned to the caller. The callback receives the response
// as plain JSON. This is a shorthand for
// [nemo_relay.RegisterLlmSanitizeResponseGuardrail].
func RegisterLlmSanitizeResponse(name string, priority int32, fn nemo_relay.LLMResponseFunc) error {
	return nemo_relay.RegisterLlmSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeResponse removes an LLM sanitize-response guardrail by
// name. This is a shorthand for [nemo_relay.DeregisterLlmSanitizeResponseGuardrail].
func DeregisterLlmSanitizeResponse(name string) error {
	return nemo_relay.DeregisterLlmSanitizeResponseGuardrail(name)
}

// RegisterContextualLlmSanitizeResponse registers a codec-aware LLM response
// sanitizer invoked as “fn(response, context)“.
func RegisterContextualLlmSanitizeResponse(name string, priority int32, fn nemo_relay.ContextualLLMResponseFunc) error {
	return nemo_relay.RegisterContextualLlmSanitizeResponseGuardrail(name, priority, fn)
}

// --- LLM Conditional Execution ---

// RegisterLlmConditionalExecution registers a guardrail that conditionally
// gates LLM execution. The callback receives LLM request parameters and returns
// nil to allow execution or a non-nil pointer to an error message string to
// reject it. This is a shorthand for
// [nemo_relay.RegisterLlmConditionalExecutionGuardrail].
func RegisterLlmConditionalExecution(name string, priority int32, fn nemo_relay.LLMConditionalFunc) error {
	return nemo_relay.RegisterLlmConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterLlmConditionalExecution removes an LLM conditional-execution
// guardrail by name. This is a shorthand for
// [nemo_relay.DeregisterLlmConditionalExecutionGuardrail].
func DeregisterLlmConditionalExecution(name string) error {
	return nemo_relay.DeregisterLlmConditionalExecutionGuardrail(name)
}

// --- Scope-local Tool Sanitize Request ---

// ScopeRegisterToolSanitizeRequest registers a scope-local guardrail that
// sanitizes tool request arguments. This is a shorthand for
// [nemo_relay.ScopeRegisterToolSanitizeRequestGuardrail].
func ScopeRegisterToolSanitizeRequest(scopeUUID, name string, priority int32, fn nemo_relay.ToolSanitizeFunc) error {
	return nemo_relay.ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeRequest removes a scope-local tool sanitize-request
// guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterToolSanitizeRequestGuardrail].
func ScopeDeregisterToolSanitizeRequest(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Sanitize Response ---

// ScopeRegisterToolSanitizeResponse registers a scope-local guardrail that
// sanitizes tool response data. This is a shorthand for
// [nemo_relay.ScopeRegisterToolSanitizeResponseGuardrail].
func ScopeRegisterToolSanitizeResponse(scopeUUID, name string, priority int32, fn nemo_relay.ToolSanitizeFunc) error {
	return nemo_relay.ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeResponse removes a scope-local tool
// sanitize-response guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterToolSanitizeResponseGuardrail].
func ScopeDeregisterToolSanitizeResponse(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Conditional Execution ---

// ScopeRegisterToolConditionalExecution registers a scope-local guardrail that
// conditionally gates tool execution. This is a shorthand for
// [nemo_relay.ScopeRegisterToolConditionalExecutionGuardrail].
func ScopeRegisterToolConditionalExecution(scopeUUID, name string, priority int32, fn nemo_relay.ToolConditionalFunc) error {
	return nemo_relay.ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolConditionalExecution removes a scope-local tool
// conditional-execution guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterToolConditionalExecutionGuardrail].
func ScopeDeregisterToolConditionalExecution(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Request ---

// ScopeRegisterLlmSanitizeRequest registers a scope-local guardrail that
// sanitizes the LLM request data. This is a shorthand for
// [nemo_relay.ScopeRegisterLlmSanitizeRequestGuardrail].
func ScopeRegisterLlmSanitizeRequest(scopeUUID, name string, priority int32, fn nemo_relay.LLMRequestFunc) error {
	return nemo_relay.ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeRequest removes a scope-local LLM sanitize-request
// guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterLlmSanitizeRequestGuardrail].
func ScopeDeregisterLlmSanitizeRequest(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Response ---

// ScopeRegisterLlmSanitizeResponse registers a scope-local guardrail that
// sanitizes LLM response data. This is a shorthand for
// [nemo_relay.ScopeRegisterLlmSanitizeResponseGuardrail].
func ScopeRegisterLlmSanitizeResponse(scopeUUID, name string, priority int32, fn nemo_relay.LLMResponseFunc) error {
	return nemo_relay.ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeResponse removes a scope-local LLM
// sanitize-response guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterLlmSanitizeResponseGuardrail].
func ScopeDeregisterLlmSanitizeResponse(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Conditional Execution ---

// ScopeRegisterLlmConditionalExecution registers a scope-local guardrail that
// conditionally gates LLM execution. This is a shorthand for
// [nemo_relay.ScopeRegisterLlmConditionalExecutionGuardrail].
func ScopeRegisterLlmConditionalExecution(scopeUUID, name string, priority int32, fn nemo_relay.LLMConditionalFunc) error {
	return nemo_relay.ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmConditionalExecution removes a scope-local LLM
// conditional-execution guardrail by name. This is a shorthand for
// [nemo_relay.ScopeDeregisterLlmConditionalExecutionGuardrail].
func ScopeDeregisterLlmConditionalExecution(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Tool Conditional Execution (standalone) ---

// ToolConditionalExecution runs the registered tool conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nemo_relay.ToolConditionalExecution].
func ToolConditionalExecution(name string, args json.RawMessage) error {
	return nemo_relay.ToolConditionalExecution(name, args)
}

// --- LLM Conditional Execution (standalone) ---

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nemo_relay.LlmConditionalExecution].
func LlmConditionalExecution(request json.RawMessage) error {
	return nemo_relay.LlmConditionalExecution(request)
}
