// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package guardrails provides shorthand access to Nexus guardrail registration.
//
// Guardrails are priority-ordered middleware that sanitize or gate tool and LLM
// calls. They run in priority order (lower values first). Function names drop
// the "Guardrail" suffix found in the parent nat_nexus package.
//
// Three guardrail categories are supported for both tools and LLMs:
//   - SanitizeRequest: modifies outgoing request arguments/parameters.
//   - SanitizeResponse: modifies incoming response data.
//   - ConditionalExecution: gates whether the call should proceed at all.
//
// Example usage:
//
//	import "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/guardrails"
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

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

// --- Tool Sanitize Request ---

// RegisterToolSanitizeRequest registers a guardrail that sanitizes tool request
// arguments before they are passed to the tool. The callback receives the tool
// name and arguments JSON and must return the (possibly modified) arguments.
// Guardrails run in priority order (lower values first). This is a shorthand
// for [nat_nexus.RegisterToolSanitizeRequestGuardrail].
func RegisterToolSanitizeRequest(name string, priority int32, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.RegisterToolSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeRequest removes a tool sanitize-request guardrail by
// name. This is a shorthand for [nat_nexus.DeregisterToolSanitizeRequestGuardrail].
func DeregisterToolSanitizeRequest(name string) error {
	return nat_nexus.DeregisterToolSanitizeRequestGuardrail(name)
}

// --- Tool Sanitize Response ---

// RegisterToolSanitizeResponse registers a guardrail that sanitizes tool
// response data before it is returned to the caller. The callback receives the
// tool name and response JSON and must return the (possibly modified) response.
// This is a shorthand for [nat_nexus.RegisterToolSanitizeResponseGuardrail].
func RegisterToolSanitizeResponse(name string, priority int32, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.RegisterToolSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeResponse removes a tool sanitize-response guardrail by
// name. This is a shorthand for [nat_nexus.DeregisterToolSanitizeResponseGuardrail].
func DeregisterToolSanitizeResponse(name string) error {
	return nat_nexus.DeregisterToolSanitizeResponseGuardrail(name)
}

// --- Tool Conditional Execution ---

// RegisterToolConditionalExecution registers a guardrail that conditionally
// gates tool execution. The callback returns nil to allow execution or a
// non-nil pointer to an error message string to reject it. This is a shorthand
// for [nat_nexus.RegisterToolConditionalExecutionGuardrail].
func RegisterToolConditionalExecution(name string, priority int32, fn nat_nexus.ToolConditionalFunc) error {
	return nat_nexus.RegisterToolConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterToolConditionalExecution removes a tool conditional-execution
// guardrail by name. This is a shorthand for
// [nat_nexus.DeregisterToolConditionalExecutionGuardrail].
func DeregisterToolConditionalExecution(name string) error {
	return nat_nexus.DeregisterToolConditionalExecutionGuardrail(name)
}

// --- LLM Sanitize Request ---

// RegisterLlmSanitizeRequest registers a guardrail that sanitizes the LLM
// request data (headers and content) before the call is made. This is a
// shorthand for [nat_nexus.RegisterLlmSanitizeRequestGuardrail].
func RegisterLlmSanitizeRequest(name string, priority int32, fn nat_nexus.LLMRequestFunc) error {
	return nat_nexus.RegisterLlmSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeRequest removes an LLM sanitize-request guardrail by
// name. This is a shorthand for [nat_nexus.DeregisterLlmSanitizeRequestGuardrail].
func DeregisterLlmSanitizeRequest(name string) error {
	return nat_nexus.DeregisterLlmSanitizeRequestGuardrail(name)
}

// --- LLM Sanitize Response ---

// RegisterLlmSanitizeResponse registers a guardrail that sanitizes LLM response
// data before it is returned to the caller. The callback receives the response
// as plain JSON. This is a shorthand for
// [nat_nexus.RegisterLlmSanitizeResponseGuardrail].
func RegisterLlmSanitizeResponse(name string, priority int32, fn nat_nexus.LLMResponseFunc) error {
	return nat_nexus.RegisterLlmSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeResponse removes an LLM sanitize-response guardrail by
// name. This is a shorthand for [nat_nexus.DeregisterLlmSanitizeResponseGuardrail].
func DeregisterLlmSanitizeResponse(name string) error {
	return nat_nexus.DeregisterLlmSanitizeResponseGuardrail(name)
}

// --- LLM Conditional Execution ---

// RegisterLlmConditionalExecution registers a guardrail that conditionally
// gates LLM execution. The callback receives LLM request parameters and returns
// nil to allow execution or a non-nil pointer to an error message string to
// reject it. This is a shorthand for
// [nat_nexus.RegisterLlmConditionalExecutionGuardrail].
func RegisterLlmConditionalExecution(name string, priority int32, fn nat_nexus.LLMConditionalFunc) error {
	return nat_nexus.RegisterLlmConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterLlmConditionalExecution removes an LLM conditional-execution
// guardrail by name. This is a shorthand for
// [nat_nexus.DeregisterLlmConditionalExecutionGuardrail].
func DeregisterLlmConditionalExecution(name string) error {
	return nat_nexus.DeregisterLlmConditionalExecutionGuardrail(name)
}

// --- Scope-local Tool Sanitize Request ---

// ScopeRegisterToolSanitizeRequest registers a scope-local guardrail that
// sanitizes tool request arguments. This is a shorthand for
// [nat_nexus.ScopeRegisterToolSanitizeRequestGuardrail].
func ScopeRegisterToolSanitizeRequest(scopeUUID string, name string, priority int32, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeRequest removes a scope-local tool sanitize-request
// guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterToolSanitizeRequestGuardrail].
func ScopeDeregisterToolSanitizeRequest(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Sanitize Response ---

// ScopeRegisterToolSanitizeResponse registers a scope-local guardrail that
// sanitizes tool response data. This is a shorthand for
// [nat_nexus.ScopeRegisterToolSanitizeResponseGuardrail].
func ScopeRegisterToolSanitizeResponse(scopeUUID string, name string, priority int32, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolSanitizeResponse removes a scope-local tool
// sanitize-response guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterToolSanitizeResponseGuardrail].
func ScopeDeregisterToolSanitizeResponse(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local Tool Conditional Execution ---

// ScopeRegisterToolConditionalExecution registers a scope-local guardrail that
// conditionally gates tool execution. This is a shorthand for
// [nat_nexus.ScopeRegisterToolConditionalExecutionGuardrail].
func ScopeRegisterToolConditionalExecution(scopeUUID string, name string, priority int32, fn nat_nexus.ToolConditionalFunc) error {
	return nat_nexus.ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterToolConditionalExecution removes a scope-local tool
// conditional-execution guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterToolConditionalExecutionGuardrail].
func ScopeDeregisterToolConditionalExecution(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Request ---

// ScopeRegisterLlmSanitizeRequest registers a scope-local guardrail that
// sanitizes the LLM request data. This is a shorthand for
// [nat_nexus.ScopeRegisterLlmSanitizeRequestGuardrail].
func ScopeRegisterLlmSanitizeRequest(scopeUUID string, name string, priority int32, fn nat_nexus.LLMRequestFunc) error {
	return nat_nexus.ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeRequest removes a scope-local LLM sanitize-request
// guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterLlmSanitizeRequestGuardrail].
func ScopeDeregisterLlmSanitizeRequest(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Sanitize Response ---

// ScopeRegisterLlmSanitizeResponse registers a scope-local guardrail that
// sanitizes LLM response data. This is a shorthand for
// [nat_nexus.ScopeRegisterLlmSanitizeResponseGuardrail].
func ScopeRegisterLlmSanitizeResponse(scopeUUID string, name string, priority int32, fn nat_nexus.LLMResponseFunc) error {
	return nat_nexus.ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmSanitizeResponse removes a scope-local LLM
// sanitize-response guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterLlmSanitizeResponseGuardrail].
func ScopeDeregisterLlmSanitizeResponse(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, name)
}

// --- Scope-local LLM Conditional Execution ---

// ScopeRegisterLlmConditionalExecution registers a scope-local guardrail that
// conditionally gates LLM execution. This is a shorthand for
// [nat_nexus.ScopeRegisterLlmConditionalExecutionGuardrail].
func ScopeRegisterLlmConditionalExecution(scopeUUID string, name string, priority int32, fn nat_nexus.LLMConditionalFunc) error {
	return nat_nexus.ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, name, priority, fn)
}

// ScopeDeregisterLlmConditionalExecution removes a scope-local LLM
// conditional-execution guardrail by name. This is a shorthand for
// [nat_nexus.ScopeDeregisterLlmConditionalExecutionGuardrail].
func ScopeDeregisterLlmConditionalExecution(scopeUUID string, name string) error {
	return nat_nexus.ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, name)
}

// --- Tool Conditional Execution (standalone) ---

// ToolConditionalExecution runs the registered tool conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nat_nexus.ToolConditionalExecution].
func ToolConditionalExecution(name string, args json.RawMessage) error {
	return nat_nexus.ToolConditionalExecution(name, args)
}

// --- LLM Conditional Execution (standalone) ---

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nat_nexus.LlmConditionalExecution].
func LlmConditionalExecution(request json.RawMessage) error {
	return nat_nexus.LlmConditionalExecution(request)
}
