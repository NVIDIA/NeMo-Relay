// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package guardrails provides shorthand access to NVAgentRT guardrail registration.
//
// Guardrails are priority-ordered middleware that sanitize or gate tool and LLM
// calls. They run in priority order (lower values first). Function names drop
// the "Guardrail" suffix found in the parent nvagentrt package.
//
// Three guardrail categories are supported for both tools and LLMs:
//   - SanitizeRequest: modifies outgoing request arguments/parameters.
//   - SanitizeResponse: modifies incoming response data.
//   - ConditionalExecution: gates whether the call should proceed at all.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/guardrails"
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

	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// --- Tool Sanitize Request ---

// RegisterToolSanitizeRequest registers a guardrail that sanitizes tool request
// arguments before they are passed to the tool. The callback receives the tool
// name and arguments JSON and must return the (possibly modified) arguments.
// Guardrails run in priority order (lower values first). This is a shorthand
// for [nvagentrt.RegisterToolSanitizeRequestGuardrail].
func RegisterToolSanitizeRequest(name string, priority int32, fn nvagentrt.ToolSanitizeFunc) error {
	return nvagentrt.RegisterToolSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeRequest removes a tool sanitize-request guardrail by
// name. This is a shorthand for [nvagentrt.DeregisterToolSanitizeRequestGuardrail].
func DeregisterToolSanitizeRequest(name string) error {
	return nvagentrt.DeregisterToolSanitizeRequestGuardrail(name)
}

// --- Tool Sanitize Response ---

// RegisterToolSanitizeResponse registers a guardrail that sanitizes tool
// response data before it is returned to the caller. The callback receives the
// tool name and response JSON and must return the (possibly modified) response.
// This is a shorthand for [nvagentrt.RegisterToolSanitizeResponseGuardrail].
func RegisterToolSanitizeResponse(name string, priority int32, fn nvagentrt.ToolSanitizeFunc) error {
	return nvagentrt.RegisterToolSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterToolSanitizeResponse removes a tool sanitize-response guardrail by
// name. This is a shorthand for [nvagentrt.DeregisterToolSanitizeResponseGuardrail].
func DeregisterToolSanitizeResponse(name string) error {
	return nvagentrt.DeregisterToolSanitizeResponseGuardrail(name)
}

// --- Tool Conditional Execution ---

// RegisterToolConditionalExecution registers a guardrail that conditionally
// gates tool execution. The callback returns nil to allow execution or a
// non-nil pointer to an error message string to reject it. This is a shorthand
// for [nvagentrt.RegisterToolConditionalExecutionGuardrail].
func RegisterToolConditionalExecution(name string, priority int32, fn nvagentrt.ToolConditionalFunc) error {
	return nvagentrt.RegisterToolConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterToolConditionalExecution removes a tool conditional-execution
// guardrail by name. This is a shorthand for
// [nvagentrt.DeregisterToolConditionalExecutionGuardrail].
func DeregisterToolConditionalExecution(name string) error {
	return nvagentrt.DeregisterToolConditionalExecutionGuardrail(name)
}

// --- LLM Sanitize Request ---

// RegisterLlmSanitizeRequest registers a guardrail that sanitizes LLM request
// parameters (HTTP method, URL, headers, body) before the call is made. This is
// a shorthand for [nvagentrt.RegisterLlmSanitizeRequestGuardrail].
func RegisterLlmSanitizeRequest(name string, priority int32, fn nvagentrt.LLMRequestFunc) error {
	return nvagentrt.RegisterLlmSanitizeRequestGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeRequest removes an LLM sanitize-request guardrail by
// name. This is a shorthand for [nvagentrt.DeregisterLlmSanitizeRequestGuardrail].
func DeregisterLlmSanitizeRequest(name string) error {
	return nvagentrt.DeregisterLlmSanitizeRequestGuardrail(name)
}

// --- LLM Sanitize Response ---

// RegisterLlmSanitizeResponse registers a guardrail that sanitizes LLM response
// JSON before it is returned to the caller. This is a shorthand for
// [nvagentrt.RegisterLlmSanitizeResponseGuardrail].
func RegisterLlmSanitizeResponse(name string, priority int32, fn nvagentrt.JSONFunc) error {
	return nvagentrt.RegisterLlmSanitizeResponseGuardrail(name, priority, fn)
}

// DeregisterLlmSanitizeResponse removes an LLM sanitize-response guardrail by
// name. This is a shorthand for [nvagentrt.DeregisterLlmSanitizeResponseGuardrail].
func DeregisterLlmSanitizeResponse(name string) error {
	return nvagentrt.DeregisterLlmSanitizeResponseGuardrail(name)
}

// --- LLM Conditional Execution ---

// RegisterLlmConditionalExecution registers a guardrail that conditionally
// gates LLM execution. The callback receives LLM request parameters and returns
// nil to allow execution or a non-nil pointer to an error message string to
// reject it. This is a shorthand for
// [nvagentrt.RegisterLlmConditionalExecutionGuardrail].
func RegisterLlmConditionalExecution(name string, priority int32, fn nvagentrt.LLMConditionalFunc) error {
	return nvagentrt.RegisterLlmConditionalExecutionGuardrail(name, priority, fn)
}

// DeregisterLlmConditionalExecution removes an LLM conditional-execution
// guardrail by name. This is a shorthand for
// [nvagentrt.DeregisterLlmConditionalExecutionGuardrail].
func DeregisterLlmConditionalExecution(name string) error {
	return nvagentrt.DeregisterLlmConditionalExecutionGuardrail(name)
}

// --- Tool Conditional Execution (standalone) ---

// ToolConditionalExecution runs the registered tool conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nvagentrt.ToolConditionalExecution].
func ToolConditionalExecution(name string, args json.RawMessage) error {
	return nvagentrt.ToolConditionalExecution(name, args)
}

// --- LLM Conditional Execution (standalone) ---

// LlmConditionalExecution runs the registered LLM conditional execution
// guardrail chain. Returns nil if all pass, or an error if blocked. This is a
// shorthand for [nvagentrt.LlmConditionalExecution].
func LlmConditionalExecution(request json.RawMessage) error {
	return nvagentrt.LlmConditionalExecution(request)
}
