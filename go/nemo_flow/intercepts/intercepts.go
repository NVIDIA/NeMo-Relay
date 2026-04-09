// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package intercepts provides shorthand access to NeMo Flow intercept registration.
//
// Intercepts are priority-ordered middleware that transform or replace tool and
// LLM calls. They run in priority order (lower values first). Function names
// drop the "Intercept" suffix found in the parent nemo_flow package.
//
// Intercept categories for both tools and LLMs:
//   - Request: transforms request arguments/parameters; supports breakChain.
//   - Execution: middleware chain — each intercept receives a next function.
//   - StreamExecution (LLM only): middleware chain for streaming calls.
//
// When breakChain is true on a request intercept, no lower-priority
// intercepts in the chain are invoked after it.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/intercepts"
//
//	// Register a tool request intercept that injects a trace ID.
//	err := intercepts.RegisterToolRequest("add-trace-id", 5, false,
//	    func(name string, args json.RawMessage) json.RawMessage {
//	        // ... inject trace ID into args ...
//	        return args
//	    },
//	)
//
//	// Later, remove it.
//	_ = intercepts.DeregisterToolRequest("add-trace-id")
package intercepts

import (
	"encoding/json"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

// --- Tool Request ---

// RegisterToolRequest registers an intercept that transforms tool request
// arguments. When breakChain is true, no lower-priority intercepts run after
// this one. This is a shorthand for [nemo_flow.RegisterToolRequestIntercept].
func RegisterToolRequest(name string, priority int32, breakChain bool, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.RegisterToolRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterToolRequest removes a tool request intercept by name. This is a
// shorthand for [nemo_flow.DeregisterToolRequestIntercept].
func DeregisterToolRequest(name string) error {
	return nemo_flow.DeregisterToolRequestIntercept(name)
}

// --- Tool Execution ---

// RegisterToolExecution registers a tool execution intercept following the
// middleware chain pattern. execFn is called with args and a next function.
// Call next to continue the chain or skip it to short-circuit. This is a
// shorthand for [nemo_flow.RegisterToolExecutionIntercept].
func RegisterToolExecution(name string, priority int32, execFn nemo_flow.ToolExecutionInterceptFunc) error {
	return nemo_flow.RegisterToolExecutionIntercept(name, priority, execFn)
}

// DeregisterToolExecution removes a tool execution intercept by name. This is a
// shorthand for [nemo_flow.DeregisterToolExecutionIntercept].
func DeregisterToolExecution(name string) error {
	return nemo_flow.DeregisterToolExecutionIntercept(name)
}

// --- LLM Request ---

// RegisterLlmRequest registers an intercept that transforms the LLM request
// (headers, content, and optionally annotated JSON). When breakChain is true,
// no lower-priority intercepts run after this one. This is a shorthand for
// [nemo_flow.RegisterLlmRequestIntercept].
func RegisterLlmRequest(name string, priority int32, breakChain bool, fn nemo_flow.LLMRequestInterceptFunc) error {
	return nemo_flow.RegisterLlmRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmRequest removes an LLM request intercept by name. This is a
// shorthand for [nemo_flow.DeregisterLlmRequestIntercept].
func DeregisterLlmRequest(name string) error {
	return nemo_flow.DeregisterLlmRequestIntercept(name)
}

// --- LLM Execution ---

// RegisterLlmExecution registers an LLM execution intercept following the
// middleware chain pattern. execFn is called with the request and a next
// function. Call next to continue the chain or skip it to short-circuit. This
// is a shorthand for [nemo_flow.RegisterLlmExecutionIntercept].
func RegisterLlmExecution(name string, priority int32, execFn nemo_flow.LLMExecutionInterceptFunc) error {
	return nemo_flow.RegisterLlmExecutionIntercept(name, priority, execFn)
}

// DeregisterLlmExecution removes an LLM execution intercept by name. This is a
// shorthand for [nemo_flow.DeregisterLlmExecutionIntercept].
func DeregisterLlmExecution(name string) error {
	return nemo_flow.DeregisterLlmExecutionIntercept(name)
}

// --- LLM Stream Execution ---

// RegisterLlmStreamExecution registers a streaming LLM execution intercept
// following the middleware chain pattern. execFn is called with the request and
// a next function. Call next to continue the chain or skip it to short-circuit.
// This is a shorthand for [nemo_flow.RegisterLlmStreamExecutionIntercept].
func RegisterLlmStreamExecution(name string, priority int32, execFn nemo_flow.LLMExecutionInterceptFunc) error {
	return nemo_flow.RegisterLlmStreamExecutionIntercept(name, priority, execFn)
}

// DeregisterLlmStreamExecution removes an LLM stream execution intercept by
// name. This is a shorthand for [nemo_flow.DeregisterLlmStreamExecutionIntercept].
func DeregisterLlmStreamExecution(name string) error {
	return nemo_flow.DeregisterLlmStreamExecutionIntercept(name)
}

// --- Scope-local Tool Request ---

// ScopeRegisterToolRequest registers a scope-local intercept that transforms
// tool request arguments. This is a shorthand for
// [nemo_flow.ScopeRegisterToolRequestIntercept].
func ScopeRegisterToolRequest(scopeUUID string, name string, priority int32, breakChain bool, fn nemo_flow.ToolSanitizeFunc) error {
	return nemo_flow.ScopeRegisterToolRequestIntercept(scopeUUID, name, priority, breakChain, fn)
}

// ScopeDeregisterToolRequest removes a scope-local tool request intercept by
// name. This is a shorthand for [nemo_flow.ScopeDeregisterToolRequestIntercept].
func ScopeDeregisterToolRequest(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterToolRequestIntercept(scopeUUID, name)
}

// --- Scope-local Tool Execution ---

// ScopeRegisterToolExecution registers a scope-local tool execution intercept
// following the middleware chain pattern. This is a shorthand for
// [nemo_flow.ScopeRegisterToolExecutionIntercept].
func ScopeRegisterToolExecution(scopeUUID string, name string, priority int32, execFn nemo_flow.ToolExecutionInterceptFunc) error {
	return nemo_flow.ScopeRegisterToolExecutionIntercept(scopeUUID, name, priority, execFn)
}

// ScopeDeregisterToolExecution removes a scope-local tool execution intercept by
// name. This is a shorthand for [nemo_flow.ScopeDeregisterToolExecutionIntercept].
func ScopeDeregisterToolExecution(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterToolExecutionIntercept(scopeUUID, name)
}

// --- Scope-local LLM Request ---

// ScopeRegisterLlmRequest registers a scope-local intercept that transforms the
// LLM request using the unified annotated-aware signature. This is a shorthand
// for [nemo_flow.ScopeRegisterLlmRequestIntercept].
func ScopeRegisterLlmRequest(scopeUUID string, name string, priority int32, breakChain bool, fn nemo_flow.LLMRequestInterceptFunc) error {
	return nemo_flow.ScopeRegisterLlmRequestIntercept(scopeUUID, name, priority, breakChain, fn)
}

// ScopeDeregisterLlmRequest removes a scope-local LLM request intercept by
// name. This is a shorthand for [nemo_flow.ScopeDeregisterLlmRequestIntercept].
func ScopeDeregisterLlmRequest(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmRequestIntercept(scopeUUID, name)
}

// --- Scope-local LLM Execution ---

// ScopeRegisterLlmExecution registers a scope-local LLM execution intercept
// following the middleware chain pattern. This is a shorthand for
// [nemo_flow.ScopeRegisterLlmExecutionIntercept].
func ScopeRegisterLlmExecution(scopeUUID string, name string, priority int32, execFn nemo_flow.LLMExecutionInterceptFunc) error {
	return nemo_flow.ScopeRegisterLlmExecutionIntercept(scopeUUID, name, priority, execFn)
}

// ScopeDeregisterLlmExecution removes a scope-local LLM execution intercept by
// name. This is a shorthand for [nemo_flow.ScopeDeregisterLlmExecutionIntercept].
func ScopeDeregisterLlmExecution(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmExecutionIntercept(scopeUUID, name)
}

// --- Scope-local LLM Stream Execution ---

// ScopeRegisterLlmStreamExecution registers a scope-local streaming LLM
// execution intercept following the middleware chain pattern. This is a shorthand
// for [nemo_flow.ScopeRegisterLlmStreamExecutionIntercept].
func ScopeRegisterLlmStreamExecution(scopeUUID string, name string, priority int32, execFn nemo_flow.LLMExecutionInterceptFunc) error {
	return nemo_flow.ScopeRegisterLlmStreamExecutionIntercept(scopeUUID, name, priority, execFn)
}

// ScopeDeregisterLlmStreamExecution removes a scope-local LLM stream execution
// intercept by name. This is a shorthand for
// [nemo_flow.ScopeDeregisterLlmStreamExecutionIntercept].
func ScopeDeregisterLlmStreamExecution(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterLlmStreamExecutionIntercept(scopeUUID, name)
}

// --- Tool Request Intercepts (standalone) ---

// ToolRequestIntercepts runs the registered tool request intercept chain and
// returns the transformed arguments. This is a shorthand for
// [nemo_flow.ToolRequestIntercepts].
func ToolRequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nemo_flow.ToolRequestIntercepts(name, args)
}

// --- LLM Request Intercepts (standalone) ---

// LlmRequestIntercepts runs the registered LLM request intercept chain and
// returns the transformed request. This is a shorthand for
// [nemo_flow.LlmRequestIntercepts].
func LlmRequestIntercepts(name string, request json.RawMessage) (json.RawMessage, error) {
	return nemo_flow.LlmRequestIntercepts(name, request)
}
