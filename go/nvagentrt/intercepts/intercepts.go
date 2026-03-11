// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package intercepts provides shorthand access to NVAgentRT intercept registration.
//
// Intercepts are priority-ordered middleware that transform or replace tool and
// LLM calls. They run in priority order (lower values first). Function names
// drop the "Intercept" suffix found in the parent nvagentrt package.
//
// Intercept categories for both tools and LLMs:
//   - Request: transforms request arguments/parameters; supports breakChain.
//   - Response: transforms response data; supports breakChain.
//   - Execution: middleware chain — each intercept receives a next function.
//   - StreamResponse (LLM only): transforms individual SSE events mid-stream.
//   - StreamExecution (LLM only): middleware chain for streaming calls.
//
// When breakChain is true on a request or response intercept, no lower-priority
// intercepts in the chain are invoked after it.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/intercepts"
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

	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// --- Tool Request ---

// RegisterToolRequest registers an intercept that transforms tool request
// arguments. When breakChain is true, no lower-priority intercepts run after
// this one. This is a shorthand for [nvagentrt.RegisterToolRequestIntercept].
func RegisterToolRequest(name string, priority int32, breakChain bool, fn nvagentrt.ToolSanitizeFunc) error {
	return nvagentrt.RegisterToolRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterToolRequest removes a tool request intercept by name. This is a
// shorthand for [nvagentrt.DeregisterToolRequestIntercept].
func DeregisterToolRequest(name string) error {
	return nvagentrt.DeregisterToolRequestIntercept(name)
}

// --- Tool Response ---

// RegisterToolResponse registers an intercept that transforms tool response
// data. When breakChain is true, no lower-priority intercepts run after this
// one. This is a shorthand for [nvagentrt.RegisterToolResponseIntercept].
func RegisterToolResponse(name string, priority int32, breakChain bool, fn nvagentrt.ToolSanitizeFunc) error {
	return nvagentrt.RegisterToolResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterToolResponse removes a tool response intercept by name. This is a
// shorthand for [nvagentrt.DeregisterToolResponseIntercept].
func DeregisterToolResponse(name string) error {
	return nvagentrt.DeregisterToolResponseIntercept(name)
}

// --- Tool Execution ---

// RegisterToolExecution registers a tool execution intercept following the
// middleware chain pattern. The condFn callback is evaluated first; if it
// returns true, execFn is called with args and a next function. Call next to
// continue the chain or skip it to short-circuit. This is a shorthand for
// [nvagentrt.RegisterToolExecutionIntercept].
func RegisterToolExecution(name string, priority int32, condFn nvagentrt.ToolExecConditionalFunc, execFn nvagentrt.ToolExecutionInterceptFunc) error {
	return nvagentrt.RegisterToolExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterToolExecution removes a tool execution intercept by name. This is a
// shorthand for [nvagentrt.DeregisterToolExecutionIntercept].
func DeregisterToolExecution(name string) error {
	return nvagentrt.DeregisterToolExecutionIntercept(name)
}

// --- LLM Request ---

// RegisterLlmRequest registers an intercept that transforms the LLM request
// JSON. When breakChain is true, no lower-priority intercepts run after this
// one. This is a shorthand for [nvagentrt.RegisterLlmRequestIntercept].
func RegisterLlmRequest(name string, priority int32, breakChain bool, fn nvagentrt.JSONFunc) error {
	return nvagentrt.RegisterLlmRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmRequest removes an LLM request intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmRequestIntercept].
func DeregisterLlmRequest(name string) error {
	return nvagentrt.DeregisterLlmRequestIntercept(name)
}

// --- LLM Response ---

// RegisterLlmResponse registers an intercept that transforms LLM response data.
// The callback receives serialized LLMResponse JSON (containing a "data" field).
// When breakChain is true, no lower-priority intercepts run after this one.
// This is a shorthand for [nvagentrt.RegisterLlmResponseIntercept].
func RegisterLlmResponse(name string, priority int32, breakChain bool, fn nvagentrt.LLMResponseFunc) error {
	return nvagentrt.RegisterLlmResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmResponse removes an LLM response intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmResponseIntercept].
func DeregisterLlmResponse(name string) error {
	return nvagentrt.DeregisterLlmResponseIntercept(name)
}

// --- LLM Stream Response ---

// RegisterLlmStreamResponse registers an intercept that transforms individual
// JSON chunks during a streaming LLM response. When breakChain is true, no
// lower-priority intercepts run after this one. This is a shorthand for
// [nvagentrt.RegisterLlmStreamResponseIntercept].
func RegisterLlmStreamResponse(name string, priority int32, breakChain bool, fn nvagentrt.ChunkInterceptFunc) error {
	return nvagentrt.RegisterLlmStreamResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmStreamResponse removes an LLM stream response intercept by
// name. This is a shorthand for [nvagentrt.DeregisterLlmStreamResponseIntercept].
func DeregisterLlmStreamResponse(name string) error {
	return nvagentrt.DeregisterLlmStreamResponseIntercept(name)
}

// --- LLM Execution ---

// RegisterLlmExecution registers an LLM execution intercept following the
// middleware chain pattern. The condFn callback is evaluated first; if it
// returns true, execFn is called with the request and a next function. Call
// next to continue the chain or skip it to short-circuit. This is a shorthand
// for [nvagentrt.RegisterLlmExecutionIntercept].
func RegisterLlmExecution(name string, priority int32, condFn nvagentrt.LLMExecConditionalFunc, execFn nvagentrt.LLMExecutionInterceptFunc) error {
	return nvagentrt.RegisterLlmExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterLlmExecution removes an LLM execution intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmExecutionIntercept].
func DeregisterLlmExecution(name string) error {
	return nvagentrt.DeregisterLlmExecutionIntercept(name)
}

// --- LLM Stream Execution ---

// RegisterLlmStreamExecution registers a streaming LLM execution intercept
// following the middleware chain pattern. The condFn callback is evaluated
// first; if it returns true, execFn is called with the request and a next
// function. Call next to continue the chain or skip it to short-circuit. This
// is a shorthand for [nvagentrt.RegisterLlmStreamExecutionIntercept].
func RegisterLlmStreamExecution(name string, priority int32, condFn nvagentrt.LLMExecConditionalFunc, execFn nvagentrt.LLMExecutionInterceptFunc) error {
	return nvagentrt.RegisterLlmStreamExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterLlmStreamExecution removes an LLM stream execution intercept by
// name. This is a shorthand for [nvagentrt.DeregisterLlmStreamExecutionIntercept].
func DeregisterLlmStreamExecution(name string) error {
	return nvagentrt.DeregisterLlmStreamExecutionIntercept(name)
}

// --- Tool Request Intercepts (standalone) ---

// ToolRequestIntercepts runs the registered tool request intercept chain and
// returns the transformed arguments. This is a shorthand for
// [nvagentrt.ToolRequestIntercepts].
func ToolRequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.ToolRequestIntercepts(name, args)
}

// --- Tool Response Intercepts (standalone) ---

// ToolResponseIntercepts runs the registered tool response intercept chain and
// returns the transformed result. This is a shorthand for
// [nvagentrt.ToolResponseIntercepts].
func ToolResponseIntercepts(name string, result json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.ToolResponseIntercepts(name, result)
}

// --- LLM Request Intercepts (standalone) ---

// LlmRequestIntercepts runs the registered LLM request intercept chain and
// returns the transformed request. This is a shorthand for
// [nvagentrt.LlmRequestIntercepts].
func LlmRequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.LlmRequestIntercepts(request)
}

// --- LLM Response Intercepts (standalone) ---

// LlmResponseIntercepts runs the registered LLM response intercept chain and
// returns the transformed response. This is a shorthand for
// [nvagentrt.LlmResponseIntercepts].
func LlmResponseIntercepts(response json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.LlmResponseIntercepts(response)
}
