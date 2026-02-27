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
//   - Execution: conditionally replaces the entire call implementation.
//   - StreamResponse (LLM only): transforms individual SSE events mid-stream.
//   - StreamExecution (LLM only): replaces the streaming call implementation.
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

// RegisterToolExecution registers an intercept that can replace tool execution
// entirely. The condFn callback is evaluated first; if it returns true, execFn
// is called instead of the original tool implementation. This is a shorthand
// for [nvagentrt.RegisterToolExecutionIntercept].
func RegisterToolExecution(name string, priority int32, condFn nvagentrt.ToolExecConditionalFunc, execFn nvagentrt.ToolExecutionFunc) error {
	return nvagentrt.RegisterToolExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterToolExecution removes a tool execution intercept by name. This is a
// shorthand for [nvagentrt.DeregisterToolExecutionIntercept].
func DeregisterToolExecution(name string) error {
	return nvagentrt.DeregisterToolExecutionIntercept(name)
}

// --- LLM Request ---

// RegisterLlmRequest registers an intercept that transforms LLM request
// parameters (HTTP method, URL, headers, body). When breakChain is true, no
// lower-priority intercepts run after this one. This is a shorthand for
// [nvagentrt.RegisterLlmRequestIntercept].
func RegisterLlmRequest(name string, priority int32, breakChain bool, fn nvagentrt.LLMRequestFunc) error {
	return nvagentrt.RegisterLlmRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmRequest removes an LLM request intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmRequestIntercept].
func DeregisterLlmRequest(name string) error {
	return nvagentrt.DeregisterLlmRequestIntercept(name)
}

// --- LLM Response ---

// RegisterLlmResponse registers an intercept that transforms LLM response JSON.
// When breakChain is true, no lower-priority intercepts run after this one.
// This is a shorthand for [nvagentrt.RegisterLlmResponseIntercept].
func RegisterLlmResponse(name string, priority int32, breakChain bool, fn nvagentrt.JSONFunc) error {
	return nvagentrt.RegisterLlmResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmResponse removes an LLM response intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmResponseIntercept].
func DeregisterLlmResponse(name string) error {
	return nvagentrt.DeregisterLlmResponseIntercept(name)
}

// --- LLM Stream Response ---

// RegisterLlmStreamResponse registers an intercept that transforms individual
// chunks during a streaming LLM response. When breakChain is true, no
// lower-priority intercepts run after this one. This is a shorthand for
// [nvagentrt.RegisterLlmStreamResponseIntercept].
func RegisterLlmStreamResponse(name string, priority int32, breakChain bool, fn nvagentrt.StringInterceptFunc) error {
	return nvagentrt.RegisterLlmStreamResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmStreamResponse removes an LLM stream response intercept by
// name. This is a shorthand for [nvagentrt.DeregisterLlmStreamResponseIntercept].
func DeregisterLlmStreamResponse(name string) error {
	return nvagentrt.DeregisterLlmStreamResponseIntercept(name)
}

// --- LLM Execution ---

// RegisterLlmExecution registers an intercept that can replace LLM execution
// entirely. The condFn callback is evaluated first; if it returns true, execFn
// is called instead of the original LLM implementation. This is a shorthand
// for [nvagentrt.RegisterLlmExecutionIntercept].
func RegisterLlmExecution(name string, priority int32, condFn nvagentrt.LLMExecConditionalFunc, execFn nvagentrt.LLMExecutionFunc) error {
	return nvagentrt.RegisterLlmExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterLlmExecution removes an LLM execution intercept by name. This is a
// shorthand for [nvagentrt.DeregisterLlmExecutionIntercept].
func DeregisterLlmExecution(name string) error {
	return nvagentrt.DeregisterLlmExecutionIntercept(name)
}

// --- LLM Stream Execution ---

// RegisterLlmStreamExecution registers an intercept that can replace streaming
// LLM execution entirely. The condFn callback is evaluated first; if it returns
// true, execFn is called instead of the original streaming implementation. This
// is a shorthand for [nvagentrt.RegisterLlmStreamExecutionIntercept].
func RegisterLlmStreamExecution(name string, priority int32, condFn nvagentrt.LLMExecConditionalFunc, execFn nvagentrt.LLMExecutionFunc) error {
	return nvagentrt.RegisterLlmStreamExecutionIntercept(name, priority, condFn, execFn)
}

// DeregisterLlmStreamExecution removes an LLM stream execution intercept by
// name. This is a shorthand for [nvagentrt.DeregisterLlmStreamExecutionIntercept].
func DeregisterLlmStreamExecution(name string) error {
	return nvagentrt.DeregisterLlmStreamExecutionIntercept(name)
}
