// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package intercepts provides shorthand access to Nexus intercept registration.
//
// Intercepts are priority-ordered middleware that transform or replace tool and
// LLM calls. They run in priority order (lower values first). Function names
// drop the "Intercept" suffix found in the parent nat_nexus package.
//
// Intercept categories for both tools and LLMs:
//   - Request: transforms request arguments/parameters; supports breakChain.
//   - Response (tool only): transforms response data; supports breakChain.
//   - Execution: middleware chain — each intercept receives a next function.
//   - StreamExecution (LLM only): middleware chain for streaming calls.
//
// When breakChain is true on a request or response intercept, no lower-priority
// intercepts in the chain are invoked after it.
//
// Example usage:
//
//	import "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/intercepts"
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

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

// --- Tool Request ---

// RegisterToolRequest registers an intercept that transforms tool request
// arguments. When breakChain is true, no lower-priority intercepts run after
// this one. This is a shorthand for [nat_nexus.RegisterToolRequestIntercept].
func RegisterToolRequest(name string, priority int32, breakChain bool, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.RegisterToolRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterToolRequest removes a tool request intercept by name. This is a
// shorthand for [nat_nexus.DeregisterToolRequestIntercept].
func DeregisterToolRequest(name string) error {
	return nat_nexus.DeregisterToolRequestIntercept(name)
}

// --- Tool Response ---

// RegisterToolResponse registers an intercept that transforms tool response
// data. When breakChain is true, no lower-priority intercepts run after this
// one. This is a shorthand for [nat_nexus.RegisterToolResponseIntercept].
func RegisterToolResponse(name string, priority int32, breakChain bool, fn nat_nexus.ToolSanitizeFunc) error {
	return nat_nexus.RegisterToolResponseIntercept(name, priority, breakChain, fn)
}

// DeregisterToolResponse removes a tool response intercept by name. This is a
// shorthand for [nat_nexus.DeregisterToolResponseIntercept].
func DeregisterToolResponse(name string) error {
	return nat_nexus.DeregisterToolResponseIntercept(name)
}

// --- Tool Execution ---

// RegisterToolExecution registers a tool execution intercept following the
// middleware chain pattern. execFn is called with args and a next function.
// Call next to continue the chain or skip it to short-circuit. This is a
// shorthand for [nat_nexus.RegisterToolExecutionIntercept].
func RegisterToolExecution(name string, priority int32, execFn nat_nexus.ToolExecutionInterceptFunc) error {
	return nat_nexus.RegisterToolExecutionIntercept(name, priority, execFn)
}

// DeregisterToolExecution removes a tool execution intercept by name. This is a
// shorthand for [nat_nexus.DeregisterToolExecutionIntercept].
func DeregisterToolExecution(name string) error {
	return nat_nexus.DeregisterToolExecutionIntercept(name)
}

// --- LLM Request ---

// RegisterLlmRequest registers an intercept that transforms the LLM request
// (headers and content). When breakChain is true, no lower-priority intercepts
// run after this one. This is a shorthand for [nat_nexus.RegisterLlmRequestIntercept].
func RegisterLlmRequest(name string, priority int32, breakChain bool, fn nat_nexus.LLMRequestFunc) error {
	return nat_nexus.RegisterLlmRequestIntercept(name, priority, breakChain, fn)
}

// DeregisterLlmRequest removes an LLM request intercept by name. This is a
// shorthand for [nat_nexus.DeregisterLlmRequestIntercept].
func DeregisterLlmRequest(name string) error {
	return nat_nexus.DeregisterLlmRequestIntercept(name)
}

// --- LLM Execution ---

// RegisterLlmExecution registers an LLM execution intercept following the
// middleware chain pattern. execFn is called with the request and a next
// function. Call next to continue the chain or skip it to short-circuit. This
// is a shorthand for [nat_nexus.RegisterLlmExecutionIntercept].
func RegisterLlmExecution(name string, priority int32, execFn nat_nexus.LLMExecutionInterceptFunc) error {
	return nat_nexus.RegisterLlmExecutionIntercept(name, priority, execFn)
}

// DeregisterLlmExecution removes an LLM execution intercept by name. This is a
// shorthand for [nat_nexus.DeregisterLlmExecutionIntercept].
func DeregisterLlmExecution(name string) error {
	return nat_nexus.DeregisterLlmExecutionIntercept(name)
}

// --- LLM Stream Execution ---

// RegisterLlmStreamExecution registers a streaming LLM execution intercept
// following the middleware chain pattern. execFn is called with the request and
// a next function. Call next to continue the chain or skip it to short-circuit.
// This is a shorthand for [nat_nexus.RegisterLlmStreamExecutionIntercept].
func RegisterLlmStreamExecution(name string, priority int32, execFn nat_nexus.LLMExecutionInterceptFunc) error {
	return nat_nexus.RegisterLlmStreamExecutionIntercept(name, priority, execFn)
}

// DeregisterLlmStreamExecution removes an LLM stream execution intercept by
// name. This is a shorthand for [nat_nexus.DeregisterLlmStreamExecutionIntercept].
func DeregisterLlmStreamExecution(name string) error {
	return nat_nexus.DeregisterLlmStreamExecutionIntercept(name)
}

// --- Tool Request Intercepts (standalone) ---

// ToolRequestIntercepts runs the registered tool request intercept chain and
// returns the transformed arguments. This is a shorthand for
// [nat_nexus.ToolRequestIntercepts].
func ToolRequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nat_nexus.ToolRequestIntercepts(name, args)
}

// --- Tool Response Intercepts (standalone) ---

// ToolResponseIntercepts runs the registered tool response intercept chain and
// returns the transformed result. This is a shorthand for
// [nat_nexus.ToolResponseIntercepts].
func ToolResponseIntercepts(name string, result json.RawMessage) (json.RawMessage, error) {
	return nat_nexus.ToolResponseIntercepts(name, result)
}

// --- LLM Request Intercepts (standalone) ---

// LlmRequestIntercepts runs the registered LLM request intercept chain and
// returns the transformed request. This is a shorthand for
// [nat_nexus.LlmRequestIntercepts].
func LlmRequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	return nat_nexus.LlmRequestIntercepts(request)
}
