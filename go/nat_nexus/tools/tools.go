// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package tools provides shorthand access to Nexus tool call operations.
//
// It re-exports the core tool lifecycle functions (ToolCall, ToolCallEnd,
// ToolCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/tools"
//
//	// Execute a tool call with an inline function.
//	result, err := tools.Execute("search", json.RawMessage(`{"q":"hello"}`),
//	    func(args json.RawMessage) (json.RawMessage, error) {
//	        // ... perform the search ...
//	        return json.RawMessage(`{"results":[]}`), nil
//	    },
//	)
package tools

import (
	"encoding/json"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

// Call starts a tool call lifecycle and returns a [nat_nexus.ToolHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nat_nexus.ToolCall].
func Call(name string, args json.RawMessage, opts ...nat_nexus.ToolCallOption) (*nat_nexus.ToolHandle, error) {
	return nat_nexus.ToolCall(name, args, opts...)
}

// CallEnd completes a tool call that was started with [Call], emitting an End
// event. This is a shorthand for [nat_nexus.ToolCallEnd].
func CallEnd(handle *nat_nexus.ToolHandle, result json.RawMessage, opts ...nat_nexus.ToolCallOption) error {
	return nat_nexus.ToolCallEnd(handle, result, opts...)
}

// Execute runs a complete tool call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, sanitize-response guardrails) and
// returns the final result JSON. This is a shorthand for
// [nat_nexus.ToolCallExecute].
func Execute(name string, args json.RawMessage, fn nat_nexus.ToolExecutionFunc, opts ...nat_nexus.ToolCallOption) (json.RawMessage, error) {
	return nat_nexus.ToolCallExecute(name, args, fn, opts...)
}

// RequestIntercepts runs the registered tool request intercept chain on the
// given arguments and returns the transformed arguments. This is a shorthand for
// [nat_nexus.ToolRequestIntercepts].
func RequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nat_nexus.ToolRequestIntercepts(name, args)
}

// ConditionalExecution runs the registered tool conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. This is a shorthand for [nat_nexus.ToolConditionalExecution].
func ConditionalExecution(name string, args json.RawMessage) error {
	return nat_nexus.ToolConditionalExecution(name, args)
}
