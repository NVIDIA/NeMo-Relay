// Package tools provides shorthand access to NVAgentRT tool call operations.
//
// It re-exports the core tool lifecycle functions (ToolCall, ToolCallEnd,
// ToolCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/tools"
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

	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// Call starts a tool call lifecycle and returns a [nvagentrt.ToolHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nvagentrt.ToolCall].
func Call(name string, args json.RawMessage, opts ...nvagentrt.ToolCallOption) (*nvagentrt.ToolHandle, error) {
	return nvagentrt.ToolCall(name, args, opts...)
}

// CallEnd completes a tool call that was started with [Call], emitting an End
// event. This is a shorthand for [nvagentrt.ToolCallEnd].
func CallEnd(handle *nvagentrt.ToolHandle, result json.RawMessage, opts ...nvagentrt.ToolCallOption) error {
	return nvagentrt.ToolCallEnd(handle, result, opts...)
}

// Execute runs a complete tool call lifecycle with the full middleware pipeline
// (request intercepts, guardrails, execution intercepts, fn, response
// intercepts, response guardrails) and returns the final result JSON. This is
// a shorthand for [nvagentrt.ToolCallExecute].
func Execute(name string, args json.RawMessage, fn nvagentrt.ToolExecutionFunc, opts ...nvagentrt.ToolCallOption) (json.RawMessage, error) {
	return nvagentrt.ToolCallExecute(name, args, fn, opts...)
}
