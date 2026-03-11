// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package tools provides shorthand access to NVMagic tool call operations.
//
// It re-exports the core tool lifecycle functions (ToolCall, ToolCallEnd,
// ToolCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvmagic/go/nvmagic/tools"
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

	"github.com/nvidia/nvmagic/go/nvmagic"
)

// Call starts a tool call lifecycle and returns a [nvmagic.ToolHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nvmagic.ToolCall].
func Call(name string, args json.RawMessage, opts ...nvmagic.ToolCallOption) (*nvmagic.ToolHandle, error) {
	return nvmagic.ToolCall(name, args, opts...)
}

// CallEnd completes a tool call that was started with [Call], emitting an End
// event. This is a shorthand for [nvmagic.ToolCallEnd].
func CallEnd(handle *nvmagic.ToolHandle, result json.RawMessage, opts ...nvmagic.ToolCallOption) error {
	return nvmagic.ToolCallEnd(handle, result, opts...)
}

// Execute runs a complete tool call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, response intercepts, sanitize-response
// guardrails) and returns the final result JSON. This is a shorthand for
// [nvmagic.ToolCallExecute].
func Execute(name string, args json.RawMessage, fn nvmagic.ToolExecutionFunc, opts ...nvmagic.ToolCallOption) (json.RawMessage, error) {
	return nvmagic.ToolCallExecute(name, args, fn, opts...)
}

// RequestIntercepts runs the registered tool request intercept chain on the
// given arguments and returns the transformed arguments. This is a shorthand for
// [nvmagic.ToolRequestIntercepts].
func RequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nvmagic.ToolRequestIntercepts(name, args)
}

// ConditionalExecution runs the registered tool conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. This is a shorthand for [nvmagic.ToolConditionalExecution].
func ConditionalExecution(name string, args json.RawMessage) error {
	return nvmagic.ToolConditionalExecution(name, args)
}

// ResponseIntercepts runs the registered tool response intercept chain on the
// given result and returns the transformed result. This is a shorthand for
// [nvmagic.ToolResponseIntercepts].
func ResponseIntercepts(name string, result json.RawMessage) (json.RawMessage, error) {
	return nvmagic.ToolResponseIntercepts(name, result)
}
