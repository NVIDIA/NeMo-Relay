// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package tools provides shorthand access to NeMo Relay tool call operations.
//
// It re-exports the core tool lifecycle functions (ToolCall, ToolCallEnd,
// ToolCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/tools"
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

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

// Call starts a tool call lifecycle and returns a [nemo_relay.ToolHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nemo_relay.ToolCall].
func Call(name string, args json.RawMessage, opts ...nemo_relay.ToolCallOption) (*nemo_relay.ToolHandle, error) {
	return nemo_relay.ToolCall(name, args, opts...)
}

// CallEnd completes a tool call that was started with [Call], emitting an End
// event. This is a shorthand for [nemo_relay.ToolCallEnd].
func CallEnd(handle *nemo_relay.ToolHandle, result json.RawMessage, opts ...nemo_relay.ToolCallOption) error {
	return nemo_relay.ToolCallEnd(handle, result, opts...)
}

// Execute runs a complete tool call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, sanitize-response guardrails) and
// returns the final result JSON. This is a shorthand for
// [nemo_relay.ToolCallExecute].
func Execute(name string, args json.RawMessage, fn nemo_relay.ToolExecutionFunc, opts ...nemo_relay.ToolCallOption) (json.RawMessage, error) {
	return nemo_relay.ToolCallExecute(name, args, fn, opts...)
}

// RequestIntercepts runs the registered tool request intercept chain on the
// given arguments and returns the transformed arguments. This is a shorthand for
// [nemo_relay.ToolRequestIntercepts].
func RequestIntercepts(name string, args json.RawMessage) (json.RawMessage, error) {
	return nemo_relay.ToolRequestIntercepts(name, args)
}

// ConditionalExecution runs the registered tool conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. This is a shorthand for [nemo_relay.ToolConditionalExecution].
func ConditionalExecution(name string, args json.RawMessage) error {
	return nemo_relay.ToolConditionalExecution(name, args)
}
