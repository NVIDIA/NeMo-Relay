// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package llm provides shorthand access to NVAgentRT LLM call operations.
//
// It re-exports the core LLM lifecycle functions (LlmCall, LlmCallEnd,
// LlmCallExecute, LlmStreamCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/llm"
//
//	native := map[string]interface{}{"model": "gpt-4", "messages": []interface{}{}}
//	result, err := llm.Execute("chat", native,
//	    func(nativeJSON json.RawMessage) (json.RawMessage, error) {
//	        // ... call the LLM API ...
//	        return json.RawMessage(`{"choices":[]}`), nil
//	    },
//	)
package llm

import (
	"encoding/json"

	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// Call starts an LLM call lifecycle and returns an [nvagentrt.LLMHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nvagentrt.LlmCall].
func Call(name string, native interface{}, opts ...nvagentrt.LLMCallOption) (*nvagentrt.LLMHandle, error) {
	return nvagentrt.LlmCall(name, native, opts...)
}

// CallEnd completes an LLM call that was started with [Call], emitting an End
// event. This is a shorthand for [nvagentrt.LlmCallEnd].
func CallEnd(handle *nvagentrt.LLMHandle, response json.RawMessage, opts ...nvagentrt.LLMCallOption) error {
	return nvagentrt.LlmCallEnd(handle, response, opts...)
}

// Execute runs a complete LLM call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, response intercepts, sanitize-response
// guardrails) and returns the final response JSON. This is a shorthand for
// [nvagentrt.LlmCallExecute].
func Execute(name string, native interface{}, fn nvagentrt.LLMExecutionFunc, opts ...nvagentrt.LLMCallOption) (json.RawMessage, error) {
	return nvagentrt.LlmCallExecute(name, native, fn, opts...)
}

// StreamExecute runs a streaming LLM call lifecycle with the full middleware
// pipeline (conditional-execution guardrails run first on the raw request) and
// returns an [nvagentrt.LlmStream] for consuming JSON chunks. This is a
// shorthand for [nvagentrt.LlmStreamCallExecute].
//
// The collector callback is invoked with each intercepted chunk JSON for
// accumulation. The finalizer callback is invoked once when the stream is
// exhausted and must return a JSON string representing the aggregated response.
// Pass nil for either to use the default no-op behavior.
func StreamExecute(name string, native interface{}, fn nvagentrt.LLMExecutionFunc, collector nvagentrt.CollectorFunc, finalizer nvagentrt.FinalizerFunc, opts ...nvagentrt.LLMCallOption) (*nvagentrt.LlmStream, error) {
	return nvagentrt.LlmStreamCallExecute(name, native, fn, collector, finalizer, opts...)
}

// RequestIntercepts runs the registered LLM request intercept chain on the
// given request and returns the transformed request. This is a shorthand for
// [nvagentrt.LlmRequestIntercepts].
func RequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.LlmRequestIntercepts(request)
}

// ConditionalExecution runs the registered LLM conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. Optional [nvagentrt.LLMCallOption] values can supply a
// [nvagentrt.WithLLMToRequest] converter. This is a shorthand for
// [nvagentrt.LlmConditionalExecution].
func ConditionalExecution(request json.RawMessage, opts ...nvagentrt.LLMCallOption) error {
	return nvagentrt.LlmConditionalExecution(request, opts...)
}

// ResponseIntercepts runs the registered LLM response intercept chain on the
// given response and returns the transformed response. This is a shorthand for
// [nvagentrt.LlmResponseIntercepts].
func ResponseIntercepts(response json.RawMessage) (json.RawMessage, error) {
	return nvagentrt.LlmResponseIntercepts(response)
}
