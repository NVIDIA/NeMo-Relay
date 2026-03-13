// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package llm provides shorthand access to NVMagic LLM call operations.
//
// It re-exports the core LLM lifecycle functions (LlmCall, LlmCallEnd,
// LlmCallExecute, LlmStreamCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvmagic/go/nvmagic/llm"
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

	"github.com/nvidia/nvmagic/go/nvmagic"
)

// Call starts an LLM call lifecycle and returns an [nvmagic.LLMHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nvmagic.LlmCall].
func Call(name string, native interface{}, opts ...nvmagic.LLMCallOption) (*nvmagic.LLMHandle, error) {
	return nvmagic.LlmCall(name, native, opts...)
}

// CallEnd completes an LLM call that was started with [Call], emitting an End
// event. This is a shorthand for [nvmagic.LlmCallEnd].
func CallEnd(handle *nvmagic.LLMHandle, response json.RawMessage, opts ...nvmagic.LLMCallOption) error {
	return nvmagic.LlmCallEnd(handle, response, opts...)
}

// Execute runs a complete LLM call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, response intercepts, sanitize-response
// guardrails) and returns the final response JSON. This is a shorthand for
// [nvmagic.LlmCallExecute].
func Execute(name string, native interface{}, fn nvmagic.LLMExecutionFunc, opts ...nvmagic.LLMCallOption) (json.RawMessage, error) {
	return nvmagic.LlmCallExecute(name, native, fn, opts...)
}

// StreamExecute runs a streaming LLM call lifecycle with the full middleware
// pipeline (conditional-execution guardrails run first on the raw request) and
// returns an [nvmagic.LlmStream] for consuming JSON chunks. This is a
// shorthand for [nvmagic.LlmStreamCallExecute].
//
// The collector callback is invoked with each intercepted chunk JSON for
// accumulation. The finalizer callback is invoked once when the stream is
// exhausted and must return a JSON string representing the aggregated response.
// Pass nil for either to use the default no-op behavior.
func StreamExecute(name string, native interface{}, fn nvmagic.LLMExecutionFunc, collector nvmagic.CollectorFunc, finalizer nvmagic.FinalizerFunc, opts ...nvmagic.LLMCallOption) (*nvmagic.LlmStream, error) {
	return nvmagic.LlmStreamCallExecute(name, native, fn, collector, finalizer, opts...)
}

// RequestIntercepts runs the registered LLM request intercept chain on the
// given request and returns the transformed request. This is a shorthand for
// [nvmagic.LlmRequestIntercepts].
func RequestIntercepts(request json.RawMessage) (json.RawMessage, error) {
	return nvmagic.LlmRequestIntercepts(request)
}

// ConditionalExecution runs the registered LLM conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. Optional [nvmagic.LLMCallOption] values can supply a
// [nvmagic.WithLLMToRequest] converter. This is a shorthand for
// [nvmagic.LlmConditionalExecution].
func ConditionalExecution(request json.RawMessage, opts ...nvmagic.LLMCallOption) error {
	return nvmagic.LlmConditionalExecution(request, opts...)
}
