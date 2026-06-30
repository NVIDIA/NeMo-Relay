// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package llm provides shorthand access to NeMo Relay LLM call operations.
//
// It re-exports the core LLM lifecycle functions (LlmCall, LlmCallEnd,
// LlmCallExecute, LlmStreamCallExecute) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/llm"
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

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

// Call starts an LLM call lifecycle and returns an [nemo_relay.LLMHandle],
// emitting a Start event. End the call with [CallEnd]. This is a shorthand for
// [nemo_relay.LlmCall].
func Call(name string, native interface{}, opts ...nemo_relay.LLMCallOption) (*nemo_relay.LLMHandle, error) {
	return nemo_relay.LlmCall(name, native, opts...)
}

// CallEnd completes an LLM call that was started with [Call], emitting an End
// event. This is a shorthand for [nemo_relay.LlmCallEnd].
func CallEnd(handle *nemo_relay.LLMHandle, response json.RawMessage, opts ...nemo_relay.LLMCallOption) error {
	return nemo_relay.LlmCallEnd(handle, response, opts...)
}

// Execute runs a complete LLM call lifecycle with the full middleware pipeline
// (conditional-execution guardrails, request intercepts, sanitize-request
// guardrails, execution intercepts, fn, sanitize-response
// guardrails) and returns the final response JSON. This is a shorthand for
// [nemo_relay.LlmCallExecute].
func Execute(name string, native interface{}, fn nemo_relay.LLMExecutionFunc, opts ...nemo_relay.LLMCallOption) (json.RawMessage, error) {
	return nemo_relay.LlmCallExecute(name, native, fn, opts...)
}

// StreamExecute runs a streaming LLM call lifecycle with the full middleware
// pipeline (conditional-execution guardrails run first on the raw request) and
// returns an [nemo_relay.LlmStream] for consuming JSON chunks. This is a
// shorthand for [nemo_relay.LlmStreamCallExecute].
//
// The collector callback is invoked with each intercepted chunk JSON for
// accumulation. The finalizer callback is invoked once when the stream is
// exhausted and must return a JSON string representing the aggregated response.
// Pass nil for either to use the default no-op behavior.
func StreamExecute(name string, native interface{}, fn nemo_relay.LLMExecutionFunc, collector nemo_relay.CollectorFunc, finalizer nemo_relay.FinalizerFunc, opts ...nemo_relay.LLMCallOption) (*nemo_relay.LlmStream, error) {
	return nemo_relay.LlmStreamCallExecute(name, native, fn, collector, finalizer, opts...)
}

// RequestIntercepts runs the registered LLM request intercept chain on the
// given request and returns the transformed request. This is a shorthand for
// [nemo_relay.LlmRequestIntercepts].
func RequestIntercepts(name string, request json.RawMessage) (nemo_relay.LLMRequestInterceptOutcome, error) {
	return nemo_relay.LlmRequestIntercepts(name, request)
}

// ConditionalExecution runs the registered LLM conditional execution guardrail
// chain. Returns nil if all guardrails pass, or an error with the rejection
// reason if blocked. The request should be in LLMRequest JSON format
// ({"headers": {...}, "content": {...}}). This is a shorthand for
// [nemo_relay.LlmConditionalExecution].
func ConditionalExecution(request json.RawMessage) error {
	return nemo_relay.LlmConditionalExecution(request)
}
