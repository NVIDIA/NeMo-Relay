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
//	req := nvagentrt.NewLLMRequest("POST", "https://api.example.com/v1/chat",
//	    map[string]interface{}{"Authorization": "Bearer tok"},
//	    map[string]interface{}{"model": "gpt-4", "messages": []interface{}{}},
//	)
//	result, err := llm.Execute("chat", req,
//	    func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
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
func Call(name string, request *nvagentrt.LLMRequest, opts ...nvagentrt.LLMCallOption) (*nvagentrt.LLMHandle, error) {
	return nvagentrt.LlmCall(name, request, opts...)
}

// CallEnd completes an LLM call that was started with [Call], emitting an End
// event. This is a shorthand for [nvagentrt.LlmCallEnd].
func CallEnd(handle *nvagentrt.LLMHandle, response json.RawMessage, opts ...nvagentrt.LLMCallOption) error {
	return nvagentrt.LlmCallEnd(handle, response, opts...)
}

// Execute runs a complete LLM call lifecycle with the full middleware pipeline
// (request intercepts, guardrails, execution intercepts, fn, response
// intercepts, response guardrails) and returns the final response JSON. This is
// a shorthand for [nvagentrt.LlmCallExecute].
func Execute(name string, request *nvagentrt.LLMRequest, fn nvagentrt.LLMExecutionFunc, opts ...nvagentrt.LLMCallOption) (json.RawMessage, error) {
	return nvagentrt.LlmCallExecute(name, request, fn, opts...)
}

// StreamExecute runs a streaming LLM call lifecycle with the full middleware
// pipeline and returns an [nvagentrt.LlmStream] for consuming SSE chunks. This
// is a shorthand for [nvagentrt.LlmStreamCallExecute].
//
// The collector callback is invoked with each intercepted chunk string for
// accumulation. The finalizer callback is invoked once when the stream is
// exhausted and must return a JSON string representing the aggregated response.
// Pass nil for either to use the default no-op behavior.
func StreamExecute(name string, request *nvagentrt.LLMRequest, fn nvagentrt.LLMExecutionFunc, collector nvagentrt.CollectorFunc, finalizer nvagentrt.FinalizerFunc, opts ...nvagentrt.LLMCallOption) (*nvagentrt.LlmStream, error) {
	return nvagentrt.LlmStreamCallExecute(name, request, fn, collector, finalizer, opts...)
}
