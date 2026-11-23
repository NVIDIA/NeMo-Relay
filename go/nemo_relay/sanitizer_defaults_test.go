// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"sync"
	"testing"
)

func TestRequestSanitizerNilFieldsPreserveObservabilityPayload(t *testing.T) {
	for _, test := range []struct {
		name    string
		request LLMRequest
		want    string
	}{
		{"headers", LLMRequest{Content: json.RawMessage(`{"safe":true}`)}, `{"headers":{},"content":{"safe":true}}`},
		{"content", LLMRequest{Headers: json.RawMessage(`{"x-safe":"kept"}`)}, `{"headers":{"x-safe":"kept"},"content":null}`},
		{"both", LLMRequest{}, `{"headers":{},"content":null}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			runWithTestScopeStack(t, func() {
				name := "sanitizer-nil-" + test.name
				var mu sync.Mutex
				var input json.RawMessage
				if err := RegisterSubscriber(name, func(event Event) {
					if event.Name() == name && event.ScopeCategory() == "start" {
						mu.Lock()
						input = event.Input()
						mu.Unlock()
					}
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterSubscriber(name)
				if err := RegisterLlmSanitizeRequestGuardrail(name, 0, func(LLMRequest, LLMSanitizeRequestContext) (LLMRequest, bool) {
					return test.request, false
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmSanitizeRequestGuardrail(name)
				if _, err := LlmCallExecute(name, requestForTest(makeRequest()), func(json.RawMessage) (json.RawMessage, error) {
					return json.RawMessage(`{}`), nil
				}); err != nil {
					t.Fatal(err)
				}
				if err := FlushSubscribers(); err != nil {
					t.Fatal(err)
				}
				mu.Lock()
				defer mu.Unlock()
				assertSemanticJSONEqual(t, input, json.RawMessage(test.want))
			})
		})
	}
}

func TestRequestSanitizerCodecNormalizesNilFields(t *testing.T) {
	runWithTestScopeStack(t, func() {
		if err := RegisterSubscriber("nil-codec-fields", func(Event) {}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("nil-codec-fields")
		codec, err := NewCustomCodec(&CodecFunc{
			Decode: func(_, _ json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"messages":[]}`), nil },
			Encode: func(_ json.RawMessage, _ json.RawMessage, content json.RawMessage) (json.RawMessage, error) {
				return content, nil
			},
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer codec.Close()
		type result struct {
			encoded LLMRequest
			err     error
		}
		observed := make(chan result, 1)
		if err := RegisterLlmSanitizeRequestGuardrail("nil-codec-fields", 0, func(request LLMRequest, context LLMSanitizeRequestContext) (LLMRequest, bool) {
			resolved := context.ResolveCodec()
			annotation, err := resolved.Decode(LLMRequest{})
			if err != nil {
				observed <- result{err: err}
				return request, false
			}
			encoded, err := resolved.Encode(annotation, LLMRequest{})
			observed <- result{encoded, err}
			return request, false
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmSanitizeRequestGuardrail("nil-codec-fields")
		if _, err := LlmCallExecute("nil-codec-fields", requestForTest(makeRequest()), func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		}, WithLLMCodec(codec)); err != nil {
			t.Fatal(err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		select {
		case got := <-observed:
			if got.err != nil {
				t.Fatal(got.err)
			}
			assertSemanticJSONEqual(t, got.encoded.Headers, json.RawMessage(`{}`))
			assertSemanticJSONEqual(t, got.encoded.Content, json.RawMessage(`null`))
		default:
			t.Fatal("request sanitizer did not run")
		}
	})
}
