// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestOwnedCodecRequestAndResponseOperations(t *testing.T) {
	codec := NewOpenAIChatCodec()
	request := LLMRequest{Headers: json.RawMessage(`{"x-test":"kept"}`), Content: json.RawMessage(`{"model":"before","messages":[{"role":"user","content":"hello"}],"temperature":0.5}`)}
	annotated, err := codec.Decode(request)
	if err != nil {
		t.Fatal(err)
	}
	var annotation map[string]any
	if err := json.Unmarshal(annotated, &annotation); err != nil {
		t.Fatal(err)
	}
	annotation["model"] = "after"
	changed, _ := json.Marshal(annotation)
	encoded, err := codec.Encode(changed, request)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONFieldString(t, encoded.Content, "model", "after")
	assertJSONFieldString(t, encoded.Headers, "x-test", "kept")
	var content map[string]any
	if err := json.Unmarshal(encoded.Content, &content); err != nil {
		t.Fatal(err)
	}
	if content["temperature"] != 0.5 {
		t.Fatalf("lost unmodified field: %s", encoded.Content)
	}
	response, err := codec.DecodeResponse(json.RawMessage(`{"id":"answer","model":"after","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONFieldString(t, response, "id", "answer")
	if _, err := NewOCIGenAIChatCodec().Decode(LLMRequest{Content: json.RawMessage(`{"modelId":"oci-model","chatRequest":{"apiFormat":"GENERIC","messages":[]}}`)}); err != nil {
		t.Fatal(err)
	}
}

func TestLLMCodecParityManualAndManaged(t *testing.T) {
	runWithTestScopeStack(t, func() {
		var mu sync.Mutex
		events := map[string][]json.RawMessage{}
		if err := RegisterSubscriber("codec-parity", func(event Event) {
			if event.Category() != "llm" {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			key := event.Name() + "/" + event.ScopeCategory()
			if event.ScopeCategory() == "start" {
				events[key] = append(events[key], append(json.RawMessage(nil), event.AnnotatedRequest()...))
			}
			if event.ScopeCategory() == "end" {
				events[key] = append(events[key], append(json.RawMessage(nil), event.AnnotatedResponse()...))
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("codec-parity")
		req := map[string]any{"headers": map[string]any{}, "content": map[string]any{"model": "request-model", "messages": []any{}}}
		response := json.RawMessage(`{"id":"decoded","model":"response-model","choices":[]}`)
		codec := NewOpenAIChatCodec()
		h, err := LlmCall("manual-codec", requestForTest(req), WithLLMCodec(codec))
		if err != nil {
			t.Fatal(err)
		}
		if err := LlmCallEnd(h, response, WithLLMResponseCodec(codec)); err != nil {
			t.Fatal(err)
		}
		h, err = LlmCall("manual-annotation", requestForTest(req), WithLLMAnnotatedRequest(json.RawMessage(`{"model":"precomputed-request"}`)))
		if err != nil {
			t.Fatal(err)
		}
		if err := LlmCallEnd(h, response, WithLLMAnnotatedResponse(json.RawMessage(`{"model":"precomputed-response"}`))); err != nil {
			t.Fatal(err)
		}
		h, err = LlmCall("manual-callback", requestForTest(req), WithLLMCodec(CodecFunc{
			Decode: func(_, _ json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"model":"callback-request"}`), nil
			},
			Encode: func(_ json.RawMessage, _ json.RawMessage, content json.RawMessage) (json.RawMessage, error) {
				return content, nil
			},
		}))
		if err != nil {
			t.Fatal(err)
		}
		if err := LlmCallEnd(h, response); err != nil {
			t.Fatal(err)
		}
		custom, err := NewCustomResponseCodec(func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"model":"custom-response"}`), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		_, err = LlmCallExecute("managed-codec", requestForTest(req), func(json.RawMessage) (json.RawMessage, error) { return response, nil }, WithLLMCodec(codec), WithLLMResponseCodec(custom))
		if err != nil {
			t.Fatal(err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		for key, model := range map[string]string{
			"manual-codec/start": "request-model", "manual-codec/end": "response-model",
			"manual-annotation/start": "precomputed-request", "manual-annotation/end": "precomputed-response",
			"manual-callback/start": "callback-request", "managed-codec/start": "request-model", "managed-codec/end": "custom-response",
		} {
			if len(events[key]) != 1 {
				t.Fatalf("expected one %s event: %v", key, events)
			}
			assertJSONFieldString(t, events[key][0], "model", model)
		}
	})
}

func TestLLMExecutionOptimizationRecorder(t *testing.T) {
	runWithTestScopeStack(t, func() {
		var retained LLMExecutionContext
		calls := 0
		if err := RegisterLlmExecutionIntercept("record-optimization", 0, func(request json.RawMessage, context LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			retained = context
			calls++
			if context.RequestCodec.Codec.CodecKind != LLMCodecBuiltin {
				return nil, errors.New("built-in request codec identity was lost")
			}
			accepted, err := context.RecordOptimizationContribution(LLMOptimizationContribution{Producer: "go-test", Kind: "custom", Applied: true})
			if err != nil {
				return nil, err
			}
			if !accepted {
				return nil, errors.New("optimization evidence rejected")
			}
			return next(request)
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmExecutionIntercept("record-optimization")
		var mu sync.Mutex
		var end json.RawMessage
		if err := RegisterSubscriber("optimization-parity", func(event Event) {
			if event.Name() == "optimization-parity" && event.ScopeCategory() == "end" {
				mu.Lock()
				end = append(json.RawMessage(nil), event.AnnotatedResponse()...)
				mu.Unlock()
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("optimization-parity")
		_, err := LlmCallExecute("optimization-parity", requestForTest(makeRequest()), func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"model":"test","choices":[]}`), nil
		}, WithLLMCodec(NewOpenAIChatCodec()), WithLLMResponseCodec(NewOpenAIChatCodec()))
		if err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("expected interceptor: %d", calls)
		}
		if _, err := retained.RecordOptimizationContribution(LLMOptimizationContribution{Producer: "late", Kind: "custom"}); !errors.Is(err, ErrLLMCodecExpired) {
			t.Fatalf("expected expired capability, got %v", err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		if !strings.Contains(string(end), `"go-test"`) {
			t.Fatalf("missing recorded contribution in End annotation: %s", end)
		}
	})
}

func TestForkScopeStackPreservesParentAndIsolatesChild(t *testing.T) {
	runWithTestScopeStack(t, func() {
		parent, err := PushScope("fork-parent", ScopeTypeAgent)
		if err != nil {
			t.Fatal(err)
		}
		defer PopScope(parent)
		child, err := ForkScopeStack()
		if err != nil {
			t.Fatal(err)
		}
		defer child.Close()
		done := make(chan error, 1)
		go func() {
			child.Run(func() {
				context, err := CapturePropagationContext()
				if err != nil {
					done <- err
					return
				}
				if context.ParentUUID != parent.UUID() {
					done <- errors.New("fork lost parent")
					return
				}
				scope, err := PushScope("child", ScopeTypeAgent)
				if err == nil {
					err = PopScope(scope)
				}
				done <- err
			})
		}()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		current, err := GetHandle()
		if err != nil {
			t.Fatal(err)
		}
		if current.UUID() != parent.UUID() {
			t.Fatal("child changed parent stack")
		}
	})
}

func TestRequestCodecCallbackStateIsReleasedBeforeNativeEntryFailures(t *testing.T) {
	count := func() int { closureRegistryMu.Lock(); defer closureRegistryMu.Unlock(); return len(closureRegistry) }
	baseline := count()
	codec := CodecFunc{
		Decode: func(_, _ json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{}`), nil },
		Encode: func(_ json.RawMessage, _ json.RawMessage, content json.RawMessage) (json.RawMessage, error) {
			return content, nil
		},
	}
	invalid := LLMRequest{Content: json.RawMessage(`{`)}
	if _, err := LlmCall("invalid", requestForTest(invalid), WithLLMCodec(codec)); err == nil {
		t.Fatal("expected invalid request")
	}
	if count() != baseline {
		t.Fatal("manual validation leaked callback state")
	}
	if _, err := LlmCallExecute("invalid", requestForTest(invalid), func(json.RawMessage) (json.RawMessage, error) { return nil, nil }, WithLLMCodec(codec)); err == nil {
		t.Fatal("expected invalid request")
	}
	if count() != baseline {
		t.Fatal("managed validation leaked callback state")
	}
	if _, err := LlmCallExecute("invalid", requestForTest(makeRequest()), func(json.RawMessage) (json.RawMessage, error) { return nil, nil }, WithLLMCodec(codec), WithLLMAnnotatedRequest(json.RawMessage(`{}`))); err == nil {
		t.Fatal("expected manual annotation rejection")
	}
	if count() != baseline {
		t.Fatal("annotation validation leaked callback state")
	}
}

func TestLLMStreamingOptimizationRecorder(t *testing.T) {
	runWithTestScopeStack(t, func() {
		recorded := false
		if err := RegisterLlmStreamExecutionIntercept("stream-record-optimization", 0, LLMExecutionInterceptFunc(func(request json.RawMessage, context LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			if context.ResponseCodec != nil {
				return nil, errors.New("streaming context has a unary response codec")
			}
			accepted, err := context.RecordOptimizationContribution(LLMOptimizationContribution{Producer: "go-stream-test", Kind: "custom", Applied: true})
			if err != nil {
				return nil, err
			}
			if !accepted {
				return nil, errors.New("stream contribution rejected")
			}
			recorded = true
			return next(request)
		}).AsStream()); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmStreamExecutionIntercept("stream-record-optimization")
		stream, err := LlmStreamCallExecute("stream-optimization", requestForTest(makeRequest()), LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"chunk":true}`), nil }).AsStream(), nil, nil, WithLLMCodec(NewOpenAIChatCodec()))
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		if _, err := stream.Next(); err != nil {
			t.Fatal(err)
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		if !recorded {
			t.Fatal("streaming intercept did not record evidence")
		}
	})
}
