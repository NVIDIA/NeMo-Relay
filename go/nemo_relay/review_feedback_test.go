// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"runtime"
	"sync"
	"testing"
	"time"
	"weak"
)

func TestJSONInputsRejectNULTrailers(t *testing.T) {
	codec := NewOpenAIChatCodec()
	defer codec.Close()
	request := requestForTest(makeRequest())
	annotation, err := codec.Decode(request)
	if err != nil {
		t.Fatal(err)
	}
	bad := func(value json.RawMessage) json.RawMessage {
		return append(append(json.RawMessage(nil), value...), []byte("\x00trailer")...)
	}
	tests := map[string]func() error{
		"agent extra": func() error {
			_, err := NewAtifExporter("review-json", "agent", "1", "model", WithAtifAgentExtra(bad(json.RawMessage(`{}`))))
			return err
		},
		"category profile": func() error {
			return EmitEvent("invalid-profile", WithEventCategoryProfile(bad(json.RawMessage(`{}`))))
		},
		"request annotation": func() error {
			_, err := LlmCall("invalid-request-annotation", request, WithLLMAnnotatedRequest(bad(annotation)))
			return err
		},
		"response annotation": func() error {
			handle, err := LlmCall("invalid-response-annotation", request)
			if err != nil {
				return err
			}
			defer LlmCallEnd(handle, json.RawMessage(`{}`))
			return LlmCallEnd(handle, json.RawMessage(`{}`), WithLLMAnnotatedResponse(bad(json.RawMessage(`{}`))))
		},
		"decode headers": func() error {
			_, err := codec.Decode(LLMRequest{Headers: bad(json.RawMessage(`{}`)), Content: request.Content})
			return err
		},
		"decode content": func() error {
			_, err := codec.Decode(LLMRequest{Headers: request.Headers, Content: bad(request.Content)})
			return err
		},
		"encode headers": func() error {
			_, err := codec.Encode(annotation, LLMRequest{Headers: bad(json.RawMessage(`{}`)), Content: request.Content})
			return err
		},
		"encode content": func() error {
			_, err := codec.Encode(annotation, LLMRequest{Headers: request.Headers, Content: bad(request.Content)})
			return err
		},
		"encode annotation": func() error { _, err := codec.Encode(bad(annotation), request); return err },
		"decode response":   func() error { _, err := codec.DecodeResponse(bad(json.RawMessage(`{}`))); return err },
	}
	runWithTestScopeStack(t, func() {
		for name, run := range tests {
			t.Run(name, func(t *testing.T) {
				if err := run(); !errors.Is(err, ErrInvalidJSON) {
					t.Fatalf("got %v, want invalid JSON", err)
				}
			})
		}
		if err := EmitEvent("cleared-profile", WithEventCategoryProfile(bad(json.RawMessage(`{}`))), WithEventCategoryProfile(nil)); err != nil {
			t.Fatal(err)
		}
		handle, err := LlmCall("replaced-annotation", request, WithLLMAnnotatedRequest(bad(annotation)), WithLLMAnnotatedRequest(annotation))
		if err != nil {
			t.Fatal(err)
		}
		if err := LlmCallEnd(handle, json.RawMessage(`{}`)); err != nil {
			t.Fatal(err)
		}
	})
}

func TestHeaderlessLifecycleRequestsUseDefaults(t *testing.T) {
	runWithTestScopeStack(t, func() {
		codec := NewOpenAIChatCodec()
		defer codec.Close()
		for _, headers := range []json.RawMessage{nil, json.RawMessage(`{}`)} {
			request := LLMRequest{Headers: headers, Content: json.RawMessage(`{"model":"example","messages":[]}`)}
			handle, err := LlmCall("header-defaults", request)
			if err != nil {
				t.Fatal(err)
			}
			if err := LlmCallEnd(handle, json.RawMessage(`{}`)); err != nil {
				t.Fatal(err)
			}
			if _, err := codec.Decode(request); err != nil {
				t.Fatal(err)
			}
			check := func(value json.RawMessage) (json.RawMessage, error) {
				var got LLMRequest
				if err := json.Unmarshal(value, &got); err != nil {
					return nil, err
				}
				var headers map[string]json.RawMessage
				if err := json.Unmarshal(got.Headers, &headers); err != nil {
					t.Errorf("invalid headers: %s", got.Headers)
				}
				// Managed execution adds trace propagation headers.
				if len(headers) != 1 || headers["traceparent"] == nil {
					t.Errorf("unexpected headers: %s", got.Headers)
				}
				return json.RawMessage(`{}`), nil
			}
			if _, err := LlmCallExecute("managed-defaults", request, check); err != nil {
				t.Fatal(err)
			}
			stream, err := LlmStreamCallExecute("stream-defaults", request, LLMExecutionFunc(check).AsStream(), nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := stream.Next(); err != nil {
				t.Fatal(err)
			}
			if err := stream.Close(); err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestContextCallsPreserveSharedOptionSlices(t *testing.T) {
	llmOptions := make([]LLMCallOption, 0, 1)
	toolOptions := make([]ToolCallOption, 0, 1)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := LlmCallExecuteContext(ctx, "shared-options", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (json.RawMessage, error) {
				return nil, errors.New("unexpected provider call")
			}, llmOptions...); !errors.Is(err, context.Canceled) {
				t.Errorf("LLM: %v", err)
			}
			if _, err := LlmStreamCallExecuteContext(ctx, "shared-stream-options", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) { return &contractSource{}, nil }, nil, nil, llmOptions...); !errors.Is(err, context.Canceled) {
				t.Errorf("stream: %v", err)
			}
			if _, err := ToolCallExecuteContext(ctx, "shared-tool-options", json.RawMessage(`{}`), func(context.Context, json.RawMessage) (ToolExecutionResult, error) {
				return ToolExecutionResult{}, errors.New("unexpected provider call")
			}, toolOptions...); !errors.Is(err, context.Canceled) {
				t.Errorf("tool: %v", err)
			}
		}()
	}
	wg.Wait()
	if llmOptions[:1][0] != nil || toolOptions[:1][0] != nil {
		t.Fatal("context call changed caller option storage")
	}
}

func TestAlreadyCancelledNextContextClosesProvider(t *testing.T) {
	var providerContext context.Context
	stream, err := LlmStreamCallExecute("cancelled-next", requestForTest(makeRequest()), func(ctx context.Context, _ json.RawMessage) (LLMStreamSource, error) {
		providerContext = ctx
		return &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`), json.RawMessage(`2`)}}, nil
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Next(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := stream.NextContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("NextContext: %v", err)
	}
	if providerContext.Err() != context.Canceled {
		t.Fatalf("provider context: %v", providerContext.Err())
	}
	if _, err := stream.Next(); err != io.EOF {
		t.Fatalf("Next: %v", err)
	}
}

//go:noinline
func abandonedStreamWithProviderHook(t *testing.T) (weak.Pointer[LlmStream], []uintptr) {
	stream, err := LlmStreamCallExecute("provider-hook-cycle", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) { return &contractSource{}, nil }, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Register through the actual provider context, as a provider can do after setup.
	ctx := stream.sources.factory.ctx
	context.AfterFunc(ctx, func() { runtime.KeepAlive(stream) })
	return weak.Make(stream), streamCallbackTokens(stream.sources)
}

func TestProviderHookDoesNotRetainAbandonedStream(t *testing.T) {
	ref, tokens := abandonedStreamWithProviderHook(t)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if !streamStillReachable(ref) && streamTokensReleased(tokens) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("provider cancellation hook retained stream or native callback state")
}

func TestProviderContextPreservesDeadlineAndRunsHooks(t *testing.T) {
	parent, cancelParent := context.WithDeadline(context.Background(), time.Now().Add(time.Hour))
	defer cancelParent()
	ctx, cancel := newProviderContext(parent)
	if got, ok := ctx.Deadline(); !ok {
		t.Fatal("missing deadline")
	} else if want, _ := parent.Deadline(); got != want {
		t.Fatal("changed deadline")
	}
	called := make(chan struct{})
	context.AfterFunc(ctx, func() { close(called) })
	cancel()
	select {
	case <-called:
	case <-time.After(time.Second):
		t.Fatal("cancellation hook did not run")
	}
	if ctx.Err() != context.Canceled {
		t.Fatalf("context error = %v", ctx.Err())
	}
	expired, expire := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer expire()
	child, stop := newProviderContext(expired)
	defer stop()
	if child.Err() != context.DeadlineExceeded {
		t.Fatalf("deadline error = %v", child.Err())
	}
}
