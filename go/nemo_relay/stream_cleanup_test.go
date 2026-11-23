// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type scopedCleanupSource struct {
	early   bool
	emitted bool
	blocked chan struct{}
	closed  atomic.Int32
}

func (source *scopedCleanupSource) Next(ctx context.Context) (json.RawMessage, error) {
	if !source.emitted {
		source.emitted = true
		return json.RawMessage(`1`), nil
	}
	if source.early {
		close(source.blocked)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	return nil, io.EOF
}
func (source *scopedCleanupSource) Close() error {
	source.closed.Add(1)
	return EmitEvent("provider-cleanup", WithEventData(json.RawMessage(`{"private":true}`)))
}

func TestProviderCleanupEventsPreserveScope(t *testing.T) {
	for _, early := range []bool{false, true} {
		name := "exhausted"
		if early {
			name = "cancelled"
		}
		t.Run(name, func(t *testing.T) {
			runWithTestScopeStack(t, func() {
				owner, err := PushScope("cleanup-owner", ScopeTypeAgent)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := PopScope(owner); err != nil {
						t.Error(err)
					}
				}()
				if err := ScopeRegisterMarkSanitizeGuardrail(owner.UUID(), "cleanup-sanitizer", 0, func(event Event, fields EventSanitizeFields) EventSanitizeFields {
					if event.Name() == "provider-cleanup" {
						fields.Data = json.RawMessage(`{"redacted":true}`)
					}
					return fields
				}); err != nil {
					t.Fatal(err)
				}
				var mu sync.Mutex
				var llmID string
				var mark Event
				if err := RegisterSubscriber("cleanup-events", func(event Event) {
					mu.Lock()
					defer mu.Unlock()
					if event.Name() == "cleanup-stream" && event.ScopeCategory() == "start" {
						llmID = event.UUID()
					}
					if event.Name() == "provider-cleanup" {
						mark = event
					}
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterSubscriber("cleanup-events")
				source := &scopedCleanupSource{early: early, blocked: make(chan struct{})}
				stream, err := LlmStreamCallExecute("cleanup-stream", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
					return source, nil
				}, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer stream.Close()
				if early {
					if _, err := stream.Next(); err != nil {
						t.Fatal(err)
					}
					select {
					case <-source.blocked:
					case <-time.After(5 * time.Second):
						t.Fatal("source read did not block")
					}
				} else {
					drainStream(t, stream)
				}
				if err := stream.Close(); err != nil {
					t.Fatal(err)
				}
				if err := FlushSubscribers(); err != nil {
					t.Fatal(err)
				}
				mu.Lock()
				defer mu.Unlock()
				if mark == nil || llmID == "" || mark.ParentUUID() != llmID {
					t.Fatalf("cleanup mark lost LLM parent: expected=%q event=%v", llmID, mark)
				}
				assertSemanticJSONEqual(t, json.RawMessage(`{"redacted":true}`), mark.Data())
				if source.closed.Load() != 1 {
					t.Fatalf("cleanup ran %d times", source.closed.Load())
				}
			})
		})
	}
}

func TestStreamAdapterPropagatesReplacedSourceCloseError(t *testing.T) {
	runWithTestScopeStack(t, func() {
		const name = "adapter-close-failure"
		if err := RegisterLlmStreamExecutionIntercept(name, 0, LLMExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			if _, err := next(request); err != nil {
				return nil, err
			}
			return next(request)
		}).AsStream()); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmStreamExecutionIntercept(name)
		source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`)}, closeErr: errors.New("first source close failed")}
		var opens atomic.Int32
		stream, err := LlmStreamCallExecute("adapter-replacement", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) { opens.Add(1); return source, nil }, nil, nil)
		if stream != nil {
			_ = stream.Close()
		}
		if err == nil || !strings.Contains(err.Error(), "first source close failed") {
			t.Fatalf("cleanup error lost: %v", err)
		}
		if opens.Load() != 1 || source.closed.Load() != 1 {
			t.Fatalf("source reopened or closed twice: opens=%d closes=%d", opens.Load(), source.closed.Load())
		}
	})
}

func TestStreamAdapterJoinsInterceptorAndCleanupErrors(t *testing.T) {
	primary := errors.New("interceptor failed")
	cleanup := errors.New("source cleanup failed")
	source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`)}, closeErr: cleanup}
	adapter := LLMExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		if _, err := next(request); err != nil {
			return nil, err
		}
		return nil, primary
	}).AsStream()
	stream, err := adapter(json.RawMessage(`{}`), LLMExecutionContext{}, func(json.RawMessage) (LLMStreamSource, error) { return source, nil })
	if stream != nil || !errors.Is(err, primary) || !errors.Is(err, cleanup) {
		t.Fatalf("expected both errors, got stream=%v error=%v", stream, err)
	}
	if source.closed.Load() != 1 {
		t.Fatalf("source closed %d times", source.closed.Load())
	}
}
