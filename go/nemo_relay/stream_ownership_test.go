// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
	"weak"
)

type capturingStreamSource struct {
	stream *LlmStream
	closed *atomic.Int32
}

func (source *capturingStreamSource) Next(context.Context) (json.RawMessage, error) {
	runtime.KeepAlive(source.stream)
	return json.RawMessage(`1`), nil
}
func (source *capturingStreamSource) Close() error { source.closed.Add(1); return nil }

//go:noinline
func abandonedCapturingSource(t *testing.T, parent context.Context, read, intercepted bool, closed *atomic.Int32) (weak.Pointer[LlmStream], context.Context, []uintptr) {
	t.Helper()
	var stream *LlmStream
	var source *capturingStreamSource
	if intercepted {
		returnedSource := make(chan *capturingStreamSource, 1)
		if err := RegisterLlmStreamExecutionIntercept("capturing-source", 0, func(_ json.RawMessage, _ LLMExecutionContext, _ func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
			result := &capturingStreamSource{closed: closed}
			returnedSource <- result
			return result, nil
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmStreamExecutionIntercept("capturing-source")
		result, err := LlmStreamCallExecuteContext(parent, "capturing-intercept", requestForTest(makeRequest()),
			func(ctx context.Context, _ json.RawMessage) (LLMStreamSource, error) {
				return nil, errors.New("interceptor should short circuit")
			}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		stream = result
		source = <-returnedSource
	} else {
		source = &capturingStreamSource{closed: closed}
		var err error
		stream, err = LlmStreamCallExecuteContext(parent, "capturing-provider", requestForTest(makeRequest()),
			func(ctx context.Context, _ json.RawMessage) (LLMStreamSource, error) {
				runtime.KeepAlive(stream) // The provider factory may also retain the handle.
				return source, nil
			}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	source.stream = stream
	if read {
		if _, err := stream.Next(); err != nil {
			t.Fatal(err)
		}
	}
	// Record this stream's tokens rather than assuming the process has no pending
	// asynchronous frees from other tests.
	ids := streamCallbackTokens(stream.sources)
	if len(ids) == 0 {
		t.Fatal("no native callback tokens to observe")
	}
	ctx := stream.sources.factory.ctx
	return weak.Make(stream), ctx, ids
}

func streamCallbackTokens(sources *streamSources) []uintptr {
	ids := []uintptr{}
	closureRegistryMu.Lock()
	for id, value := range closureRegistry {
		switch value := value.(type) {
		case weak.Pointer[foreignSourceState]:
			if state := value.Value(); state != nil && state.owner.Value() == sources {
				ids = append(ids, id)
			}
		case weak.Pointer[streamFactory]:
			if value.Value() == sources.factory {
				ids = append(ids, id)
			}
		}
	}
	closureRegistryMu.Unlock()
	return ids
}

func streamTokensReleased(ids []uintptr) bool {
	closureRegistryMu.Lock()
	defer closureRegistryMu.Unlock()
	for _, id := range ids {
		if _, exists := closureRegistry[id]; exists {
			return false
		}
	}
	return true
}

func TestCapturingProviderSourcesDoNotRootManagedStream(t *testing.T) {
	for _, intercepted := range []bool{false, true} {
		for _, read := range []bool{false, true} {
			name := "provider"
			if intercepted {
				name = "intercept"
			}
			if read {
				name += "-started"
			} else {
				name += "-unstarted"
			}
			t.Run(name, func(t *testing.T) {
				parent, cancel := context.WithCancel(context.Background())
				defer cancel()
				var closed atomic.Int32
				ref, providerContext, ids := abandonedCapturingSource(t, parent, read, intercepted, &closed)
				deadline := time.Now().Add(5 * time.Second)
				for time.Now().Before(deadline) {
					runtime.GC()
					if !streamStillReachable(ref) && providerContext.Err() != nil && streamTokensReleased(ids) {
						break
					}
					time.Sleep(10 * time.Millisecond)
				}
				if streamStillReachable(ref) {
					t.Fatal("provider callback retained its abandoned managed stream")
				}
				if providerContext.Err() != context.Canceled {
					t.Fatalf("provider context was not cancelled: %v", providerContext.Err())
				}
				if !streamTokensReleased(ids) {
					t.Fatal("native callback state was not released")
				}
				if parent.Err() != nil {
					t.Fatalf("parent was cancelled: %v", parent.Err())
				}
				if closed.Load() > 1 {
					t.Fatalf("provider closed %d times", closed.Load())
				}
				runtime.KeepAlive(parent)
			})
		}
	}
}

func TestStreamSourcesSurviveGCThroughNestedContinuations(t *testing.T) {
	if err := RegisterLlmStreamExecutionIntercept("gc-outer", 0, LLMExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		runtime.GC()
		return next(request)
	}).AsStream()); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmStreamExecutionIntercept("gc-outer")
	if err := RegisterLlmStreamExecutionIntercept("gc-lazy", 1, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
		runtime.GC()
		return &lazyOpenSource{request: request, open: next}, nil
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmStreamExecutionIntercept("gc-lazy")
	source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`), json.RawMessage(`2`)}}
	stream, err := LlmStreamCallExecute("gc-nested", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
		runtime.GC()
		return source, nil
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for _, want := range []string{"1", "2"} {
		runtime.GC()
		got, err := stream.Next()
		if err != nil || string(got) != want {
			t.Fatalf("Next = %s, %v; want %s", got, err, want)
		}
	}
	if _, err := stream.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Next = %v; want EOF", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if source.closed.Load() != 1 {
		t.Fatalf("provider closed %d times", source.closed.Load())
	}
}

func TestNativeOwnedSourceSurvivesGCWithoutManagedHandle(t *testing.T) {
	source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`)}}
	ptr, err := foreignStreamSource(source, context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	native := ownedStreamSource(ptr)
	defer native.Close()
	runtime.GC()
	got, err := native.Next(context.Background())
	if err != nil || string(got) != "1" {
		t.Fatalf("Next = %s, %v", got, err)
	}
	if err := native.Close(); err != nil {
		t.Fatal(err)
	}
	if source.closed.Load() != 1 {
		t.Fatalf("provider closed %d times", source.closed.Load())
	}
}

//go:noinline
func abandonedStreamWithBlockedRead(t *testing.T, parent context.Context) (weak.Pointer[LlmStream], context.Context, []uintptr) {
	t.Helper()
	var stream *LlmStream
	var reads atomic.Int32
	blocked := make(chan struct{})
	source := &contractSource{next: func(ctx context.Context) (json.RawMessage, error) {
		if reads.Add(1) == 1 {
			return json.RawMessage(`1`), nil
		}
		close(blocked)
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	var err error
	stream, err = LlmStreamCallExecuteContext(parent, "blocked-factory", requestForTest(makeRequest()),
		func(context.Context, json.RawMessage) (LLMStreamSource, error) {
			runtime.KeepAlive(stream) // Capture the handle in the factory, not the read.
			return source, nil
		}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(); err != nil {
		_ = stream.Close()
		t.Fatal(err)
	}
	select {
	case <-blocked:
	case <-time.After(5 * time.Second):
		_ = stream.Close()
		t.Fatal("provider did not start its second read")
	}
	return weak.Make(stream), stream.sources.factory.ctx, streamCallbackTokens(stream.sources)
}

func TestBlockedProviderReadDoesNotRetainFactory(t *testing.T) {
	for _, intercepted := range []bool{false, true} {
		name := "direct"
		if intercepted {
			name = "lazy-interceptor"
		}
		t.Run(name, func(t *testing.T) {
			if intercepted {
				if err := RegisterLlmStreamExecutionIntercept("blocked-read-lazy", 0, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
					return &lazyOpenSource{request: request, open: next}, nil
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmStreamExecutionIntercept("blocked-read-lazy")
			}
			parent, cancel := context.WithCancel(context.Background())
			defer cancel()
			ref, providerContext, ids := abandonedStreamWithBlockedRead(t, parent)
			deadline := time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				runtime.GC()
				if !streamStillReachable(ref) && providerContext.Err() != nil && streamTokensReleased(ids) {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			if streamStillReachable(ref) {
				t.Fatal("blocked read retained the provider factory and its abandoned stream")
			}
			if providerContext.Err() != context.Canceled {
				t.Fatalf("provider context was not cancelled: %v", providerContext.Err())
			}
			if !streamTokensReleased(ids) {
				t.Fatal("native callback tokens were not released after the blocked read ended")
			}
			if parent.Err() != nil {
				t.Fatalf("parent context was cancelled: %v", parent.Err())
			}
			runtime.KeepAlive(parent)
		})
	}
}
