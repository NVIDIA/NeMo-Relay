// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"
)

// Hide the underlying context's private values so cancellation uses the
// documented AfterFunc hook, allowing registrations to be counted directly.
type registrationCountingContext struct {
	context.Context
	registrations atomic.Int32
}

func (ctx *registrationCountingContext) Value(any) any { return nil }

func (ctx *registrationCountingContext) AfterFunc(callback func()) func() bool {
	ctx.registrations.Add(1)
	var once sync.Once
	release := func() { once.Do(func() { ctx.registrations.Add(-1) }) }
	stop := context.AfterFunc(ctx.Context, func() {
		defer release()
		callback()
	})
	return func() bool {
		if stop() {
			release()
			return true
		}
		return false
	}
}

//go:noinline
func streamStillReachable(ref weak.Pointer[LlmStream]) bool { return ref.Value() != nil }

//go:noinline
func abandonedContextStream(t *testing.T, parent context.Context, source *contractSource) (weak.Pointer[LlmStream], context.Context, []uintptr) {
	t.Helper()
	providerContext := make(chan context.Context, 1)
	stream, err := LlmStreamCallExecuteContext(parent, "abandoned-context-stream", requestForTest(makeRequest()),
		func(ctx context.Context, _ json.RawMessage) (LLMStreamSource, error) {
			providerContext <- ctx
			return source, nil
		}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(); err != nil {
		_ = stream.Close()
		t.Fatal(err)
	}
	return weak.Make(stream), <-providerContext, streamCallbackTokens(stream.sources)
}

func TestContextBoundUnreachableStreamIsReleased(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &registrationCountingContext{Context: parent}
	var reads atomic.Int32
	source := &contractSource{next: func(ctx context.Context) (json.RawMessage, error) {
		if reads.Add(1) == 1 {
			return json.RawMessage(`1`), nil
		}
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	ref, providerContext, ids := abandonedContextStream(t, ctx, source)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if !streamStillReachable(ref) && streamTokensReleased(ids) && providerContext.Err() != nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if streamStillReachable(ref) {
		t.Fatal("live context retained an unreachable stream")
	}
	if source.closed.Load() > 1 {
		t.Fatalf("provider closed %d times", source.closed.Load())
	}
	if !streamTokensReleased(ids) {
		t.Fatal("native provider callbacks were not released")
	}
	if providerContext.Err() != context.Canceled {
		t.Fatalf("provider context was not cancelled: %v", providerContext.Err())
	}
	if ctx.Err() != nil {
		t.Fatalf("parent context was cancelled: %v", ctx.Err())
	}
	if count := ctx.registrations.Load(); count != 0 {
		t.Fatalf("collected stream left %d parent-context registrations", count)
	}
	runtime.KeepAlive(ctx)
}

func TestContextBoundStreamCloseReleasesRegistrations(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &registrationCountingContext{Context: parent}
	stream, err := LlmStreamCallExecuteContext(ctx, "closed-context-stream", requestForTest(makeRequest()),
		func(context.Context, json.RawMessage) (LLMStreamSource, error) { return &contractSource{}, nil }, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if count := ctx.registrations.Load(); count != 0 {
		t.Fatalf("closed stream left %d parent-context registrations", count)
	}
	if ctx.Err() != nil {
		t.Fatalf("parent context was cancelled: %v", ctx.Err())
	}
}
