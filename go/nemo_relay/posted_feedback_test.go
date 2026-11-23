// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStreamSetupDeadlineReachesInterceptorSources(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseRead := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseRead()
	source := &contractSource{next: func(ctx context.Context) (json.RawMessage, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-release:
			return nil, errors.New("test released blocked source")
		}
	}}
	outer := LLMExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		return next(request)
	}).AsStream()
	if err := RegisterLlmStreamExecutionIntercept("deadline-outer", 0, outer); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmStreamExecutionIntercept("deadline-outer")
	if err := RegisterLlmStreamExecutionIntercept("deadline-inner", 1, func(json.RawMessage, LLMExecutionContext, func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
		return source, nil
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmStreamExecutionIntercept("deadline-inner")
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		stream, err := LlmStreamCallExecuteContext(ctx, "setup-deadline", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
			return nil, errors.New("replacement source should skip provider")
		}, nil, nil)
		if stream != nil {
			_ = stream.Close()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("setup error = %v", err)
		}
	case <-time.After(2 * time.Second):
		releaseRead()
		select {
		case <-done:
		case <-time.After(time.Second):
		}
		t.Fatal("stream setup did not honor deadline")
	}
	if got := source.closed.Load(); got != 1 {
		t.Fatalf("source closed %d times", got)
	}
}

func TestManagedCallsPreserveOutcomeWhenProviderCancels(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "provider error"}[fail], func(t *testing.T) {
			for _, tool := range []bool{false, true} {
				ctx, cancel := context.WithCancel(context.Background())
				var calls atomic.Int32
				providerErr := errors.New("selected provider failure")
				var err error
				if tool {
					var result ToolExecutionResult
					result, err = ToolCallExecuteContext(ctx, "completed-tool", json.RawMessage(`{}`), func(context.Context, json.RawMessage) (ToolExecutionResult, error) {
						calls.Add(1)
						cancel()
						if fail {
							return ToolExecutionResult{}, providerErr
						}
						return ToolExecutionResult{Result: json.RawMessage(`42`)}, nil
					})
					if !fail && string(result.Result) != `42` {
						t.Errorf("tool result = %s", result.Result)
					}
				} else {
					var result json.RawMessage
					result, err = LlmCallExecuteContext(ctx, "completed-llm", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (json.RawMessage, error) {
						calls.Add(1)
						cancel()
						if fail {
							return nil, providerErr
						}
						return json.RawMessage(`42`), nil
					})
					if !fail && string(result) != `42` {
						t.Errorf("LLM result = %s", result)
					}
				}
				cancel()
				if calls.Load() != 1 {
					t.Fatalf("provider calls = %d", calls.Load())
				}
				if fail {
					if err == nil || !strings.Contains(err.Error(), providerErr.Error()) || !errors.Is(err, providerErr) || errors.Is(err, context.Canceled) {
						t.Errorf("provider failure replaced: %v", err)
					}
				} else if err != nil {
					t.Errorf("success replaced: %v", err)
				}
			}
		})
	}
}

func TestProviderContextPreservesCancellationCauses(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		cause := errors.New("upstream quota")
		var parent context.Context
		var cancel context.CancelFunc
		if deadline {
			parent, cancel = context.WithDeadlineCause(context.Background(), time.Now().Add(20*time.Millisecond), cause)
		} else {
			var cancelCause context.CancelCauseFunc
			parent, cancelCause = context.WithCancelCause(context.Background())
			cancel = func() { cancelCause(cause) }
		}
		provider, stop := newProviderContext(parent)
		child, stopChild := context.WithCancel(provider)
		if !deadline {
			cancel()
		}
		select {
		case <-provider.Done():
		case <-time.After(time.Second):
			t.Fatal("provider not cancelled")
		}
		select {
		case <-child.Done():
		case <-time.After(time.Second):
			t.Fatal("child not cancelled")
		}
		wantErr := context.Canceled
		if deadline {
			wantErr = context.DeadlineExceeded
		}
		if provider.Err() != wantErr || child.Err() != wantErr {
			t.Errorf("Err = %v / %v", provider.Err(), child.Err())
		}
		if context.Cause(provider) != cause || context.Cause(child) != cause {
			t.Errorf("cause = %v / %v", context.Cause(provider), context.Cause(child))
		}
		stop()
		stopChild()
		cancel()
		if context.Cause(provider) != cause {
			t.Fatal("Close replaced cancellation cause")
		}
	}
}

func TestStreamAdapterPreservesEarlierCleanupError(t *testing.T) {
	cleanup := errors.New("first cleanup failure")
	primary := errors.New("interceptor failure")
	source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`)}, closeErr: cleanup}
	adapter := LLMExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		if _, err := next(request); err != nil {
			return nil, err
		}
		if _, err := next(request); !errors.Is(err, cleanup) {
			t.Errorf("next cleanup error = %v", err)
		}
		return nil, primary
	}).AsStream()
	stream, err := adapter(json.RawMessage(`{}`), LLMExecutionContext{}, func(json.RawMessage) (LLMStreamSource, error) { return source, nil })
	if stream != nil || !errors.Is(err, primary) || !errors.Is(err, cleanup) {
		t.Fatalf("joined error = %v", err)
	}
	if source.closed.Load() != 1 {
		t.Fatalf("source closed %d times", source.closed.Load())
	}
}

func TestStreamSetupPreservesSelectedOutcome(t *testing.T) {
	for _, fail := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		providerErr := errors.New("selected stream factory failure")
		source := &contractSource{}
		stream, err := LlmStreamCallExecuteContext(ctx, "stream-setup-outcome", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
			cancel()
			if fail {
				return nil, providerErr
			}
			return source, nil
		}, nil, nil)
		cancel()
		if fail {
			if stream != nil {
				_ = stream.Close()
				t.Error("failed factory returned a stream")
			}
			if !errors.Is(err, providerErr) || errors.Is(err, context.Canceled) {
				t.Errorf("factory failure replaced: %v", err)
			}
		} else {
			if err != nil || stream == nil {
				t.Fatalf("successful factory replaced: stream=%v err=%v", stream, err)
			}
			if err := stream.Close(); err != nil {
				t.Fatal(err)
			}
			if source.closed.Load() != 1 {
				t.Fatalf("source closed %d times", source.closed.Load())
			}
		}
	}
}
