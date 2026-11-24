// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

type retryFailure struct{ attempt int }

func (failure *retryFailure) Error() string { return "provider retry failed" }

func TestProviderRetriesPreserveSelectedFailure(t *testing.T) {
	first, selected := &retryFailure{attempt: 1}, &retryFailure{attempt: 2}
	if err := RegisterLlmExecutionIntercept("retry-error-identity", 0, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		_, _ = next(request)
		return next(request)
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmExecutionIntercept("retry-error-identity")
	var calls atomic.Int32
	_, err := LlmCallExecuteContext(context.Background(), "retry-error-identity", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (json.RawMessage, error) {
		if calls.Add(1) == 1 {
			return nil, first
		}
		return nil, selected
	})
	var actual *retryFailure
	if !errors.As(err, &actual) || actual != selected || errors.Is(err, first) || !errors.Is(err, ErrInternal) {
		t.Fatalf("selected failure = %#v, error = %v", actual, err)
	}
}

func TestStreamIterationPreservesProviderCause(t *testing.T) {
	for _, intercept := range []bool{false, true} {
		t.Run(fmt.Sprint(intercept), func(t *testing.T) {
			sentinel := errors.New("provider read failure")
			var upstreamInspected atomic.Bool
			if intercept {
				if err := RegisterLlmStreamExecutionIntercept("read-error-identity", 0, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
					upstream, err := next(request)
					if err != nil {
						return nil, err
					}
					return &inspectingStreamSource{LLMStreamSource: upstream, onReadError: func(err error) {
						upstreamInspected.Store(errors.Is(err, sentinel))
					}}, nil
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmStreamExecutionIntercept("read-error-identity")
			}
			var calls atomic.Int32
			stream, err := LlmStreamCallExecuteContext(context.Background(), "read-error-identity", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
				return &contractSource{next: func(context.Context) (json.RawMessage, error) {
					if calls.Add(1) == 1 {
						return json.RawMessage(`1`), nil
					}
					return nil, sentinel
				}}, nil
			}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			if _, err := stream.Next(); err != nil {
				t.Fatal(err)
			}
			_, err = stream.Next()
			var native *RelayError
			if !errors.Is(err, sentinel) || !errors.As(err, &native) || native.Code != StatusUnknown {
				t.Fatalf("read error = %v", err)
			}
			if intercept && !upstreamInspected.Load() {
				t.Fatal("interceptor lost upstream cause")
			}
		})
	}
}

// Keep parent delivery pending until cleanup has run.
type delayedCancellationParent struct {
	context.Context
	done chan struct{}
}

func (parent *delayedCancellationParent) Done() <-chan struct{} { return parent.done }
func (parent *delayedCancellationParent) AfterFunc(func()) func() bool {
	return func() bool { return false }
}

func TestProviderCleanupPreservesPendingParentCancellation(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(fmt.Sprint(deadline), func(t *testing.T) {
			cause := errors.New("parent cancellation cause")
			var carrier context.Context
			var cancel context.CancelFunc
			if deadline {
				carrier, cancel = context.WithDeadlineCause(context.Background(), time.Now().Add(20*time.Millisecond), cause)
			} else {
				var cancelCause context.CancelCauseFunc
				carrier, cancelCause = context.WithCancelCause(context.Background())
				cancel = func() { cancelCause(cause) }
			}
			defer cancel()
			parent := &delayedCancellationParent{Context: carrier, done: make(chan struct{})}
			provider, cleanup := newProviderContext(parent)
			if !deadline {
				cancel()
			}
			select {
			case <-carrier.Done():
			case <-time.After(time.Second):
				t.Fatal("parent did not cancel")
			}
			close(parent.done)
			cleanup()
			if provider.Err() != carrier.Err() || context.Cause(provider) != cause {
				t.Fatalf("provider Err = %v, Cause = %v", provider.Err(), context.Cause(provider))
			}
		})
	}
}

func TestOwnedCodecsRejectInvalidCallbackJSON(t *testing.T) {
	invalid := json.RawMessage("{}\x00trailer")
	codec, err := NewCustomCodec(&CodecFunc{
		Decode: func(_, _ json.RawMessage) (json.RawMessage, error) { return invalid, nil },
		Encode: func(_, _, _ json.RawMessage) (json.RawMessage, error) { return invalid, nil },
	}, func(json.RawMessage) (json.RawMessage, error) { return invalid, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer codec.Close()
	for name, call := range map[string]func() error{
		"decode": func() error { _, err := codec.Decode(LLMRequest{Content: json.RawMessage(`{}`)}); return err },
		"encode": func() error {
			_, err := codec.Encode(json.RawMessage(`{}`), LLMRequest{Content: json.RawMessage(`{}`)})
			return err
		},
		"response": func() error { _, err := codec.DecodeResponse(json.RawMessage(`{}`)); return err },
	} {
		t.Run(name, func(t *testing.T) {
			if err := call(); err == nil || !strings.Contains(err.Error(), "invalid JSON") {
				t.Fatalf("codec result = %v", err)
			}
		})
	}
}

func TestSuppressedProviderErrorDoesNotMatchInterceptorMessage(t *testing.T) {
	provider := errors.New("same failure")
	if err := RegisterLlmExecutionIntercept("suppressed-error", 0, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		_, _ = next(request)
		return nil, errors.New(provider.Error())
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmExecutionIntercept("suppressed-error")
	_, err := LlmCallExecuteContext(context.Background(), "suppressed-error", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (json.RawMessage, error) { return nil, provider })
	if err == nil || errors.Is(err, provider) || !errors.Is(err, ErrInternal) {
		t.Fatalf("interceptor error = %v", err)
	}
}

type inspectingStreamSource struct {
	LLMStreamSource
	onReadError  func(error)
	onCloseError func(error)
}

func (source *inspectingStreamSource) Next(ctx context.Context) (json.RawMessage, error) {
	value, err := source.LLMStreamSource.Next(ctx)
	if err != nil && source.onReadError != nil {
		source.onReadError(err)
	}
	return value, err
}
func (source *inspectingStreamSource) Close() error {
	err := source.LLMStreamSource.Close()
	if err != nil && source.onCloseError != nil {
		source.onCloseError(err)
	}
	return err
}

func TestManagedStreamClosePreservesProviderCause(t *testing.T) {
	for _, intercept := range []bool{false, true} {
		t.Run(fmt.Sprint(intercept), func(t *testing.T) {
			sentinel := errors.New("provider cleanup failure")
			var upstreamInspected atomic.Bool
			if intercept {
				if err := RegisterLlmStreamExecutionIntercept("close-error-identity", 0, func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
					upstream, err := next(request)
					if err != nil {
						return nil, err
					}
					return &inspectingStreamSource{LLMStreamSource: upstream, onCloseError: func(err error) { upstreamInspected.Store(errors.Is(err, sentinel)) }}, nil
				}); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmStreamExecutionIntercept("close-error-identity")
			}
			stream, err := LlmStreamCallExecute("close-error-identity", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) {
				return &contractSource{closeErr: sentinel}, nil
			}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			for range 2 {
				if err := stream.Close(); !errors.Is(err, sentinel) || !errors.Is(err, ErrInternal) {
					t.Fatalf("close error = %v", err)
				}
			}
			if intercept && !upstreamInspected.Load() {
				t.Fatal("interceptor lost upstream cleanup cause")
			}
		})
	}
}

func TestCodecOptionsAllowValidReplacement(t *testing.T) {
	closed := NewOpenAIChatCodec()
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	valid := NewOpenAIChatCodec()
	defer valid.Close()
	request := requestForTest(makeRequest())
	annotation, err := valid.Decode(request)
	if err != nil {
		t.Fatal(err)
	}
	custom := CodecFunc{
		Decode: func(_, _ json.RawMessage) (json.RawMessage, error) { return annotation, nil },
		Encode: func(_, _, content json.RawMessage) (json.RawMessage, error) { return content, nil },
	}
	for name, options := range map[string][]LLMCallOption{
		"request reset":     {WithLLMCodec(closed), WithLLMCodec(nil)},
		"request handle":    {WithLLMCodec(closed), WithLLMCodec(valid)},
		"request callbacks": {WithLLMCodec(closed), WithLLMCodec(custom)},
		"response handle":   {WithLLMResponseCodec(closed), WithLLMResponseCodec(valid)},
		"response reset":    {WithLLMResponseCodec(closed), WithLLMResponseCodec(nil)},
	} {
		t.Run(name, func(t *testing.T) {
			handle, err := LlmCall("replacement-codec", request, options...)
			if err != nil {
				t.Fatal(err)
			}
			if err := LlmCallEnd(handle, json.RawMessage(`{}`)); err != nil {
				t.Fatal(err)
			}
		})
	}
	for name, options := range map[string][]LLMCallOption{
		"request error":  {WithLLMCodec(closed), WithLLMResponseCodec(closed), WithLLMResponseCodec(valid)},
		"response error": {WithLLMResponseCodec(closed), WithLLMCodec(closed), WithLLMCodec(valid)},
	} {
		t.Run(name, func(t *testing.T) {
			if handle, err := LlmCall("invalid-codec", request, options...); err == nil {
				_ = LlmCallEnd(handle, json.RawMessage(`{}`))
				t.Fatal("replacement cleared another codec's error")
			}
		})
	}
}

type signaledCloseSource struct {
	contractSource
	closeDone chan struct{}
}

func (source *signaledCloseSource) Close() error {
	err := source.contractSource.Close()
	close(source.closeDone)
	return err
}

func TestStreamClosePreservesAllSelectedCauses(t *testing.T) {
	readErr := errors.New("undelivered provider read")
	cleanupErr := errors.New("provider cleanup")
	finalizerErr := errors.New("finalizer failure")
	var calls atomic.Int32
	source := &signaledCloseSource{closeDone: make(chan struct{}), contractSource: contractSource{
		next: func(context.Context) (json.RawMessage, error) {
			if calls.Add(1) == 1 {
				return json.RawMessage(`1`), nil
			}
			return nil, readErr
		}, closeErr: cleanupErr,
	}}
	stream, err := LlmStreamCallExecute("combined-stream-errors", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) { return source, nil }, nil, func() (json.RawMessage, error) { return nil, finalizerErr })
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Next(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-source.closeDone:
	case <-time.After(5 * time.Second):
		t.Fatal("provider did not finish before close")
	}
	for range 2 {
		err := stream.Close()
		for _, cause := range []error{readErr, cleanupErr, finalizerErr, ErrInternal} {
			if !errors.Is(err, cause) {
				t.Errorf("close lost %v: %v", cause, err)
			}
		}
	}
}

func TestStreamClosePreservesFinalizerCause(t *testing.T) {
	cause := errors.New("standalone finalizer failure")
	stream, err := LlmStreamCallExecute("finalizer-cause", requestForTest(makeRequest()), func(context.Context, json.RawMessage) (LLMStreamSource, error) { return &contractSource{}, nil }, nil, func() (json.RawMessage, error) { return nil, cause })
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if err := stream.Close(); !errors.Is(err, cause) || !errors.Is(err, ErrInternal) {
		t.Fatalf("close error = %v", err)
	}
}
