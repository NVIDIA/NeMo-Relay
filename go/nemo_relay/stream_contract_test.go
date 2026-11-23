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

type contractSource struct {
	chunks   []json.RawMessage
	index    int
	closed   atomic.Int32
	next     func(context.Context) (json.RawMessage, error)
	closeErr error
}

func (s *contractSource) Next(ctx context.Context) (json.RawMessage, error) {
	if s.next != nil {
		return s.next(ctx)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.index == len(s.chunks) {
		return nil, io.EOF
	}
	v := s.chunks[s.index]
	s.index++
	return v, nil
}
func (s *contractSource) Close() error { s.closed.Add(1); return s.closeErr }

type contractMappedSource struct {
	source   LLMStreamSource
	open     func(json.RawMessage) (LLMStreamSource, error)
	request  json.RawMessage
	recorder *LLMOptimizationRecorder
}

func (s *contractMappedSource) Next(ctx context.Context) (json.RawMessage, error) {
	if s.source == nil {
		var err error
		s.source, err = s.open(s.request)
		if err != nil {
			return nil, err
		}
	}
	value, err := s.source.Next(ctx)
	if err != nil {
		return nil, err
	}
	accepted, err := s.recorder.RecordOptimizationContribution(LLMOptimizationContribution{Producer: "deferred-stream", Kind: "custom", Applied: true})
	if err != nil {
		return nil, err
	}
	if !accepted {
		return nil, errors.New("deferred contribution rejected")
	}
	return json.RawMessage(`{"wrapped":` + string(value) + `}`), nil
}
func (s *contractMappedSource) Close() error {
	if s.source != nil {
		return s.source.Close()
	}
	return nil
}

func TestTrueStreamingAndDeferredOptimizationRecorder(t *testing.T) {
	runWithTestScopeStack(t, func() {
		var recorder *LLMOptimizationRecorder
		var mu sync.Mutex
		var end json.RawMessage
		if err := RegisterSubscriber("true-stream-events", func(event Event) {
			if event.Name() == "true-stream" && event.ScopeCategory() == "end" {
				mu.Lock()
				end = append(json.RawMessage(nil), event.AnnotatedResponse()...)
				mu.Unlock()
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("true-stream-events")
		intercept := LLMStreamExecutionInterceptFunc(func(request json.RawMessage, ctx LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
			var err error
			recorder, err = ctx.RetainOptimizationRecorder()
			if err != nil {
				return nil, err
			}
			return &contractMappedSource{open: next, request: request, recorder: recorder}, nil
		})
		if err := RegisterLlmStreamExecutionIntercept("true-stream-intercept", 0, intercept); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmStreamExecutionIntercept("true-stream-intercept")
		var opened atomic.Int32
		source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`), json.RawMessage(`2`), json.RawMessage(`3`)}}
		stream, err := LlmStreamCallExecute("true-stream", requestForTest(makeRequest()), LLMStreamExecutionFunc(func(context.Context, json.RawMessage) (LLMStreamSource, error) { opened.Add(1); return source, nil }), nil, func() (json.RawMessage, error) { return json.RawMessage(`{"model":"stream-model","choices":[]}`), nil }, WithLLMResponseCodec(NewOpenAIChatCodec()))
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		if opened.Load() != 0 {
			t.Fatal("provider opened before the first read")
		}
		if recorder == nil {
			t.Fatal("missing retained recorder")
		}
		defer recorder.Close()
		for _, want := range []string{`{"wrapped":1}`, `{"wrapped":2}`, `{"wrapped":3}`} {
			got, err := stream.Next()
			if err != nil {
				t.Fatal(err)
			}
			assertSemanticJSONEqual(t, []byte(want), got)
		}
		if _, err := stream.Next(); !errors.Is(err, io.EOF) {
			t.Fatalf("expected EOF, got %v", err)
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		if source.closed.Load() != 1 {
			t.Fatalf("source closed %d times", source.closed.Load())
		}
		if accepted, err := recorder.RecordOptimizationContribution(LLMOptimizationContribution{Producer: "late", Kind: "custom"}); err != nil || accepted {
			t.Fatalf("late record: %v %v", accepted, err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		if !strings.Contains(string(end), "deferred-stream") {
			t.Fatalf("missing deferred evidence: %s", end)
		}
	})
}

func TestStreamContextCancellationUnblocksReadAndClosesOnce(t *testing.T) {
	for _, callContext := range []bool{false, true} {
		t.Run(map[bool]string{false: "next-context", true: "call-context"}[callContext], func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			entered := make(chan struct{})
			var once sync.Once
			source := &contractSource{next: func(ctx context.Context) (json.RawMessage, error) {
				once.Do(func() { close(entered) })
				<-ctx.Done()
				return nil, ctx.Err()
			}}
			provider := LLMStreamExecutionFunc(func(context.Context, json.RawMessage) (LLMStreamSource, error) { return source, nil })
			var stream *LlmStream
			var err error
			if callContext {
				stream, err = LlmStreamCallExecuteContext(ctx, "cancel-stream", requestForTest(makeRequest()), provider, nil, nil)
			} else {
				stream, err = LlmStreamCallExecute("cancel-stream", requestForTest(makeRequest()), provider, nil, nil)
			}
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			result := make(chan error, 1)
			go func() {
				if callContext {
					_, err := stream.Next()
					result <- err
				} else {
					_, err := stream.NextContext(ctx)
					result <- err
				}
			}()
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("read did not start")
			}
			cancel()
			select {
			case err := <-result:
				if !errors.Is(err, context.Canceled) && !(callContext && errors.Is(err, io.EOF)) {
					t.Fatalf("unexpected cancellation: %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("cancelled read blocked")
			}
			_ = stream.Close()
			if source.closed.Load() != 1 {
				t.Fatalf("source closed %d times", source.closed.Load())
			}
		})
	}
}

func TestManagedContextCancellation(t *testing.T) {
	for _, tool := range []bool{false, true} {
		t.Run(map[bool]string{false: "llm", true: "tool"}[tool], func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			var err error
			if tool {
				_, err = ToolCallExecuteContext(ctx, "cancel-tool", json.RawMessage(`{}`), func(ctx context.Context, _ json.RawMessage) (ToolExecutionResult, error) {
					<-ctx.Done()
					return ToolExecutionResult{}, ctx.Err()
				})
			} else {
				_, err = LlmCallExecuteContext(ctx, "cancel-llm", requestForTest(makeRequest()), func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
					<-ctx.Done()
					return nil, ctx.Err()
				})
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("expected deadline, got %v", err)
			}
		})
	}
}

func TestStreamSourceErrorsAndCloseFailures(t *testing.T) {
	source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`), json.RawMessage(`2`)}, closeErr: errors.New("provider close failed")}
	stream, err := LlmStreamCallExecute("close-error", requestForTest(makeRequest()), LLMStreamExecutionFunc(func(context.Context, json.RawMessage) (LLMStreamSource, error) { return source, nil }), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for range 2 {
		if _, err := stream.Next(); err != nil {
			t.Fatal(err)
		}
	}
	for {
		_, err := stream.Next()
		if err != nil {
			break
		}
	}
	if err := stream.Close(); err == nil || !strings.Contains(err.Error(), "provider close failed") {
		t.Fatalf("missing close error: %v", err)
	}
	if source.closed.Load() != 1 {
		t.Fatalf("source closed %d times", source.closed.Load())
	}
}

func TestCustomCodecCloseDuringReadKeepsCallbacksAlive(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseDecode := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseDecode()
	codec, err := NewCustomCodec(&CodecFunc{
		Decode: func(_, _ json.RawMessage) (json.RawMessage, error) {
			close(entered)
			<-release
			return json.RawMessage(`{"model":"custom"}`), nil
		},
		Encode: func(_, _ json.RawMessage, content json.RawMessage) (json.RawMessage, error) { return content, nil },
	}, func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"model":"response"}`), nil })
	if err != nil {
		t.Fatal(err)
	}
	defer func() { releaseDecode(); _ = codec.Close() }()
	done := make(chan error, 1)
	go func() { _, err := codec.Decode(LLMRequest{Content: json.RawMessage(`{}`)}); done <- err }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("decode callback did not start")
	}
	closed := make(chan error, 1)
	go func() { closed <- codec.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("codec Close blocked on active decode")
	}
	releaseDecode()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("decode did not finish")
	}
	if _, err := codec.Decode(LLMRequest{}); err == nil {
		t.Fatal("closed codec remained callable")
	}
	if _, err := LlmCall("closed-codec", requestForTest(makeRequest()), WithLLMCodec(codec)); err == nil {
		t.Fatal("closed codec accepted")
	}
	valid := NewOpenAIChatCodec()
	defer valid.Close()
	if _, err := LlmCall("closed-codec-with-valid-response", requestForTest(makeRequest()), WithLLMCodec(codec), WithLLMResponseCodec(valid)); err == nil {
		t.Fatal("valid response codec hid the closed request codec error")
	}

}

type lazyOpenSource struct {
	request json.RawMessage
	open    func(json.RawMessage) (LLMStreamSource, error)
	source  LLMStreamSource
}

func (s *lazyOpenSource) Next(ctx context.Context) (json.RawMessage, error) {
	if s.source == nil {
		var err error
		s.source, err = s.open(s.request)
		if err != nil {
			return nil, err
		}
	}
	return s.source.Next(ctx)
}
func (s *lazyOpenSource) Close() error {
	if s.source != nil {
		return s.source.Close()
	}
	return nil
}

func TestClosingStreamCancelsLazyProviderOpen(t *testing.T) {
	entered := make(chan struct{})
	if err := RegisterLlmStreamExecutionIntercept("lazy-cancel-open", 0, LLMStreamExecutionInterceptFunc(func(request json.RawMessage, _ LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
		return &lazyOpenSource{request: request, open: next}, nil
	})); err != nil {
		t.Fatal(err)
	}
	defer DeregisterLlmStreamExecutionIntercept("lazy-cancel-open")
	stream, err := LlmStreamCallExecute("lazy-cancel-open", requestForTest(makeRequest()), LLMStreamExecutionFunc(func(ctx context.Context, _ json.RawMessage) (LLMStreamSource, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	read := make(chan error, 1)
	go func() { _, err := stream.Next(); read <- err }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("provider did not open lazily")
	}
	closed := make(chan error, 1)
	go func() { closed <- stream.Close() }()
	select {
	case <-read:
	case <-time.After(3 * time.Second):
		t.Fatal("lazy provider open did not cancel")
	}
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not finish")
	}
}
