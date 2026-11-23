// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>
typedef struct FfiLlmStreamSource FfiLlmStreamSource;
typedef int32_t (*NemoRelayStreamReadFn)(void*, char**);
typedef void (*NemoRelayStreamCancelFn)(void*);
typedef int32_t (*NemoRelayStreamCloseFn)(void*);
typedef void (*NemoRelayFreeFn)(void*);
extern FfiLlmStreamSource* nemo_relay_stream_source_create(NemoRelayStreamReadFn, NemoRelayStreamCancelFn, NemoRelayStreamCloseFn, void*, NemoRelayFreeFn);
extern int32_t nemo_relay_stream_source_next(const FfiLlmStreamSource*, char**);
extern void nemo_relay_stream_source_cancel(const FfiLlmStreamSource*);
extern int32_t nemo_relay_stream_source_close(const FfiLlmStreamSource*);
extern void nemo_relay_stream_source_free(FfiLlmStreamSource*);
extern void nemo_relay_string_free(char*);
extern int32_t goSourceReadTrampoline(void*, char**);
extern void goSourceCancelTrampoline(void*);
extern int32_t goSourceCloseTrampoline(void*);
extern void goSourceFreeTrampoline(void*);
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"runtime"
	"sync"
	"unsafe"
	"weak"
)

// LLMStreamSource is an owned, pull-based provider stream. Next must honor ctx
// cancellation. Close releases provider resources after reads stop and runs once.
// Sources must allow cancellation from a different goroutine. Explicitly close
// the managed stream to ensure provider cleanup runs. Garbage collection cancels
// the provider context but can skip source callbacks that are no longer live.
// Owned callbacks may use their supplied capabilities and emit observability
// events. Other Relay API calls, including managed-stream control and flushes,
// are invalid from these callbacks.
type LLMStreamSource interface {
	Next(context.Context) (json.RawMessage, error)
	Close() error
}

// LLMStreamExecutionFunc opens a provider stream for the request.
type LLMStreamExecutionFunc func(context.Context, json.RawMessage) (LLMStreamSource, error)

// LLMStreamExecutionInterceptFunc wraps the complete upstream source. The next
// continuation keeps its scope and trace context. The source can wait until a
// read to call it. It stays valid until the returned source closes. Callback
// codec context expires on return. Retain the optimization recorder to record
// evidence later.
type LLMStreamExecutionInterceptFunc func(json.RawMessage, LLMExecutionContext, func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error)

var streamCallbackOwners sync.Map // native callback token -> weak.Pointer[streamSources]

// streamSources owns Go callback state while a managed stream is live. Native
// callback tokens contain only weak references, so sources can capture the stream
// without creating a global root. Active reads retain only their source state;
// they must not retain this owner and its provider factory while blocked.
type streamSources struct {
	providerErrors providerErrorCapture
	mu             sync.Mutex
	states         map[*foreignSourceState]struct{}
	factory        *streamFactory
}

func (sources *streamSources) add(state *foreignSourceState) {
	sources.mu.Lock()
	defer sources.mu.Unlock()
	if sources.states == nil {
		sources.states = make(map[*foreignSourceState]struct{})
	}
	sources.states[state] = struct{}{}
}
func (sources *streamSources) remove(state *foreignSourceState) {
	sources.mu.Lock()
	defer sources.mu.Unlock()
	delete(sources.states, state)
}

type streamFactory struct {
	sources *streamSources
	ctx     context.Context
	fn      LLMStreamExecutionFunc
}
type foreignSourceState struct {
	owner  weak.Pointer[streamSources]
	readMu sync.Mutex
	source LLMStreamSource
	ctx    context.Context
	cancel context.CancelFunc
	once   sync.Once
	err    error
}

func foreignStreamSource(source LLMStreamSource, ctx context.Context, owner *streamSources) (*C.FfiLlmStreamSource, error) {
	if source == nil {
		return nil, errors.New("stream source is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	readCtx, cancel := context.WithCancel(ctx)
	state := &foreignSourceState{source: source, ctx: readCtx, cancel: cancel, owner: weak.Make(owner)}
	var token unsafe.Pointer
	if owner == nil {
		// A Go interceptor can also serve a stream called through the raw C ABI.
		// With no Go handle, native source ownership must keep callback state alive.
		token = registerClosure(state)
	} else {
		owner.add(state)
		token = registerClosure(weak.Make(state))
	}
	ptr, err := callPointer(func() *C.FfiLlmStreamSource {
		return C.nemo_relay_stream_source_create(C.NemoRelayStreamReadFn(C.goSourceReadTrampoline), C.NemoRelayStreamCancelFn(C.goSourceCancelTrampoline), C.NemoRelayStreamCloseFn(C.goSourceCloseTrampoline), token, C.NemoRelayFreeFn(C.goSourceFreeTrampoline))
	})
	if err != nil {
		if owner != nil {
			owner.remove(state)
		}
		cancel()
	}
	return ptr, err
}

// watchCancellation waits for an already-running cancellation callback during
// cleanup, keeping its borrowed native pointer alive until it has finished.
func watchCancellation(ctx context.Context, cancel func()) func() {
	done := make(chan struct{})
	stop := context.AfterFunc(ctx, func() { defer close(done); cancel() })
	return func() {
		if !stop() {
			<-done
		}
	}
}

type nativeStreamSource struct {
	cleanup runtime.Cleanup
	mu      sync.RWMutex
	ptr     *C.FfiLlmStreamSource
	once    sync.Once
	err     error
}

func ownedStreamSource(ptr *C.FfiLlmStreamSource) *nativeStreamSource {
	source := &nativeStreamSource{ptr: ptr}
	source.cleanup = runtime.AddCleanup(source, releaseNativeStreamSource, ptr)
	return source
}
func releaseNativeStreamSource(ptr *C.FfiLlmStreamSource) { C.nemo_relay_stream_source_free(ptr) }

func (source *nativeStreamSource) Next(ctx context.Context) (json.RawMessage, error) {
	if ctx == nil {
		return nil, errors.New("context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	source.mu.RLock()
	defer source.mu.RUnlock()
	if source.ptr == nil {
		return nil, io.EOF
	}
	stop := watchCancellation(ctx, func() { C.nemo_relay_stream_source_cancel(source.ptr) })
	defer stop()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var output *C.char
	status := C.nemo_relay_stream_source_next(source.ptr, &output)
	if output != nil {
		defer C.nemo_relay_string_free(output)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch status {
	case 0:
		return nil, io.EOF
	case 1:
		return json.RawMessage(C.GoString(output)), nil
	default:
		return nil, lastError()
	}
}
func (source *nativeStreamSource) Close() error {
	source.once.Do(func() {
		source.mu.RLock()
		if source.ptr != nil {
			C.nemo_relay_stream_source_cancel(source.ptr)
		}
		source.mu.RUnlock()
		source.mu.Lock()
		defer source.mu.Unlock()
		if source.ptr != nil {
			source.err = callStatus(func() C.int32_t { return C.nemo_relay_stream_source_close(source.ptr) })
			source.cleanup.Stop()
			C.nemo_relay_stream_source_free(source.ptr)
			source.ptr = nil
		}
		runtime.KeepAlive(source)
	})
	return source.err
}

type prefixStreamSource struct {
	first json.RawMessage
	tail  LLMStreamSource
}

func (s *prefixStreamSource) Next(ctx context.Context) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.first != nil {
		v := s.first
		s.first = nil
		return v, nil
	}
	if s.tail == nil {
		return nil, io.EOF
	}
	return s.tail.Next(ctx)
}
func (s *prefixStreamSource) Close() error {
	if s.tail != nil {
		return s.tail.Close()
	}
	return nil
}

// AsStream adapts a single-response execution callback to a one-chunk stream.
// Use LLMStreamExecutionFunc directly for a provider that produces multiple chunks.
func (fn LLMExecutionFunc) AsStream() LLMStreamExecutionFunc {
	if fn == nil {
		return nil
	}
	return func(_ context.Context, request json.RawMessage) (LLMStreamSource, error) {
		value, err := fn(request)
		if err != nil {
			return nil, err
		}
		return &prefixStreamSource{first: value}, nil
	}
}

// AsStream adapts a single-response interceptor to streaming execution. It
// applies the callback to the first chunk and preserves the remaining stream.
// Calling next closes any previous source. If cleanup fails, next returns that
// error. If the interceptor also fails, the returned error includes both failures.
func (fn LLMExecutionInterceptFunc) AsStream() LLMStreamExecutionInterceptFunc {
	if fn == nil {
		return nil
	}
	return func(request json.RawMessage, context LLMExecutionContext, next func(json.RawMessage) (LLMStreamSource, error)) (LLMStreamSource, error) {
		var tail LLMStreamSource
		var cleanupErr error
		readContext := context.callContext
		if readContext == nil {
			readContext = backgroundContext()
		}
		result, err := fn(request, context, func(request json.RawMessage) (json.RawMessage, error) {
			if tail != nil {
				previous := tail
				tail = nil
				if err := previous.Close(); err != nil {
					cleanupErr = errors.Join(cleanupErr, err)
					return nil, err
				}
			}
			stream, err := next(request)
			if err != nil {
				return nil, err
			}
			tail = stream
			value, err := stream.Next(readContext)
			if errors.Is(err, io.EOF) {
				if err := readContext.Err(); err != nil {
					return nil, err
				}
				return json.RawMessage(`null`), nil
			}
			return value, err
		})
		if err != nil {
			err = errors.Join(err, cleanupErr)
			if tail != nil {
				err = errors.Join(err, tail.Close())
			}
			return nil, err
		}
		return &prefixStreamSource{first: result, tail: tail}, nil
	}
}
func backgroundContext() context.Context { return context.Background() }

func lookupSourceState(data unsafe.Pointer) *foreignSourceState {
	switch state := lookupClosure(data).(type) {
	case weak.Pointer[foreignSourceState]:
		return state.Value()
	case *foreignSourceState:
		return state
	default:
		return nil
	}
}

//export goSourceReadTrampoline
func goSourceReadTrampoline(data unsafe.Pointer, output **C.char) (status C.int32_t) {
	defer recoverCallbackPanic("stream source read", func() { status = -1 })
	state := lookupSourceState(data)
	if state == nil {
		return 0
	}
	defer runtime.KeepAlive(state)
	state.readMu.Lock()
	defer state.readMu.Unlock()
	value, err := state.source.Next(state.ctx)
	if errors.Is(err, io.EOF) {
		return 0
	}
	if err != nil {
		if owner := state.owner.Value(); owner != nil {
			owner.providerErrors.record(err)
		}
		setLastErrorMessage(err.Error())
		return -1
	}
	if !json.Valid(value) {
		setLastErrorMessage("stream source returned invalid JSON")
		return -1
	}
	*output = relayCallbackString(string(value))
	return 1
}

//export goSourceCancelTrampoline
func goSourceCancelTrampoline(data unsafe.Pointer) {
	if state := lookupSourceState(data); state != nil {
		state.cancel()
	}
}

//export goSourceCloseTrampoline
func goSourceCloseTrampoline(data unsafe.Pointer) (status C.int32_t) {
	defer recoverCallbackPanic("stream source close", func() { status = 5 })
	state := lookupSourceState(data)
	if state == nil {
		return 0
	}
	defer runtime.KeepAlive(state)
	state.once.Do(func() {
		state.readMu.Lock()
		defer state.readMu.Unlock()
		state.cancel()
		state.err = state.source.Close()
	})
	if state.err != nil {
		setLastErrorMessage(state.err.Error())
		return 5
	}
	return 0
}

//export goSourceFreeTrampoline
func goSourceFreeTrampoline(data unsafe.Pointer) {
	if state := lookupSourceState(data); state != nil {
		if owner := state.owner.Value(); owner != nil {
			owner.remove(state)
		}
	}
	unregisterClosure(data)
}
