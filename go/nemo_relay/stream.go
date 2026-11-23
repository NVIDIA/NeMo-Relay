// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiStream FfiStream;

extern int32_t nemo_relay_stream_next(FfiStream* stream, char** out_chunk);
extern void nemo_relay_stream_cancel(FfiStream* stream);
extern int32_t nemo_relay_stream_close(FfiStream* stream);
extern void nemo_relay_stream_free(FfiStream* stream);
extern void nemo_relay_string_free(char* ptr);
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"runtime"
	"sync"
	"weak"
)

// LlmStream wraps a streaming LLM response returned by [LlmStreamCallExecute].
// Call Next to read JSON chunks from the provider source.
//
// Usage pattern:
//
//	stream, err := nemo_relay.LlmStreamCallExecute("chat", req, myExecFn, collector, finalizer)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer stream.Close()
//
//	for {
//	    chunk, err := stream.Next()
//	    if err == io.EOF {
//	        break
//	    }
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    fmt.Print(chunk)
//	}
//
// Calls to Next and Close are synchronized. If not closed explicitly, the
// native stream is released after the handle becomes unreachable. The collector
// and finalizer do not run after that. Garbage collection can skip provider
// cleanup callbacks. Call Close to ensure provider Close callbacks run.
//
// Each stream carries its own collector and finalizer, so multiple streams can
// operate concurrently without interfering with one another.
type LlmStream struct {
	sources        *streamSources
	cancelProvider context.CancelFunc
	callContext    context.Context
	stopContext    func() bool
	mu             sync.Mutex
	ptr            *C.FfiStream
	callbacks      *streamCallbacks
	cleanup        runtime.Cleanup
	closed         bool
	closing        bool
	inFlight       int
	idle           chan struct{}
	closeDone      chan struct{}
	closeErr       error
}

type streamCallbacks struct {
	providerErrors *providerErrorCapture
	collector      CollectorFunc
	finalizer      FinalizerFunc
}

func newLlmStream(ptr *C.FfiStream, callbacks *streamCallbacks, cancelProvider context.CancelFunc, sources *streamSources) *LlmStream {
	if ptr == nil {
		return nil
	}
	s := &LlmStream{ptr: ptr, callbacks: callbacks, cancelProvider: cancelProvider, sources: sources}
	s.cleanup = runtime.AddCleanup(s, releaseStream, streamCleanup{ptr: ptr, cancelProvider: cancelProvider})
	return s
}

type streamCleanup struct {
	ptr            *C.FfiStream
	cancelProvider context.CancelFunc
	stopContext    func() bool
}

func releaseStream(cleanup streamCleanup) {
	if cleanup.stopContext != nil {
		cleanup.stopContext()
	}
	if cleanup.cancelProvider != nil {
		cleanup.cancelProvider()
	}
	C.nemo_relay_stream_free(cleanup.ptr)
}

// Next returns the next chunk from the stream as a JSON value. It returns
// [io.EOF] when the stream is exhausted and all chunks have been consumed.
// Stream execution intercepts can wrap or replace the provider source before
// chunks reach the caller.
//
// An explicitly closed stream returns io.EOF. A stream bound to a cancelled
// context returns that context's error.
func (s *LlmStream) Next() (json.RawMessage, error) {
	s.mu.Lock()
	ctx := s.callContext
	if ctx != nil && ctx.Err() != nil {
		s.mu.Unlock()
		return nil, ctx.Err()
	}
	if s.closed || s.closing || s.ptr == nil {
		s.mu.Unlock()
		return nil, io.EOF
	}
	ptr := s.ptr
	sources := s.sources
	s.inFlight++
	if s.inFlight == 1 {
		s.idle = make(chan struct{})
	}
	s.mu.Unlock()
	defer s.finishNext()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var chunk *C.char
	rc := C.nemo_relay_stream_next(ptr, &chunk)
	if ctx != nil && ctx.Err() != nil {
		if chunk != nil {
			C.nemo_relay_string_free(chunk)
		}
		return nil, ctx.Err()
	}
	switch rc {
	case 1:
		text := C.GoString(chunk)
		C.nemo_relay_string_free(chunk)
		return json.RawMessage(text), nil
	case 0:
		return nil, io.EOF
	default:
		err := lastError()
		if sources != nil {
			err = sources.providerErrors.restore(err)
		}
		return nil, err
	}
}

func (s *LlmStream) finishNext() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFlight--
	if s.inFlight == 0 {
		close(s.idle)
	}
}

// Close stops the producer, waits for cleanup, and releases the underlying C
// stream resources. It is safe to call Close more than once. Later calls
// return the first result. Call Close outside owned callbacks. Managed-stream
// control from a callback is invalid. After Close, [LlmStream.Next] returns
// [io.EOF], or the context error if the stream's context was cancelled.
// The finalizer runs once with any collected partial response.
// Close returns any finalizer error.
func (s *LlmStream) Close() error {
	s.mu.Lock()
	if s.closed {
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	if s.closing {
		done := s.closeDone
		s.mu.Unlock()
		<-done
		s.mu.Lock()
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	s.closing = true
	s.closeDone = make(chan struct{})
	ptr := s.ptr
	cancelProvider := s.cancelProvider
	s.mu.Unlock()
	if cancelProvider != nil {
		cancelProvider()
	}
	C.nemo_relay_stream_cancel(ptr)
	s.closeNative(ptr)
	s.mu.Lock()
	err := s.closeErr
	s.mu.Unlock()
	return err
}

func (s *LlmStream) closeNative(ptr *C.FfiStream) {
	s.mu.Lock()
	for s.inFlight > 0 {
		idle := s.idle
		s.mu.Unlock()
		<-idle
		s.mu.Lock()
	}
	s.mu.Unlock()
	err := callStatus(func() C.int32_t { return C.nemo_relay_stream_close(ptr) })
	s.mu.Lock()
	if s.sources != nil {
		err = s.sources.providerErrors.restore(err)
	}
	stopContext := s.stopContext
	s.stopContext = nil
	s.closeErr = err
	s.ptr = nil
	s.sources = nil
	s.closed = true
	close(s.closeDone)
	s.mu.Unlock()
	if stopContext != nil {
		stopContext()
	}
	s.cleanup.Stop()
	C.nemo_relay_stream_free(ptr)
	runtime.KeepAlive(s)
}

func (s *LlmStream) bindContext(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.callContext = ctx
	if !s.closed {
		ref := weak.Make(s)
		s.stopContext = context.AfterFunc(ctx, func() {
			if stream := ref.Value(); stream != nil {
				_ = stream.Close()
			}
		})
		// Include the newly registered watcher in GC cleanup as well. Its stop
		// function retains only the weak stream reference in the callback.
		s.cleanup.Stop()
		s.cleanup = runtime.AddCleanup(s, releaseStream, streamCleanup{
			ptr: s.ptr, cancelProvider: s.cancelProvider, stopContext: s.stopContext,
		})
	}
}

// NextContext waits for a chunk until ctx is done. Cancellation closes the
// stream. A cancelled read is never reused for another iteration.
func (s *LlmStream) NextContext(ctx context.Context) (json.RawMessage, error) {
	if ctx == nil {
		return nil, errors.New("context is nil")
	}
	if err := ctx.Err(); err != nil {
		_ = s.Close()
		return nil, err
	}
	stop := context.AfterFunc(ctx, func() { _ = s.Close() })
	defer stop()
	value, err := s.Next()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return value, err
}
