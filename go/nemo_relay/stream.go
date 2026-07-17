// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiStream FfiStream;

extern int32_t nemo_relay_stream_next(FfiStream* stream, char** out_chunk);
extern int32_t nemo_relay_stream_close(FfiStream* stream);
extern void nemo_relay_stream_free(FfiStream* stream);
extern void nemo_relay_string_free(char* ptr);
*/
import "C"

import (
	"encoding/json"
	"io"
	"runtime"
	"sync"
)

// LlmStream wraps a streaming LLM response returned by [LlmStreamCallExecute].
// It provides an iterator-style interface for consuming Server-Sent Event (SSE)
// chunks from the LLM.
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
// underlying C resources are freed automatically by a Go runtime finalizer.
//
// Each stream carries its own collector and finalizer callbacks, so multiple
// streams can operate concurrently without interfering with one another.
type LlmStream struct {
	nextMu    sync.Mutex
	mu        sync.Mutex
	ptr       *C.FfiStream
	closed    bool
	closing   bool
	inFlight  int
	idle      chan struct{}
	closeDone chan struct{}
	closeErr  error
	collector CollectorFunc
	finalizer FinalizerFunc
}

func llmStreamNextResult(rc int32, chunk json.RawMessage, collector CollectorFunc, finalizer *FinalizerFunc) (json.RawMessage, error) {
	switch rc {
	case 1:
		if collector != nil {
			collector(chunk)
		}
		return chunk, nil
	case 0:
		if finalizer != nil && *finalizer != nil {
			(*finalizer)()
			*finalizer = nil
		}
		return nil, io.EOF
	default:
		return nil, lastError()
	}
}

func newLlmStream(ptr *C.FfiStream, collector CollectorFunc, finalizer FinalizerFunc) *LlmStream {
	if ptr == nil {
		return nil
	}
	s := &LlmStream{
		ptr:       ptr,
		collector: collector,
		finalizer: finalizer,
	}
	runtime.SetFinalizer(s, (*LlmStream).release)
	return s
}

// release frees the native handle without waiting for producer cleanup or
// invoking user callbacks. Explicit Close owns deterministic cleanup.
func (s *LlmStream) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ptr == nil {
		return
	}
	C.nemo_relay_stream_free(s.ptr)
	s.ptr = nil
	s.closed = true
	s.collector = nil
	s.finalizer = nil
}

// Next returns the next chunk from the stream as a JSON value. It returns
// [io.EOF] when the stream is exhausted and all chunks have been consumed.
// Any registered stream execution intercepts are applied to each chunk before
// it is returned.
//
// If a collector function was provided when creating the stream, it is called
// with each chunk. When the stream is exhausted (EOF), the finalizer function
// (if provided) is called exactly once.
//
// If the stream has already been closed, Next returns io.EOF.
func (s *LlmStream) Next() (json.RawMessage, error) {
	s.nextMu.Lock()
	defer s.nextMu.Unlock()

	s.mu.Lock()
	if s.closed || s.closing || s.ptr == nil {
		s.mu.Unlock()
		return nil, io.EOF
	}
	ptr := s.ptr
	s.inFlight++
	if s.inFlight == 1 {
		s.idle = make(chan struct{})
	}
	s.mu.Unlock()

	var chunk *C.char
	rc := C.nemo_relay_stream_next(ptr, &chunk)
	defer s.finishNext()

	if rc == 1 {
		// Chunk available
		text := C.GoString(chunk)
		C.nemo_relay_string_free(chunk)
		chunk := json.RawMessage(text)
		s.mu.Lock()
		collector := s.collector
		s.mu.Unlock()
		if collector != nil {
			collector(chunk)
		}
		return chunk, nil
	}
	if rc == 0 {
		s.mu.Lock()
		finalizer := s.finalizer
		s.finalizer = nil
		s.mu.Unlock()
		if finalizer != nil {
			finalizer()
		}
		return nil, io.EOF
	}
	return nil, lastError()
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
// stream resources. It is safe to call Close multiple times; subsequent calls
// are no-ops. After Close is called, any further calls to [LlmStream.Next]
// return [io.EOF]. The finalizer runs once with any collected partial response.
func (s *LlmStream) Close() error {
	s.mu.Lock()
	if s.closing {
		done := s.closeDone
		s.mu.Unlock()
		<-done
		s.mu.Lock()
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	if s.closed || s.ptr == nil {
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	s.closing = true
	s.closeDone = make(chan struct{})
	ptr := s.ptr
	runtime.SetFinalizer(s, nil)
	s.mu.Unlock()

	status := C.nemo_relay_stream_close(ptr)
	var err error
	if status != 0 {
		err = lastError()
	}

	s.mu.Lock()
	if s.inFlight > 0 {
		s.mu.Unlock()
		// A collector can close its own stream. Finish cleanup after the callback
		// returns instead of waiting here and deadlocking that callback.
		go s.finishClose(ptr, err)
		return err
	}
	s.mu.Unlock()
	s.finishClose(ptr, err)
	return err
}

func (s *LlmStream) finishClose(ptr *C.FfiStream, err error) {
	s.mu.Lock()
	for s.inFlight > 0 {
		idle := s.idle
		s.mu.Unlock()
		<-idle
		s.mu.Lock()
	}
	s.closeErr = err
	s.ptr = nil
	s.closed = true
	s.collector = nil
	finalizer := s.finalizer
	s.finalizer = nil
	s.mu.Unlock()

	C.nemo_relay_stream_free(ptr)

	s.mu.Lock()
	s.closing = false
	close(s.closeDone)
	s.mu.Unlock()

	if finalizer != nil {
		finalizer()
	}
}
