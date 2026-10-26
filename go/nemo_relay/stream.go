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
	"strconv"
	"strings"
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
// native stream is released once the handle is unreachable, and the collector
// and finalizer do not run after that.
//
// Each stream carries its own collector and finalizer, so multiple streams can
// operate concurrently without interfering with one another.
type LlmStream struct {
	mu        sync.Mutex
	ptr       *C.FfiStream
	callbacks *streamCallbacks
	cleanup   runtime.Cleanup
	closed    bool
	closing   bool
	inFlight  int
	idle      chan struct{}
	closeDone chan struct{}
	closeErr  error
}

type streamCallbacks struct {
	collector CollectorFunc
	finalizer FinalizerFunc

	mu                sync.Mutex
	callbackGoroutine uint64
	callbackDone      chan struct{}
}

func (c *streamCallbacks) enter() (release func()) {
	done := make(chan struct{})
	c.mu.Lock()
	c.callbackGoroutine = currentGoroutineID()
	c.callbackDone = done
	c.mu.Unlock()
	return func() {
		c.mu.Lock()
		c.callbackGoroutine = 0
		c.callbackDone = nil
		c.mu.Unlock()
		close(done)
	}
}

func (c *streamCallbacks) activeCallback(id uint64) (<-chan struct{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.callbackGoroutine != 0 && c.callbackGoroutine == id {
		return c.callbackDone, true
	}
	return nil, false
}

func currentGoroutineID() uint64 {
	var stack [64]byte
	n := runtime.Stack(stack[:], false)
	fields := strings.Fields(string(stack[:n]))
	if len(fields) < 2 {
		return 0
	}
	id, _ := strconv.ParseUint(fields[1], 10, 64)
	return id
}

func newLlmStream(ptr *C.FfiStream, callbacks *streamCallbacks) *LlmStream {
	if ptr == nil {
		return nil
	}
	s := &LlmStream{ptr: ptr, callbacks: callbacks}
	s.cleanup = runtime.AddCleanup(s, releaseStream, ptr)
	return s
}

// releaseStream frees the native stream of an unreachable handle. Relay
// cancels the producer and emits the end event, and the Go callbacks report
// the release instead of running.
func releaseStream(ptr *C.FfiStream) {
	C.nemo_relay_stream_free(ptr)
}

// Next returns the next chunk from the stream as a JSON value. It returns
// [io.EOF] when the stream is exhausted and all chunks have been consumed.
// Any registered stream execution intercepts are applied to each chunk before
// it is returned.
//
// If the stream has already been closed, Next returns io.EOF.
func (s *LlmStream) Next() (json.RawMessage, error) {
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
	defer s.finishNext()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var chunk *C.char
	switch rc := C.nemo_relay_stream_next(ptr, &chunk); rc {
	case 1:
		text := C.GoString(chunk)
		C.nemo_relay_string_free(chunk)
		return json.RawMessage(text), nil
	case 0:
		return nil, io.EOF
	default:
		return nil, lastError()
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
// stream resources. It is safe to call Close multiple times; later calls
// return the first result. A Close from inside this stream's own collector or
// finalizer returns nil at once and finishes cleanup after the callback
// returns. After Close is called, any further calls to [LlmStream.Next] return
// [io.EOF]. The finalizer runs once with any collected partial response, and a
// finalizer failure is the Close error.
func (s *LlmStream) Close() error {
	callerID := currentGoroutineID()
	s.mu.Lock()
	if s.closed {
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	if s.closing {
		done := s.closeDone
		s.mu.Unlock()
		if _, reentrant := s.callbacks.activeCallback(callerID); reentrant {
			return nil
		}
		<-done
		s.mu.Lock()
		err := s.closeErr
		s.mu.Unlock()
		return err
	}
	s.closing = true
	s.closeDone = make(chan struct{})
	ptr := s.ptr
	s.mu.Unlock()

	if callbackDone, reentrant := s.callbacks.activeCallback(callerID); reentrant {
		go func() {
			<-callbackDone
			s.closeNative(ptr)
		}()
		return nil
	}
	s.closeNative(ptr)
	s.mu.Lock()
	err := s.closeErr
	s.mu.Unlock()
	return err
}

func (s *LlmStream) closeNative(ptr *C.FfiStream) {
	err := callStatus(func() C.int32_t { return C.nemo_relay_stream_close(ptr) })
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
	close(s.closeDone)
	s.mu.Unlock()
	s.cleanup.Stop()
	C.nemo_relay_stream_free(ptr)
	runtime.KeepAlive(s)
}
