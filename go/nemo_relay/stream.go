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
// The stream is not safe for concurrent use. If not closed explicitly, the
// underlying C resources are freed automatically by a Go runtime finalizer.
//
// Each stream carries its own collector and finalizer callbacks, so multiple
// streams can operate concurrently without interfering with one another.
type LlmStream struct {
	ptr       *C.FfiStream
	closed    bool
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
	runtime.SetFinalizer(s, func(s *LlmStream) {
		_ = s.Close()
	})
	return s
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
	if s.closed || s.ptr == nil {
		return nil, io.EOF
	}

	var chunk *C.char
	rc := C.nemo_relay_stream_next(s.ptr, &chunk)

	if rc == 1 {
		// Chunk available
		text := C.GoString(chunk)
		C.nemo_relay_string_free(chunk)
		return llmStreamNextResult(int32(rc), json.RawMessage(text), s.collector, &s.finalizer)
	}
	return llmStreamNextResult(int32(rc), nil, s.collector, &s.finalizer)
}

// Close stops the producer, waits for cleanup, and releases the underlying C
// stream resources. It is safe to call Close multiple times; subsequent calls
// are no-ops. After Close is called, any further calls to [LlmStream.Next]
// return [io.EOF]. The finalizer runs once with any collected partial response.
func (s *LlmStream) Close() error {
	if !s.closed && s.ptr != nil {
		status := C.nemo_relay_stream_close(s.ptr)
		var err error
		if status != 0 {
			err = lastError()
		}
		C.nemo_relay_stream_free(s.ptr)
		s.ptr = nil
		s.closed = true
		s.collector = nil
		if s.finalizer != nil {
			s.finalizer()
			s.finalizer = nil
		}
		return err
	}
	return nil
}
