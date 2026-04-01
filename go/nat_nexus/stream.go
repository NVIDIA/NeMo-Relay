// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiStream FfiStream;

extern int32_t nat_nexus_stream_next(FfiStream* stream, char** out_chunk);
extern void nat_nexus_stream_free(FfiStream* stream);
extern void nat_nexus_string_free(char* ptr);
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
//	stream, err := nat_nexus.LlmStreamCallExecute("chat", req, myExecFn, collector, finalizer)
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
		s.Close()
	})
	return s
}

// Next returns the next chunk from the stream as a JSON value. It returns
// [io.EOF] when the stream is exhausted and all chunks have been consumed.
// Any registered stream response intercepts are applied to each chunk before
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
	rc := C.nat_nexus_stream_next(s.ptr, &chunk)

	switch rc {
	case 1:
		// Chunk available
		text := C.GoString(chunk)
		C.nat_nexus_string_free(chunk)
		msg := json.RawMessage(text)
		// Feed chunk to the per-stream collector if one was provided.
		if s.collector != nil {
			s.collector(msg)
		}
		return msg, nil
	case 0:
		// Stream done -- invoke the per-stream finalizer if one was provided.
		if s.finalizer != nil {
			s.finalizer()
			s.finalizer = nil // ensure single invocation
		}
		return nil, io.EOF
	default:
		// Error
		return nil, lastError()
	}
}

// Close releases the underlying C stream resources. It is safe to call Close
// multiple times; subsequent calls are no-ops. After Close is called, any
// further calls to [LlmStream.Next] return [io.EOF].
//
// If the stream has not been fully consumed, the finalizer (if provided) will
// NOT be called. Callers should consume the stream to completion or explicitly
// handle finalization if needed before closing early.
func (s *LlmStream) Close() {
	if !s.closed && s.ptr != nil {
		C.nat_nexus_stream_free(s.ptr)
		s.ptr = nil
		s.closed = true
		s.collector = nil
		s.finalizer = nil
	}
}
