// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nvmagic

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct FfiStream FfiStream;

extern int32_t nvmagic_stream_next(FfiStream* stream, char** out_chunk);
extern void nvmagic_stream_free(FfiStream* stream);
extern void nvmagic_string_free(char* ptr);
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
//	stream, err := nvmagic.LlmStreamCallExecute("chat", req, myExecFn)
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
type LlmStream struct {
	ptr    *C.FfiStream
	closed bool
}

func newLlmStream(ptr *C.FfiStream) *LlmStream {
	if ptr == nil {
		return nil
	}
	s := &LlmStream{ptr: ptr}
	runtime.SetFinalizer(s, func(s *LlmStream) {
		s.Close()
	})
	return s
}

// Next returns the next chunk from the stream as a JSON value. It returns
// [io.EOF] when the stream is exhausted and all chunks have been consumed.
// Any registered stream response intercepts are applied to each chunk before
// it is returned. If the stream has already been closed, Next returns io.EOF.
func (s *LlmStream) Next() (json.RawMessage, error) {
	if s.closed || s.ptr == nil {
		return nil, io.EOF
	}

	var chunk *C.char
	rc := C.nvmagic_stream_next(s.ptr, &chunk)

	switch rc {
	case 1:
		// Chunk available
		text := C.GoString(chunk)
		C.nvmagic_string_free(chunk)
		return json.RawMessage(text), nil
	case 0:
		// Stream done
		return nil, io.EOF
	default:
		// Error
		return nil, lastError()
	}
}

// Close releases the underlying C stream resources. It is safe to call Close
// multiple times; subsequent calls are no-ops. After Close is called, any
// further calls to [LlmStream.Next] return [io.EOF].
func (s *LlmStream) Close() {
	if !s.closed && s.ptr != nil {
		C.nvmagic_stream_free(s.ptr)
		s.ptr = nil
		s.closed = true
	}
}
