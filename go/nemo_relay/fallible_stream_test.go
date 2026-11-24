// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestCollectorReturnedErrorTerminatesStream(t *testing.T) {
	runWithTestScopeStack(t, func() {
		_, metadata, done := captureLlmEndEvent(t, "go_collector_error")
		stream, err := LlmStreamCallExecute("collector-error", requestForTest(makeRequest()),
			LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"chunk":1}`), nil }).AsStream(),
			func(json.RawMessage) error { return errors.New("collector storage unavailable") },
			func() (json.RawMessage, error) { return json.RawMessage(`{"partial":true}`), nil })
		if err != nil {
			t.Fatal(err)
		}
		_, err = stream.Next()
		if !errors.Is(err, ErrUnknownStatus) || !strings.Contains(err.Error(), "collector storage unavailable") {
			t.Fatalf("collector error lost: %v", err)
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("already delivered error repeated: %v", err)
		}
		done()
		if metadata()["otel.status_code"] != "ERROR" {
			t.Fatalf("collector failure must mark end ERROR: %v", metadata())
		}
	})
}

func TestFinalizerReturnedErrorReportedByClose(t *testing.T) {
	runWithTestScopeStack(t, func() {
		output, metadata, done := captureLlmEndEvent(t, "go_finalizer_error")
		stream, err := LlmStreamCallExecute("finalizer-error", requestForTest(makeRequest()),
			LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"chunk":1}`), nil }).AsStream(), nil,
			func() (json.RawMessage, error) {
				return json.RawMessage(`{"discard":true}`), errors.New("aggregate persistence failed")
			},
			WithLLMData(json.RawMessage(`{"fallback":true}`)))
		if err != nil {
			t.Fatal(err)
		}
		drainStream(t, stream)
		closeErr := stream.Close()
		if !errors.Is(closeErr, ErrInternal) || !strings.Contains(closeErr.Error(), "aggregate persistence failed") {
			t.Fatalf("finalizer error lost: %v", closeErr)
		}
		done()
		assertFinalizerFailureEndEvent(t, output, metadata, closeErr)
	})
}

func TestFinalizerNilJSONMeansNull(t *testing.T) {
	runWithTestScopeStack(t, func() {
		stream, err := LlmStreamCallExecute("nil-aggregate", requestForTest(makeRequest()),
			LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`1`), nil }).AsStream(), nil,
			func() (json.RawMessage, error) { return nil, nil })
		if err != nil {
			t.Fatal(err)
		}
		drainStream(t, stream)
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestUndeliveredCollectorErrorReportedByClose(t *testing.T) {
	runWithTestScopeStack(t, func() {
		failed := make(chan struct{})
		calls := 0
		source := &contractSource{chunks: []json.RawMessage{json.RawMessage(`1`), json.RawMessage(`2`)}}
		stream, err := LlmStreamCallExecute("collector-error-on-close", requestForTest(makeRequest()),
			func(context.Context, json.RawMessage) (LLMStreamSource, error) { return source, nil },
			func(json.RawMessage) error {
				calls++
				if calls == 2 {
					close(failed)
					return errors.New("undelivered collector error")
				}
				return nil
			}, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		if _, err := stream.Next(); err != nil {
			t.Fatal(err)
		}
		select {
		case <-failed:
		case <-time.After(5 * time.Second):
			t.Fatal("collector did not fail")
		}
		err = stream.Close()
		if !errors.Is(err, ErrInternal) || !strings.Contains(err.Error(), "undelivered collector error") {
			t.Fatalf("close lost undelivered error: %v", err)
		}
	})
}
