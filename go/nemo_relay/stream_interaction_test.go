// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"testing"
)

type interactionSource struct {
	next  func(context.Context) (json.RawMessage, error)
	close func() error
}

func (source *interactionSource) Next(ctx context.Context) (json.RawMessage, error) {
	if source.next == nil {
		return nil, io.EOF
	}
	return source.next(ctx)
}
func (source *interactionSource) Close() error {
	if source.close == nil {
		return nil
	}
	return source.close()
}

func TestStreamCallbacksCanEmitObservabilityEvents(t *testing.T) {
	runWithTestScopeStack(t, func() {
		var mu sync.Mutex
		counts := map[string]int{}
		if err := RegisterSubscriber("callback-events", func(event Event) {
			if event.Kind() == "mark" {
				mu.Lock()
				counts[event.Name()]++
				mu.Unlock()
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("callback-events")
		read := false
		source := &interactionSource{
			next: func(context.Context) (json.RawMessage, error) {
				if read {
					return nil, io.EOF
				}
				read = true
				return json.RawMessage(`1`), EmitEvent("callback-read")
			},
			close: func() error { return EmitEvent("callback-close") },
		}
		stream, err := LlmStreamCallExecute("callback-events-stream", requestForTest(makeRequest()),
			func(context.Context, json.RawMessage) (LLMStreamSource, error) {
				return source, EmitEvent("callback-open")
			},
			func(json.RawMessage) error { return EmitEvent("callback-collector") },
			func() (json.RawMessage, error) { return nil, EmitEvent("callback-finalizer") })
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		drainStream(t, stream)
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		// Lifecycle and flush operations run outside owned callbacks.
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		for _, phase := range []string{"open", "read", "collector", "finalizer", "close"} {
			if count := counts["callback-"+phase]; count != 1 {
				t.Errorf("%s emitted %d events, want 1", phase, count)
			}
		}
	})
}
