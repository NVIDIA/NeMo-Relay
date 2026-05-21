// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package subscribers_test

import (
	"sync"
	"testing"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	subscriberspkg "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/subscribers"
)

func assertSeenStart(t *testing.T, seenStart bool) {
	t.Helper()
	if !seenStart {
		t.Fatal("expected global subscriber to see a start event")
	}
}

func countScopedMarks(t *testing.T, handle *nemo_relay.ScopeHandle) int {
	t.Helper()

	var markCount int
	var mu sync.Mutex
	if err := subscriberspkg.ScopeRegister(handle.UUID(), "subs_local", func(event nemo_relay.Event) {
		if event.Kind() == "mark" {
			mu.Lock()
			markCount++
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("ScopeRegister failed: %v", err)
	}

	if err := nemo_relay.EmitEvent("first-mark"); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if err := subscriberspkg.ScopeDeregister(handle.UUID(), "subs_local"); err != nil {
		t.Fatalf("ScopeDeregister failed: %v", err)
	}
	if err := nemo_relay.EmitEvent("second-mark"); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	return markCount
}

func TestSubscriberShorthands(t *testing.T) {
	var seenStart bool
	var mu sync.Mutex

	if err := subscriberspkg.Register("subs_global", func(event nemo_relay.Event) {
		if event.Kind() == "scope" && event.ScopeCategory() == "start" {
			mu.Lock()
			seenStart = true
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	handle, err := nemo_relay.PushScope("subs_scope", nemo_relay.ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := nemo_relay.PopScope(handle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := subscriberspkg.Deregister("subs_global"); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}

	mu.Lock()
	assertSeenStart(t, seenStart)
	mu.Unlock()
}

func TestScopeSubscriberShorthands(t *testing.T) {
	stack, err := nemo_relay.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nemo_relay.PushScope("subs_local_scope", nemo_relay.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nemo_relay.PopScope(handle)

		markCount := countScopedMarks(t, handle)
		if markCount != 1 {
			t.Fatalf("expected exactly one scoped mark, got %d", markCount)
		}
	})
}
