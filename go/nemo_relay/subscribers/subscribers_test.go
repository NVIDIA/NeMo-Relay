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

func TestSubscriptionHandleShorthand(t *testing.T) {
	var events []string
	var mu sync.Mutex

	subscription, err := subscriberspkg.Subscribe(func(event nemo_relay.Event) {
		mu.Lock()
		events = append(events, event.Name())
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}

	if err := nemo_relay.EmitEvent("subs_handle_before_close"); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if closed, err := subscription.Close(); err != nil {
		t.Fatalf("Subscription Close failed: %v", err)
	} else if !closed {
		t.Fatal("Subscription Close did not report a live close")
	}
	if closed, err := subscription.Close(); err != nil {
		t.Fatalf("second Subscription Close failed: %v", err)
	} else if closed {
		t.Fatal("second Subscription Close reported a live close")
	}
	if err := nemo_relay.EmitEvent("subs_handle_after_close"); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 1 || events[0] != "subs_handle_before_close" {
		t.Fatalf("unexpected subscription events: %#v", events)
	}
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

func TestScopeSubscriptionHandleShorthand(t *testing.T) {
	stack, err := nemo_relay.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nemo_relay.PushScope("subs_scope_handle_owner", nemo_relay.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		popped := false
		defer func() {
			if !popped {
				_ = nemo_relay.PopScope(handle)
			}
		}()

		var explicitEvents []string
		var explicitMu sync.Mutex
		explicit, err := subscriberspkg.ScopeSubscribe(handle.UUID(), func(event nemo_relay.Event) {
			explicitMu.Lock()
			explicitEvents = append(explicitEvents, event.Name())
			explicitMu.Unlock()
		})
		if err != nil {
			t.Fatalf("ScopeSubscribe failed: %v", err)
		}
		if err := nemo_relay.EmitEvent("subs_scope_handle_before_close"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}
		if closed, err := explicit.Close(); err != nil {
			t.Fatalf("Scope subscription Close failed: %v", err)
		} else if !closed {
			t.Fatal("Scope subscription Close did not report a live close")
		}
		if closed, err := explicit.Close(); err != nil {
			t.Fatalf("second Scope subscription Close failed: %v", err)
		} else if closed {
			t.Fatal("second Scope subscription Close reported a live close")
		}
		if err := nemo_relay.EmitEvent("subs_scope_handle_after_close"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}

		explicitMu.Lock()
		if len(explicitEvents) != 1 || explicitEvents[0] != "subs_scope_handle_before_close" {
			t.Fatalf("unexpected explicit scope subscription events: %#v", explicitEvents)
		}
		explicitMu.Unlock()

		var cleanupEvents []string
		var cleanupMu sync.Mutex
		cleanup, err := subscriberspkg.ScopeSubscribe(handle.UUID(), func(event nemo_relay.Event) {
			cleanupMu.Lock()
			cleanupEvents = append(cleanupEvents, event.Name())
			cleanupMu.Unlock()
		})
		if err != nil {
			t.Fatalf("ScopeSubscribe cleanup failed: %v", err)
		}
		if err := nemo_relay.PopScope(handle); err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
		popped = true
		if closed, err := cleanup.Close(); err != nil {
			t.Fatalf("cleanup subscription Close failed: %v", err)
		} else if closed {
			t.Fatal("cleanup subscription Close reported a live close after scope cleanup")
		}

		cleanupMu.Lock()
		defer cleanupMu.Unlock()
		if len(cleanupEvents) != 1 || cleanupEvents[0] != "subs_scope_handle_owner" {
			t.Fatalf("unexpected cleanup scope subscription events: %#v", cleanupEvents)
		}
	})
}
