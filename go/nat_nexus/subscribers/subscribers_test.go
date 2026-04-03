// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package subscribers_test

import (
	"sync"
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	subscriberspkg "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/subscribers"
)

func TestSubscriberShorthands(t *testing.T) {
	var seenStart bool
	var mu sync.Mutex

	if err := subscriberspkg.Register("subs_global", func(event *nat_nexus.Event) {
		if event.Type() == nat_nexus.EventTypeStart {
			mu.Lock()
			seenStart = true
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	handle, err := nat_nexus.PushScope("subs_scope", nat_nexus.ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := nat_nexus.PopScope(handle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := subscriberspkg.Deregister("subs_global"); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}

	mu.Lock()
	if !seenStart {
		t.Fatal("expected global subscriber to see a start event")
	}
	mu.Unlock()
}

func TestScopeSubscriberShorthands(t *testing.T) {
	stack, err := nat_nexus.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nat_nexus.PushScope("subs_local_scope", nat_nexus.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nat_nexus.PopScope(handle)

		var markCount int
		var mu sync.Mutex
		if err := subscriberspkg.ScopeRegister(handle.UUID(), "subs_local", func(event *nat_nexus.Event) {
			if event.Type() == nat_nexus.EventTypeMark {
				mu.Lock()
				markCount++
				mu.Unlock()
			}
		}); err != nil {
			t.Fatalf("ScopeRegister failed: %v", err)
		}

		if err := nat_nexus.EmitEvent("first-mark"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}
		if err := subscriberspkg.ScopeDeregister(handle.UUID(), "subs_local"); err != nil {
			t.Fatalf("ScopeDeregister failed: %v", err)
		}
		if err := nat_nexus.EmitEvent("second-mark"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}

		mu.Lock()
		if markCount != 1 {
			t.Fatalf("expected exactly one scoped mark, got %d", markCount)
		}
		mu.Unlock()
	})
}
