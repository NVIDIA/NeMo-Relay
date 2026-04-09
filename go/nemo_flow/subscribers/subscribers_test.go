// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package subscribers_test

import (
	"sync"
	"testing"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
	subscriberspkg "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/subscribers"
)

func TestSubscriberShorthands(t *testing.T) {
	var seenStart bool
	var mu sync.Mutex

	if err := subscriberspkg.Register("subs_global", func(event nemo_flow.Event) {
		if event.Kind() == "ScopeStart" {
			mu.Lock()
			seenStart = true
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	handle, err := nemo_flow.PushScope("subs_scope", nemo_flow.ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := nemo_flow.PopScope(handle); err != nil {
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
	stack, err := nemo_flow.NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := nemo_flow.PushScope("subs_local_scope", nemo_flow.ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer nemo_flow.PopScope(handle)

		var markCount int
		var mu sync.Mutex
		if err := subscriberspkg.ScopeRegister(handle.UUID(), "subs_local", func(event nemo_flow.Event) {
			if event.Kind() == "Mark" {
				mu.Lock()
				markCount++
				mu.Unlock()
			}
		}); err != nil {
			t.Fatalf("ScopeRegister failed: %v", err)
		}

		if err := nemo_flow.EmitEvent("first-mark"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}
		if err := subscriberspkg.ScopeDeregister(handle.UUID(), "subs_local"); err != nil {
			t.Fatalf("ScopeDeregister failed: %v", err)
		}
		if err := nemo_flow.EmitEvent("second-mark"); err != nil {
			t.Fatalf("EmitEvent failed: %v", err)
		}

		mu.Lock()
		if markCount != 1 {
			t.Fatalf("expected exactly one scoped mark, got %d", markCount)
		}
		mu.Unlock()
	})
}
