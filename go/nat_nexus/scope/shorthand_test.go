// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package scope_test

import (
	"sync"
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/scope"
)

func TestScopeShorthands(t *testing.T) {
	var sawMark bool
	var mu sync.Mutex

	if err := nat_nexus.RegisterSubscriber("scope_shortcuts_sub", func(event nat_nexus.Event) {
		if event.Kind() == "Mark" && event.Name() == "scope-mark" {
			mu.Lock()
			sawMark = true
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer nat_nexus.DeregisterSubscriber("scope_shortcuts_sub")

	handle, err := scope.Push("scope-shortcuts", nat_nexus.ScopeTypeFunction)
	if err != nil {
		t.Fatalf("Push failed: %v", err)
	}

	current, err := scope.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle failed: %v", err)
	}
	if current.UUID() != handle.UUID() {
		t.Fatalf("expected current scope %s, got %s", handle.UUID(), current.UUID())
	}

	if err := scope.Event("scope-mark"); err != nil {
		t.Fatalf("Event failed: %v", err)
	}
	if err := scope.Pop(handle); err != nil {
		t.Fatalf("Pop failed: %v", err)
	}

	mu.Lock()
	if !sawMark {
		t.Fatal("expected to observe scoped mark event")
	}
	mu.Unlock()
}
