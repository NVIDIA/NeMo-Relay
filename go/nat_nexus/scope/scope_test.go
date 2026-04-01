// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package scope_test

import (
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/scope"
)

// ============================================================================
// WithScope
// ============================================================================

func TestWithScopeNormalReturn(t *testing.T) {
	// Capture the current top-of-stack before pushing.
	before, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle before: %v", err)
	}

	// WithScope pushes and returns a cleanup function.
	cleanup := scope.WithScope("with_scope_test", nat_nexus.ScopeTypeAgent)
	defer cleanup()

	// While inside the scope, the top-of-stack should be our new scope.
	during, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle during: %v", err)
	}
	if during.Name() != "with_scope_test" {
		t.Fatalf("expected 'with_scope_test', got '%s'", during.Name())
	}

	// Call cleanup explicitly to verify double-cleanup is safe (defer will call it again).
	cleanup()

	// After cleanup the scope should be popped.
	after, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("expected stack to return to %s, got %s", before.UUID(), after.UUID())
	}
}

func TestWithScopeDeferCleanup(t *testing.T) {
	before, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle: %v", err)
	}

	func() {
		defer scope.WithScope("deferred_scope", nat_nexus.ScopeTypeFunction)()

		current, err := nat_nexus.GetHandle()
		if err != nil {
			t.Fatalf("GetHandle inside: %v", err)
		}
		if current.Name() != "deferred_scope" {
			t.Fatalf("expected 'deferred_scope', got '%s'", current.Name())
		}
	}()

	after, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not popped after defer")
	}
}

func TestWithScopeCleanupOnPanic(t *testing.T) {
	before, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle: %v", err)
	}

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic, got none")
			}
		}()
		defer scope.WithScope("panic_scope", nat_nexus.ScopeTypeTool)()

		// Verify the scope is pushed.
		current, _ := nat_nexus.GetHandle()
		if current.Name() != "panic_scope" {
			t.Fatalf("expected 'panic_scope', got '%s'", current.Name())
		}

		panic("test panic")
	}()

	// After recovering from panic, scope should be popped.
	after, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after panic: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not popped after panic")
	}
}

// ============================================================================
// WithScopeHandle
// ============================================================================

func TestWithScopeHandleNormalReturn(t *testing.T) {
	handle, cleanup := scope.WithScopeHandle("handle_test", nat_nexus.ScopeTypeAgent)
	defer cleanup()

	if handle == nil {
		t.Fatal("expected non-nil handle")
	}
	if handle.Name() != "handle_test" {
		t.Fatalf("expected 'handle_test', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("expected non-empty UUID")
	}
	if handle.Type() != nat_nexus.ScopeTypeAgent {
		t.Fatalf("expected ScopeTypeAgent, got %d", handle.Type())
	}
}

func TestWithScopeHandleCleanupOnPanic(t *testing.T) {
	before, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle: %v", err)
	}

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic")
			}
		}()
		handle, cleanup := scope.WithScopeHandle("panic_handle", nat_nexus.ScopeTypeFunction)
		defer cleanup()

		if handle == nil {
			t.Fatal("expected non-nil handle")
		}

		panic("test panic")
	}()

	after, err := nat_nexus.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not cleaned up after panic")
	}
}

func TestWithScopeWithOptions(t *testing.T) {
	handle, cleanup := scope.WithScopeHandle(
		"opts_test",
		nat_nexus.ScopeTypeFunction,
		nat_nexus.WithScopeAttributes(nat_nexus.ScopeAttrParallel),
	)
	defer cleanup()

	if handle == nil {
		t.Fatal("expected non-nil handle")
	}
	if handle.Attributes()&nat_nexus.ScopeAttrParallel == 0 {
		t.Fatal("expected PARALLEL attribute to be set")
	}
}

func TestWithScopeNested(t *testing.T) {
	before, _ := nat_nexus.GetHandle()

	h1, cleanup1 := scope.WithScopeHandle("outer", nat_nexus.ScopeTypeAgent)
	defer cleanup1()

	h2, cleanup2 := scope.WithScopeHandle("inner", nat_nexus.ScopeTypeFunction)
	defer cleanup2()

	current, _ := nat_nexus.GetHandle()
	if current.Name() != "inner" {
		t.Fatalf("expected 'inner', got '%s'", current.Name())
	}

	// Pop inner
	cleanup2()
	current, _ = nat_nexus.GetHandle()
	if current.Name() != "outer" {
		t.Fatalf("expected 'outer', got '%s'", current.Name())
	}

	// Pop outer
	cleanup1()
	current, _ = nat_nexus.GetHandle()
	if current.UUID() != before.UUID() {
		t.Fatalf("expected root scope")
	}

	_ = h1
	_ = h2
}
