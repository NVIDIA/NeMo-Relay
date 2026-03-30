// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"sync"
	"testing"
)

func TestNewScopeStack(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	if stack.ptr == nil {
		t.Fatal("expected non-nil ptr")
	}
}

func TestScopeStackClose(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	stack.Close()
	// Double close should be safe
	stack.Close()

	if stack.ptr != nil {
		t.Fatal("expected nil ptr after Close")
	}
}

func TestScopeStackActiveInsideRun(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	var active bool
	stack.Run(func() {
		active = ScopeStackActive()
	})

	if !active {
		t.Error("expected ScopeStackActive() to be true inside Run")
	}
}

func TestScopeStackRunIsolation(t *testing.T) {
	stack1, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 1 failed: %v", err)
	}
	defer stack1.Close()

	stack2, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 2 failed: %v", err)
	}
	defer stack2.Close()

	var wg sync.WaitGroup
	var name1, name2 string

	wg.Add(2)

	go func() {
		defer wg.Done()
		stack1.Run(func() {
			handle, err := PushScope("goroutine1_scope", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope in goroutine 1 failed: %v", err)
				return
			}
			_ = handle
			h, err := GetHandle()
			if err != nil {
				t.Errorf("GetHandle in goroutine 1 failed: %v", err)
				return
			}
			name1 = h.Name()
		})
	}()

	go func() {
		defer wg.Done()
		stack2.Run(func() {
			handle, err := PushScope("goroutine2_scope", ScopeTypeTool)
			if err != nil {
				t.Errorf("PushScope in goroutine 2 failed: %v", err)
				return
			}
			_ = handle
			h, err := GetHandle()
			if err != nil {
				t.Errorf("GetHandle in goroutine 2 failed: %v", err)
				return
			}
			name2 = h.Name()
		})
	}()

	wg.Wait()

	if name1 != "goroutine1_scope" {
		t.Errorf("expected 'goroutine1_scope', got '%s'", name1)
	}
	if name2 != "goroutine2_scope" {
		t.Errorf("expected 'goroutine2_scope', got '%s'", name2)
	}
}
