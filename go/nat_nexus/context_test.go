// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
	"fmt"
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

// ============================================================================
// Multiple goroutines with independent scope stacks
// ============================================================================

func TestMultipleGoroutinesIndependentScopeStacks(t *testing.T) {
	const goroutines = 8
	var wg sync.WaitGroup
	results := make([]string, goroutines)
	errs := make([]error, goroutines)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			stack, err := NewScopeStack()
			if err != nil {
				errs[idx] = err
				return
			}
			defer stack.Close()

			stack.Run(func() {
				scopeName := "goroutine_scope"
				handle, err := PushScope(scopeName, ScopeTypeAgent)
				if err != nil {
					errs[idx] = err
					return
				}
				defer PopScope(handle)

				h, err := GetHandle()
				if err != nil {
					errs[idx] = err
					return
				}
				results[idx] = h.Name()
			})
		}(i)
	}

	wg.Wait()

	for i := 0; i < goroutines; i++ {
		if errs[i] != nil {
			t.Fatalf("goroutine %d failed: %v", i, errs[i])
		}
		if results[i] != "goroutine_scope" {
			t.Fatalf("goroutine %d: expected 'goroutine_scope', got '%s'", i, results[i])
		}
	}
}

func TestCreateScopeStackCreatesFreshStack(t *testing.T) {
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

	// Each stack should be independent - push a scope in stack1,
	// verify stack2 does not see it
	var name1, name2 string
	var wg sync.WaitGroup

	wg.Add(2)

	go func() {
		defer wg.Done()
		stack1.Run(func() {
			h, _ := PushScope("stack1_scope", ScopeTypeAgent)
			defer PopScope(h)
			current, _ := GetHandle()
			name1 = current.Name()
		})
	}()

	go func() {
		defer wg.Done()
		stack2.Run(func() {
			// Should see root scope, not stack1_scope
			h, _ := PushScope("stack2_scope", ScopeTypeAgent)
			defer PopScope(h)
			current, _ := GetHandle()
			name2 = current.Name()
		})
	}()

	wg.Wait()

	if name1 != "stack1_scope" {
		t.Fatalf("expected 'stack1_scope', got '%s'", name1)
	}
	if name2 != "stack2_scope" {
		t.Fatalf("expected 'stack2_scope', got '%s'", name2)
	}
}

func TestConcurrentScopeStacksWithToolCalls(t *testing.T) {
	const goroutines = 5
	var wg sync.WaitGroup
	results := make([]json.RawMessage, goroutines)
	errs := make([]error, goroutines)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			stack, err := NewScopeStack()
			if err != nil {
				errs[idx] = err
				return
			}
			defer stack.Close()

			stack.Run(func() {
				handle, err := PushScope("tool_scope", ScopeTypeAgent)
				if err != nil {
					errs[idx] = err
					return
				}
				defer PopScope(handle)

				argsJSON := json.RawMessage(fmt.Sprintf(`{"index": %d}`, idx))
				result, err := ToolCallExecute("concurrent_tool", argsJSON,
					func(args json.RawMessage) (json.RawMessage, error) {
						return args, nil
					},
				)
				if err != nil {
					errs[idx] = err
					return
				}
				results[idx] = result
			})
		}(i)
	}

	wg.Wait()

	for i := 0; i < goroutines; i++ {
		if errs[i] != nil {
			t.Fatalf("goroutine %d failed: %v", i, errs[i])
		}
		if results[i] == nil {
			t.Fatalf("goroutine %d returned nil result", i)
		}
	}
}

func TestScopeStackRunNestedScopes(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		// Build up a scope hierarchy inside a dedicated scope stack
		s1, err := PushScope("agent", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope agent failed: %v", err)
		}

		s2, err := PushScope("function", ScopeTypeFunction)
		if err != nil {
			t.Fatalf("PushScope function failed: %v", err)
		}

		s3, err := PushScope("tool", ScopeTypeTool)
		if err != nil {
			t.Fatalf("PushScope tool failed: %v", err)
		}

		current, _ := GetHandle()
		if current.Name() != "tool" {
			t.Fatalf("expected 'tool', got '%s'", current.Name())
		}

		PopScope(s3)
		current, _ = GetHandle()
		if current.Name() != "function" {
			t.Fatalf("expected 'function', got '%s'", current.Name())
		}

		PopScope(s2)
		current, _ = GetHandle()
		if current.Name() != "agent" {
			t.Fatalf("expected 'agent', got '%s'", current.Name())
		}

		PopScope(s1)
	})
}
