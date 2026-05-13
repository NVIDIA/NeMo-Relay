// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"testing"
)

const newScopeStackFailed = "NewScopeStack failed: %v"

func scopeNameInStack(stack *ScopeStack, scopeName string, scopeType ScopeType) (string, error) {
	var currentName string
	var runErr error
	stack.Run(func() {
		handle, err := PushScope(scopeName, scopeType)
		if err != nil {
			runErr = err
			return
		}
		defer PopScope(handle)

		current, err := GetHandle()
		if err != nil {
			runErr = err
			return
		}
		currentName = current.Name()
	})
	return currentName, runErr
}

func runScopeNameAsync(
	wg *sync.WaitGroup,
	stack *ScopeStack,
	scopeName string,
	scopeType ScopeType,
	name *string,
	err *error,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		*name, *err = scopeNameInStack(stack, scopeName, scopeType)
	}()
}

func runIsolatedScopeName(idx int, results []string, errs []error) {
	stack, err := NewScopeStack()
	if err != nil {
		errs[idx] = err
		return
	}
	defer stack.Close()

	name, err := scopeNameInStack(stack, "goroutine_scope", ScopeTypeAgent)
	if err != nil {
		errs[idx] = err
		return
	}
	results[idx] = name
}

func concurrentToolCallResult(idx int) (json.RawMessage, error) {
	stack, err := NewScopeStack()
	if err != nil {
		return nil, err
	}
	defer stack.Close()

	var result json.RawMessage
	var runErr error
	stack.Run(func() {
		handle, err := PushScope("tool_scope", ScopeTypeAgent)
		if err != nil {
			runErr = err
			return
		}
		defer PopScope(handle)

		argsJSON := json.RawMessage(fmt.Sprintf(`{"index": %d}`, idx))
		result, runErr = ToolCallExecute("concurrent_tool", argsJSON, func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		})
	})
	return result, runErr
}

func TestNewScopeStack(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(newScopeStackFailed, err)
	}
	defer stack.Close()

	if stack.ptr == nil {
		t.Fatal("expected non-nil ptr")
	}
}

func TestScopeStackClose(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(newScopeStackFailed, err)
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
		t.Fatalf(newScopeStackFailed, err)
	}
	defer stack.Close()

	var active bool
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if ScopeStackActive() {
		t.Fatal("expected ScopeStackActive() to be false before Run")
	}
	stack.Run(func() {
		active = ScopeStackActive()
	})

	if !active {
		t.Error("expected ScopeStackActive() to be true inside Run")
	}
	if ScopeStackActive() {
		t.Error("expected ScopeStackActive() to be restored after Run")
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
	var err1, err2 error

	runScopeNameAsync(&wg, stack1, "goroutine1_scope", ScopeTypeAgent, &name1, &err1)
	runScopeNameAsync(&wg, stack2, "goroutine2_scope", ScopeTypeTool, &name2, &err2)

	wg.Wait()

	if err1 != nil {
		t.Fatalf("scope stack 1 failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("scope stack 2 failed: %v", err2)
	}

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
			runIsolatedScopeName(idx, results, errs)
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
			results[idx], errs[idx] = concurrentToolCallResult(idx)
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
		t.Fatalf(newScopeStackFailed, err)
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
