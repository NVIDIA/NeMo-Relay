// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
)

const (
	getHandleFailed               = "GetHandle failed: %v"
	invalidUUID                   = "not-a-uuid"
	newScopeStackFailed           = "NewScopeStack failed: %v"
	newStackFromPropagationFailed = "NewScopeStackFromPropagation failed: %v"
	propagationParentUUID         = "018f13f0-7c1a-7a80-8000-000000000002"
	propagationRootUUID           = "018f13f0-7c1a-7a80-8000-000000000001"
)

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
		executionResult, err := ToolCallExecute("concurrent_tool", argsJSON, func(args json.RawMessage) (ToolExecutionResult, error) {
			return toolExecutionResult(args), nil
		})
		result = executionResult.Result
		runErr = err
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

func TestNewScopeStackFromPropagationUsesParentAsCurrentHandle(t *testing.T) {
	rootUUID := propagationRootUUID
	parentUUID := propagationParentUUID
	stack, err := NewScopeStackFromPropagation(PropagationContext{
		Version:    1,
		RootUUID:   &rootUUID,
		ParentUUID: parentUUID,
	})
	if err != nil {
		t.Fatalf(newStackFromPropagationFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := GetHandle()
		if err != nil {
			t.Fatalf(getHandleFailed, err)
		}
		if handle.UUID() != parentUUID {
			t.Fatalf("expected parent UUID %s, got %s", parentUUID, handle.UUID())
		}
	})
}

func TestPropagationContextCaptureAndValidation(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(newScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		assertCapturedPropagationContexts(t)
	})
	assertInvalidPropagationContexts(t)
}

func TestCaptureTraceparentUsesCurrentScope(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(newScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("traceparent", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		traceparent, err := CaptureTraceparent()
		if err != nil {
			t.Fatalf("CaptureTraceparent failed: %v", err)
		}
		parentHex := strings.ReplaceAll(handle.UUID(), "-", "")
		expected := fmt.Sprintf("00-%s-%s-01", parentHex, parentHex[len(parentHex)-16:])
		if traceparent != expected {
			t.Fatalf("expected traceparent %s, got %s", expected, traceparent)
		}
	})
}

func assertCapturedPropagationContexts(t *testing.T) {
	t.Helper()
	context, err := CapturePropagationContext()
	if err != nil {
		t.Fatalf("CapturePropagationContext failed: %v", err)
	}
	if context.Version != 1 || context.RootUUID == nil || context.ParentUUID == "" {
		t.Fatalf("unexpected rooted context: %+v", context)
	}
	rootless, err := CaptureRootlessPropagationContext()
	if err != nil || rootless.RootUUID != nil || rootless.ParentUUID != context.ParentUUID {
		t.Fatalf("unexpected explicit rootless context: %+v, %v", rootless, err)
	}

	rootUUID := propagationRootUUID
	withRoot, err := CapturePropagationContextWithRoot(&rootUUID)
	if err != nil {
		t.Fatalf("CapturePropagationContextWithRoot failed: %v", err)
	}
	if withRoot.RootUUID == nil || *withRoot.RootUUID != rootUUID {
		t.Fatalf("expected root UUID %s, got %+v", rootUUID, withRoot.RootUUID)
	}

	withNilRoot, err := CapturePropagationContextWithRoot(nil)
	if err != nil {
		t.Fatalf("CapturePropagationContextWithRoot(nil) failed: %v", err)
	}
	if withNilRoot.RootUUID != nil || withNilRoot.ParentUUID != context.ParentUUID {
		t.Fatalf("unexpected nil-root context: %+v", withNilRoot)
	}
}

func assertInvalidPropagationContexts(t *testing.T) {
	t.Helper()
	invalidRoot := invalidUUID
	if _, err := CapturePropagationContextWithRoot(&invalidRoot); err == nil {
		t.Fatal("expected invalid root UUID to be rejected")
	}

	for _, context := range []PropagationContext{
		{Version: 2, ParentUUID: propagationParentUUID},
		{Version: 1, ParentUUID: invalidUUID},
	} {
		if _, err := NewScopeStackFromPropagation(context); err == nil {
			t.Fatalf("expected invalid context to be rejected: %+v", context)
		}
	}
}

func TestPropagationContextJSONRoundTripAndValidation(t *testing.T) {
	rootUUID := propagationRootUUID
	context := PropagationContext{
		Version:    1,
		RootUUID:   &rootUUID,
		ParentUUID: propagationParentUUID,
	}

	payload, err := context.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal([]byte(payload), &wire); err != nil {
		t.Fatalf("serialized context was not JSON: %v", err)
	}
	if wire["version"] != float64(1) || wire["root_uuid"] != rootUUID || wire["parent_uuid"] != context.ParentUUID {
		t.Fatalf("unexpected propagation JSON: %s", payload)
	}

	decoded, err := PropagationContextFromJSON(payload)
	if err != nil {
		t.Fatalf("PropagationContextFromJSON failed: %v", err)
	}
	if decoded.Version != context.Version || decoded.RootUUID == nil || *decoded.RootUUID != rootUUID || decoded.ParentUUID != context.ParentUUID {
		t.Fatalf("expected round-tripped context %+v, got %+v", context, decoded)
	}

	for _, payload := range []string{
		"not JSON",
		`{"version":2,"parent_uuid":"018f13f0-7c1a-7a80-8000-000000000002"}`,
		`{"version":1,"parent_uuid":"not-a-uuid"}`,
	} {
		if _, err := PropagationContextFromJSON(payload); err == nil {
			t.Fatalf("expected invalid context JSON to be rejected: %s", payload)
		}
	}

	if _, err := (PropagationContext{Version: 1, ParentUUID: invalidUUID}).ToJSON(); err == nil {
		t.Fatal("expected ToJSON to reject an invalid propagation context")
	}
}

func TestNewScopeStackFromRootlessAndRootParentPropagation(t *testing.T) {
	parentUUID := "018f13f0-7c1a-7a80-8000-000000000004"
	for _, context := range []PropagationContext{
		{Version: 1, ParentUUID: parentUUID},
		{Version: 1, RootUUID: &parentUUID, ParentUUID: parentUUID},
	} {
		stack, err := NewScopeStackFromPropagation(context)
		if err != nil {
			t.Fatalf(newStackFromPropagationFailed, err)
		}

		stack.Run(func() {
			handle, err := GetHandle()
			if err != nil {
				t.Fatalf(getHandleFailed, err)
			}
			if handle.UUID() != parentUUID {
				t.Fatalf("expected propagated parent %s, got %s", parentUUID, handle.UUID())
			}
		})
		stack.Close()
	}
}

func TestPropagatedScopeStackRunRestoresOuterBinding(t *testing.T) {
	outer, err := NewScopeStack()
	if err != nil {
		t.Fatalf(newScopeStackFailed, err)
	}
	defer outer.Close()

	parentUUID := "018f13f0-7c1a-7a80-8000-000000000005"
	propagated, err := NewScopeStackFromPropagation(PropagationContext{Version: 1, ParentUUID: parentUUID})
	if err != nil {
		t.Fatalf(newStackFromPropagationFailed, err)
	}
	defer propagated.Close()

	outer.Run(func() {
		outerHandle, err := GetHandle()
		if err != nil {
			t.Fatalf(getHandleFailed, err)
		}
		propagated.Run(func() {
			handle, err := GetHandle()
			if err != nil {
				t.Fatalf(getHandleFailed, err)
			}
			if handle.UUID() != parentUUID {
				t.Fatalf("expected propagated parent %s, got %s", parentUUID, handle.UUID())
			}
		})
		restored, err := GetHandle()
		if err != nil {
			t.Fatalf("GetHandle failed after propagated Run: %v", err)
		}
		if restored.UUID() != outerHandle.UUID() {
			t.Fatalf("expected outer stack to be restored, got %s", restored.UUID())
		}
	})
}

func TestPropagatedScopeStacksRemainIsolated(t *testing.T) {
	rootUUID := propagationRootUUID
	first, err := NewScopeStackFromPropagation(PropagationContext{Version: 1, RootUUID: &rootUUID, ParentUUID: propagationParentUUID})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := NewScopeStackFromPropagation(PropagationContext{Version: 1, RootUUID: &rootUUID, ParentUUID: "018f13f0-7c1a-7a80-8000-000000000003"})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()

	first.Run(func() {
		handle, _ := GetHandle()
		if handle.UUID() != propagationParentUUID {
			t.Fatalf("unexpected first propagated parent: %s", handle.UUID())
		}
	})
	second.Run(func() {
		handle, _ := GetHandle()
		if handle.UUID() != "018f13f0-7c1a-7a80-8000-000000000003" {
			t.Fatalf("unexpected second propagated parent: %s", handle.UUID())
		}
	})
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
