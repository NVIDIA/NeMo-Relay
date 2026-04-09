// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
)

// ============================================================================
// Scope operations
// ============================================================================

func TestPushPopScope(t *testing.T) {
	handle, err := PushScope("test_scope", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if handle == nil {
		t.Fatal("PushScope returned nil handle")
	}
	if handle.Name() != "test_scope" {
		t.Fatalf("expected name 'test_scope', got '%s'", handle.Name())
	}

	current, err := GetHandle()
	if err != nil {
		t.Fatalf("GetHandle failed: %v", err)
	}
	if current == nil {
		t.Fatal("GetHandle returned nil")
	}
	if current.Name() != "test_scope" {
		t.Fatalf("expected current to be 'test_scope', got '%s'", current.Name())
	}

	err = PopScope(handle)
	if err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
}

func TestScopeHandleProperties(t *testing.T) {
	handle, err := PushScope("props_test", ScopeTypeRetriever)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}
	if handle.Name() != "props_test" {
		t.Fatalf("expected 'props_test', got '%s'", handle.Name())
	}
	if handle.Type() != ScopeTypeRetriever {
		t.Fatalf("expected ScopeTypeRetriever, got %d", handle.Type())
	}
}

func TestPushScopeWithAttributes(t *testing.T) {
	handle, err := PushScope("parallel", ScopeTypeFunction, WithScopeAttributes(ScopeAttrParallel))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	if handle.Attributes()&ScopeAttrParallel == 0 {
		t.Fatal("expected PARALLEL attribute to be set")
	}
}

func TestPushScopeWithParent(t *testing.T) {
	parent, err := PushScope("parent", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope parent failed: %v", err)
	}
	defer PopScope(parent)

	child, err := PushScope("child", ScopeTypeFunction, WithParent(parent))
	if err != nil {
		t.Fatalf("PushScope child failed: %v", err)
	}
	defer PopScope(child)

	if child.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), child.ParentUUID())
	}
}

func TestNestedScopes(t *testing.T) {
	s1, _ := PushScope("level1", ScopeTypeAgent)
	s2, _ := PushScope("level2", ScopeTypeFunction)
	s3, _ := PushScope("level3", ScopeTypeTool)

	current, _ := GetHandle()
	if current.Name() != "level3" {
		t.Fatalf("expected 'level3', got '%s'", current.Name())
	}

	PopScope(s3)
	current, _ = GetHandle()
	if current.Name() != "level2" {
		t.Fatalf("expected 'level2', got '%s'", current.Name())
	}

	PopScope(s2)
	current, _ = GetHandle()
	if current.Name() != "level1" {
		t.Fatalf("expected 'level1', got '%s'", current.Name())
	}

	PopScope(s1)
}

func TestPopInvalidScopeErrors(t *testing.T) {
	handle, _ := PushScope("once", ScopeTypeAgent)
	PopScope(handle)
	err := PopScope(handle)
	if err == nil {
		t.Fatal("expected error when popping already-popped scope")
	}
}

func TestAllScopeTypes(t *testing.T) {
	types := []ScopeType{
		ScopeTypeAgent, ScopeTypeFunction, ScopeTypeTool, ScopeTypeLlm,
		ScopeTypeRetriever, ScopeTypeEmbedder, ScopeTypeReranker,
		ScopeTypeGuardrail, ScopeTypeEvaluator, ScopeTypeCustom, ScopeTypeUnknown,
	}
	for _, st := range types {
		handle, err := PushScope("type_test", st)
		if err != nil {
			t.Fatalf("PushScope with type %d failed: %v", st, err)
		}
		PopScope(handle)
	}
}

// ============================================================================
// Events
// ============================================================================

func TestEmitEvent(t *testing.T) {
	err := EmitEvent("my_mark")
	if err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
}

func TestEmitEventWithData(t *testing.T) {
	err := EmitEvent("data_mark",
		WithEventData(json.RawMessage(`{"key": "value"}`)),
		WithEventMetadata(json.RawMessage(`{"version": 1}`)),
	)
	if err != nil {
		t.Fatalf("EmitEvent with data failed: %v", err)
	}
}

func TestEmitEventWithParent(t *testing.T) {
	handle, _ := PushScope("evt_scope", ScopeTypeAgent)
	defer PopScope(handle)

	err := EmitEvent("scoped_mark", WithEventParent(handle))
	if err != nil {
		t.Fatalf("EmitEvent with parent failed: %v", err)
	}
}

// ============================================================================
// Subscribers
// ============================================================================

func TestSubscriberRegistration(t *testing.T) {
	count := 0
	var mu sync.Mutex
	err := RegisterSubscriber("go_test_sub", func(event Event) {
		mu.Lock()
		count++
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}

	// Push scope emits start event
	handle, _ := PushScope("s", ScopeTypeFunction)
	PopScope(handle)

	mu.Lock()
	c := count
	mu.Unlock()
	if c < 2 {
		t.Fatalf("expected at least 2 events (start+end), got %d", c)
	}

	err = DeregisterSubscriber("go_test_sub")
	if err != nil {
		t.Fatalf("DeregisterSubscriber failed: %v", err)
	}
}

func TestDuplicateSubscriberFails(t *testing.T) {
	RegisterSubscriber("go_dup_sub", func(event Event) {})
	err := RegisterSubscriber("go_dup_sub", func(event Event) {})
	if err == nil {
		t.Fatal("expected error for duplicate subscriber")
	}
	DeregisterSubscriber("go_dup_sub")
}

func TestSubscriberEventProperties(t *testing.T) {
	var events []struct {
		uuid      string
		name      string
		kind      string
		timestamp string
	}
	var mu sync.Mutex

	RegisterSubscriber("go_evt_props", func(event Event) {
		mu.Lock()
		events = append(events, struct {
			uuid      string
			name      string
			kind      string
			timestamp string
		}{
			uuid:      event.UUID(),
			name:      event.Name(),
			kind:      event.Kind(),
			timestamp: event.Timestamp(),
		})
		mu.Unlock()
	})

	handle, _ := PushScope("prop_test", ScopeTypeAgent)
	PopScope(handle)
	DeregisterSubscriber("go_evt_props")

	mu.Lock()
	defer mu.Unlock()
	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}
	if events[0].kind != "ScopeStart" {
		t.Fatalf("expected ScopeStart event, got %s", events[0].kind)
	}
	if events[0].uuid == "" {
		t.Fatal("event UUID is empty")
	}
	if events[0].timestamp == "" {
		t.Fatal("event timestamp is empty")
	}
}

func TestMarkEvent(t *testing.T) {
	var markSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_mark_sub", func(event Event) {
		if event.Kind() == "Mark" {
			mu.Lock()
			markSeen = true
			mu.Unlock()
		}
	})

	EmitEvent("test_mark", WithEventData(json.RawMessage(`{"info": "test"}`)))
	DeregisterSubscriber("go_mark_sub")

	mu.Lock()
	if !markSeen {
		t.Fatal("mark event was not received")
	}
	mu.Unlock()
}

func TestEventScopeTypeOnlyOnScopeEvents(t *testing.T) {
	var seenScope, seenTool, seenLLM, seenMark bool
	var mu sync.Mutex

	RegisterSubscriber("go_scope_type_contract", func(event Event) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case event.Kind() == "ScopeStart" && event.Name() == "scope_type_child":
			seenScope = true
			if event.ScopeType() != "Function" {
				t.Fatalf("expected scope event ScopeType Function, got %q", event.ScopeType())
			}
		case event.Kind() == "ToolStart" && event.Name() == "scope_type_tool":
			seenTool = true
			if event.ScopeType() != "" {
				t.Fatalf("expected tool event ScopeType to be empty, got %q", event.ScopeType())
			}
		case event.Kind() == "LLMStart" && event.Name() == "scope_type_llm":
			seenLLM = true
			if event.ScopeType() != "" {
				t.Fatalf("expected llm event ScopeType to be empty, got %q", event.ScopeType())
			}
		case event.Kind() == "Mark" && event.Name() == "scope_type_mark":
			seenMark = true
			if event.ScopeType() != "" {
				t.Fatalf("expected mark event ScopeType to be empty, got %q", event.ScopeType())
			}
		}
	})
	defer DeregisterSubscriber("go_scope_type_contract")

	parent, err := PushScope("scope_type_parent", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope parent failed: %v", err)
	}
	child, err := PushScope("scope_type_child", ScopeTypeFunction)
	if err != nil {
		t.Fatalf("PushScope child failed: %v", err)
	}

	toolHandle, err := ToolCall("scope_type_tool", json.RawMessage(`{"x":1}`))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if err := ToolCallEnd(toolHandle, json.RawMessage(`{"y":2}`)); err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}

	llmHandle, err := LlmCall("scope_type_llm", makeRequest())
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if err := LlmCallEnd(llmHandle, json.RawMessage(`{"done":true}`)); err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}

	if err := EmitEvent("scope_type_mark"); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}

	if err := PopScope(child); err != nil {
		t.Fatalf("PopScope child failed: %v", err)
	}
	if err := PopScope(parent); err != nil {
		t.Fatalf("PopScope parent failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !seenScope || !seenTool || !seenLLM || !seenMark {
		t.Fatalf("missing expected events: scope=%v tool=%v llm=%v mark=%v", seenScope, seenTool, seenLLM, seenMark)
	}
}

// ============================================================================
// Deeply nested scopes
// ============================================================================

func TestDeeplyNestedScopes(t *testing.T) {
	const depth = 15
	handles := make([]*ScopeHandle, depth)

	for i := 0; i < depth; i++ {
		name := fmt.Sprintf("level_%02d", i)
		h, err := PushScope(name, ScopeTypeFunction)
		if err != nil {
			t.Fatalf("PushScope at depth %d failed: %v", i, err)
		}
		handles[i] = h
	}

	// Verify current scope is the deepest
	current, err := GetHandle()
	if err != nil {
		t.Fatalf("GetHandle failed: %v", err)
	}
	if current.Name() != "level_14" {
		t.Fatalf("expected 'level_14', got '%s'", current.Name())
	}

	// Pop all in reverse order, verifying each level
	for i := depth - 1; i >= 0; i-- {
		err := PopScope(handles[i])
		if err != nil {
			t.Fatalf("PopScope at depth %d failed: %v", i, err)
		}
		if i > 0 {
			current, err := GetHandle()
			if err != nil {
				t.Fatalf("GetHandle after pop at depth %d failed: %v", i, err)
			}
			expectedName := fmt.Sprintf("level_%02d", i-1)
			if current.Name() != expectedName {
				t.Fatalf("after pop depth %d: expected '%s', got '%s'", i, expectedName, current.Name())
			}
		}
	}
}

func TestPushScopeWithCombinedAttributes(t *testing.T) {
	attrs := ScopeAttrParallel | ScopeAttrRelocatable
	handle, err := PushScope("combined_attrs", ScopeTypeAgent, WithScopeAttributes(attrs))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	if handle.Attributes()&ScopeAttrParallel == 0 {
		t.Fatal("expected PARALLEL attribute")
	}
	if handle.Attributes()&ScopeAttrRelocatable == 0 {
		t.Fatal("expected RELOCATABLE attribute")
	}
}

func TestScopeWithDataAndMetadata(t *testing.T) {
	handle, err := PushScope("data_scope", ScopeTypeAgent,
		WithData(json.RawMessage(`{"user_id": "u123"}`)),
		WithMetadata(json.RawMessage(`{"trace_id": "t456"}`)),
	)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	data := handle.Data()
	if data == nil {
		t.Fatal("expected non-nil data")
	}
	var d map[string]interface{}
	json.Unmarshal(data, &d)
	if d["user_id"] != "u123" {
		t.Fatalf("expected user_id=u123, got %v", d["user_id"])
	}

	meta := handle.Metadata()
	if meta == nil {
		t.Fatal("expected non-nil metadata")
	}
	var m map[string]interface{}
	json.Unmarshal(meta, &m)
	if m["trace_id"] != "t456" {
		t.Fatalf("expected trace_id=t456, got %v", m["trace_id"])
	}
}

func TestScopeEventWithDataAndMetadata(t *testing.T) {
	var capturedData, capturedMeta json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_evt_data_meta_sub", func(event Event) {
		if event.Kind() == "Mark" {
			mu.Lock()
			capturedData = append(json.RawMessage(nil), event.Data()...)
			capturedMeta = append(json.RawMessage(nil), event.Metadata()...)
			mu.Unlock()
		}
	})

	EmitEvent("data_meta_mark",
		WithEventData(json.RawMessage(`{"payload": "hello"}`)),
		WithEventMetadata(json.RawMessage(`{"version": 2}`)),
	)
	DeregisterSubscriber("go_evt_data_meta_sub")

	mu.Lock()
	defer mu.Unlock()

	var d map[string]interface{}
	json.Unmarshal(capturedData, &d)
	if d["payload"] != "hello" {
		t.Fatalf("expected payload=hello, got %v", d["payload"])
	}

	var m map[string]interface{}
	json.Unmarshal(capturedMeta, &m)
	if m["version"].(float64) != 2 {
		t.Fatalf("expected version=2, got %v", m["version"])
	}
}

func TestConcurrentScopePushPop(t *testing.T) {
	const goroutines = 10
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			stack, err := NewScopeStack()
			if err != nil {
				errCh <- err
				return
			}
			defer stack.Close()

			stack.Run(func() {
				for j := 0; j < 5; j++ {
					name := "concurrent_scope"
					h, err := PushScope(name, ScopeTypeFunction)
					if err != nil {
						errCh <- err
						return
					}
					if err := PopScope(h); err != nil {
						errCh <- err
						return
					}
				}
			})
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent scope operation failed: %v", err)
	}
}

func TestSubscriberReceivesAllEventFields(t *testing.T) {
	type eventData struct {
		uuid       string
		name       string
		kind       string
		timestamp  string
		parentUUID string
		scopeType  string
	}
	var events []eventData
	var mu sync.Mutex

	RegisterSubscriber("go_full_evt_sub", func(event Event) {
		mu.Lock()
		events = append(events, eventData{
			uuid:       event.UUID(),
			name:       event.Name(),
			kind:       event.Kind(),
			timestamp:  event.Timestamp(),
			parentUUID: event.ParentUUID(),
			scopeType:  event.ScopeType(),
		})
		mu.Unlock()
	})

	handle, _ := PushScope("field_test", ScopeTypeAgent)
	PopScope(handle)
	DeregisterSubscriber("go_full_evt_sub")

	mu.Lock()
	defer mu.Unlock()

	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}

	start := events[0]
	if start.kind != "ScopeStart" {
		t.Fatalf("expected ScopeStart event, got %s", start.kind)
	}
	if start.uuid == "" {
		t.Fatal("event UUID is empty")
	}
	if start.timestamp == "" {
		t.Fatal("event timestamp is empty")
	}
	if start.name != "field_test" {
		t.Fatalf("expected name 'field_test', got '%s'", start.name)
	}
}
