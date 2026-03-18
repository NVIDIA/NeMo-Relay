// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
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
	err := RegisterSubscriber("go_test_sub", func(event *Event) {
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
	RegisterSubscriber("go_dup_sub", func(event *Event) {})
	err := RegisterSubscriber("go_dup_sub", func(event *Event) {})
	if err == nil {
		t.Fatal("expected error for duplicate subscriber")
	}
	DeregisterSubscriber("go_dup_sub")
}

func TestSubscriberEventProperties(t *testing.T) {
	var events []struct {
		uuid      string
		name      string
		eventType EventType
		timestamp string
	}
	var mu sync.Mutex

	RegisterSubscriber("go_evt_props", func(event *Event) {
		mu.Lock()
		events = append(events, struct {
			uuid      string
			name      string
			eventType EventType
			timestamp string
		}{
			uuid:      event.UUID(),
			name:      event.Name(),
			eventType: event.Type(),
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
	if events[0].eventType != EventTypeStart {
		t.Fatalf("expected Start event, got %d", events[0].eventType)
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

	RegisterSubscriber("go_mark_sub", func(event *Event) {
		if event.Type() == EventTypeMark {
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
