// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"sync"
	"testing"
)

const (
	duplicateEventSanitizer = "go-event-duplicate"
	duplicateToolSanitizer  = "go-tool-duplicate"
	invalidScopeUUID        = "not-a-uuid"
)

func registeredClosureCount() int {
	closureRegistryMu.Lock()
	defer closureRegistryMu.Unlock()
	return len(closureRegistry)
}

func TestEventSanitizerRegistries(t *testing.T) {
	runTestWithScopeStack(t, testEventSanitizerRegistries)
}

func TestEventSanitizerMarshalFailureClearsObservabilityFields(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		var mu sync.Mutex
		var events []Event
		registerEventSanitizerSubscriber(t, &mu, &events)

		if err := RegisterMarkSanitizeGuardrail("go-mark-sanitize-invalid", 0, func(_ Event, _ EventSanitizeFields) EventSanitizeFields {
			return EventSanitizeFields{Data: json.RawMessage("{")}
		}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = DeregisterMarkSanitizeGuardrail("go-mark-sanitize-invalid") })

		if err := EmitEvent("invalid-sanitizer", WithEventData(json.RawMessage(`{"secret":true}`)), WithEventMetadata(json.RawMessage(`{"secret":true}`))); err != nil {
			t.Fatal(err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}

		mu.Lock()
		defer mu.Unlock()
		if len(events) != 1 {
			t.Fatalf("expected one event, got %d", len(events))
		}
		if len(events[0].Data()) != 0 || len(events[0].CategoryProfile()) != 0 || len(events[0].Metadata()) != 0 {
			t.Fatalf("expected cleared observability fields, got data=%s category_profile=%s metadata=%s", events[0].Data(), events[0].CategoryProfile(), events[0].Metadata())
		}
	})
}

func testEventSanitizerRegistries(t *testing.T) {
	var mu sync.Mutex
	var events []Event
	registerEventSanitizerSubscriber(t, &mu, &events)
	registerEventSanitizerGuardrails(t)
	emitSanitizerTestEvents(t)
	assertSanitizedTestEvents(t, &mu, events)
}

func registerEventSanitizerSubscriber(t *testing.T, mu *sync.Mutex, events *[]Event) {
	t.Helper()
	if err := RegisterSubscriber("go-event-sanitize-sub", func(event Event) {
		mu.Lock()
		*events = append(*events, event)
		mu.Unlock()
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterSubscriber("go-event-sanitize-sub") })
}

func registerEventSanitizerGuardrails(t *testing.T) {
	t.Helper()
	if err := RegisterMarkSanitizeGuardrail("go-mark-sanitize", 0, func(event Event, fields EventSanitizeFields) EventSanitizeFields {
		if event.Name() != "checkpoint" {
			t.Fatalf("unexpected event context: %s", event.Name())
		}
		fields.Data = json.RawMessage(`{"safe":true}`)
		fields.CategoryProfile = json.RawMessage(`{"subtype":"go.sanitized"}`)
		fields.Metadata = json.RawMessage("null")
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterMarkSanitizeGuardrail("go-mark-sanitize") })
	if err := RegisterScopeSanitizeStartGuardrail("go-scope-start", 0, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Metadata = json.RawMessage(`{"phase":"start"}`)
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterScopeSanitizeStartGuardrail("go-scope-start") })
	if err := RegisterScopeSanitizeEndGuardrail("go-scope-end", 0, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Metadata = json.RawMessage(`{"phase":"end"}`)
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterScopeSanitizeEndGuardrail("go-scope-end") })
}

func emitSanitizerTestEvents(t *testing.T) {
	t.Helper()
	handle, err := PushScope("generic", ScopeTypeCustom)
	if err != nil {
		t.Fatal(err)
	}
	if err := PopScope(handle); err != nil {
		t.Fatal(err)
	}

	if err := EmitEvent("checkpoint", WithEventData(json.RawMessage(`{"secret":true}`)), WithEventMetadata(json.RawMessage(`{"secret":true}`))); err != nil {
		t.Fatal(err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}
}

func assertSanitizedTestEvents(t *testing.T, mu *sync.Mutex, events []Event) {
	t.Helper()
	mu.Lock()
	mark := events[len(events)-1]
	mu.Unlock()
	if string(mark.Data()) != `{"safe":true}` || string(mark.CategoryProfile()) != `{"subtype":"go.sanitized"}` || len(mark.Metadata()) != 0 {
		t.Fatalf("unexpected sanitized fields: data=%s category_profile=%s metadata=%s", mark.Data(), mark.CategoryProfile(), mark.Metadata())
	}
	var phases []string
	for _, event := range events {
		if event.Name() == "generic" {
			phases = append(phases, string(event.Metadata()))
		}
	}
	if len(phases) != 2 || phases[0] != `{"phase":"start"}` || phases[1] != `{"phase":"end"}` {
		t.Fatalf("unexpected scope sanitizer phases: %v", phases)
	}
}

func TestScopeLocalEventSanitizerInheritanceAndCleanup(t *testing.T) {
	runTestWithScopeStack(t, testScopeLocalEventSanitizerInheritanceAndCleanup)
}

func TestScopeLocalEventSanitizersCanBeDeregistered(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		owner, err := PushScope("deregister-owner", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer func() {
			if err := PopScope(owner); err != nil {
				t.Fatalf("PopScope failed: %v", err)
			}
		}()

		passThrough := func(_ Event, fields EventSanitizeFields) EventSanitizeFields { return fields }
		for _, sanitizer := range []struct {
			name       string
			register   func() error
			deregister func() error
		}{
			{
				name: "mark",
				register: func() error {
					return ScopeRegisterMarkSanitizeGuardrail(owner.UUID(), "deregister-mark", 0, passThrough)
				},
				deregister: func() error { return ScopeDeregisterMarkSanitizeGuardrail(owner.UUID(), "deregister-mark") },
			},
			{
				name: "scope start",
				register: func() error {
					return ScopeRegisterScopeSanitizeStartGuardrail(owner.UUID(), "deregister-start", 0, passThrough)
				},
				deregister: func() error { return ScopeDeregisterScopeSanitizeStartGuardrail(owner.UUID(), "deregister-start") },
			},
			{
				name: "scope end",
				register: func() error {
					return ScopeRegisterScopeSanitizeEndGuardrail(owner.UUID(), "deregister-end", 0, passThrough)
				},
				deregister: func() error { return ScopeDeregisterScopeSanitizeEndGuardrail(owner.UUID(), "deregister-end") },
			},
		} {
			if err := sanitizer.register(); err != nil {
				t.Fatalf("register %s sanitizer: %v", sanitizer.name, err)
			}
			if err := sanitizer.deregister(); err != nil {
				t.Fatalf("deregister %s sanitizer: %v", sanitizer.name, err)
			}
		}
	})
}

func testScopeLocalEventSanitizerInheritanceAndCleanup(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]json.RawMessage{}
	if err := RegisterSubscriber("go-local-event-sub", func(event Event) {
		mu.Lock()
		seen[event.Name()+":"+event.ScopeCategory()] = append(json.RawMessage(nil), event.Data()...)
		mu.Unlock()
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterSubscriber("go-local-event-sub")

	owner, err := PushScope("owner", ScopeTypeAgent)
	if err != nil {
		t.Fatal(err)
	}
	if err := ScopeRegisterMarkSanitizeGuardrail(owner.UUID(), "go-local-mark", 0, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Data = json.RawMessage(`{"local":true}`)
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	if err := ScopeRegisterScopeSanitizeStartGuardrail(owner.UUID(), "go-local-start", 0, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Data = json.RawMessage(`{"local_phase":"start"}`)
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	if err := ScopeRegisterScopeSanitizeEndGuardrail(owner.UUID(), "go-local-end", 0, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Data = json.RawMessage(`{"local_phase":"end"}`)
		return fields
	}); err != nil {
		t.Fatal(err)
	}
	if err := EmitEvent("inside", WithEventData(json.RawMessage(`{"raw":true}`))); err != nil {
		t.Fatal(err)
	}
	child, err := PushScope("child", ScopeTypeFunction)
	if err != nil {
		t.Fatal(err)
	}
	if err := EmitEvent("inherited", WithEventData(json.RawMessage(`{"raw":true}`))); err != nil {
		t.Fatal(err)
	}
	if err := PopScope(child); err != nil {
		t.Fatal(err)
	}
	if err := PopScope(owner); err != nil {
		t.Fatal(err)
	}
	if err := EmitEvent("outside", WithEventData(json.RawMessage(`{"raw":true}`))); err != nil {
		t.Fatal(err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}
	if string(seen["inside:"]) != `{"local":true}` ||
		string(seen["inherited:"]) != `{"local":true}` ||
		string(seen["outside:"]) != `{"raw":true}` ||
		string(seen["child:start"]) != `{"local_phase":"start"}` ||
		string(seen["child:end"]) != `{"local_phase":"end"}` {
		t.Fatalf("unexpected scope-local results: %#v", seen)
	}
}

func TestEventSanitizerRegistrationErrorsReleaseCallbacks(t *testing.T) {
	baseline := registeredClosureCount()
	passThrough := func(_ Event, fields EventSanitizeFields) EventSanitizeFields { return fields }

	if err := RegisterMarkSanitizeGuardrail(duplicateEventSanitizer, 0, passThrough); err != nil {
		t.Fatal(err)
	}
	err := RegisterMarkSanitizeGuardrail(duplicateEventSanitizer, 0, passThrough)
	if err == nil {
		t.Fatal("expected duplicate event sanitizer registration to fail")
	}
	if afterDuplicate := registeredClosureCount(); afterDuplicate != baseline+1 {
		t.Fatalf("duplicate registration leaked callback: baseline=%d current=%d", baseline, afterDuplicate)
	}
	if err := DeregisterMarkSanitizeGuardrail(duplicateEventSanitizer); err != nil {
		t.Fatal(err)
	}
	if err := RegisterToolSanitizeRequestGuardrail(duplicateToolSanitizer, 0, func(_ string, args json.RawMessage) json.RawMessage { return args }); err != nil {
		t.Fatal(err)
	}
	err = RegisterToolSanitizeRequestGuardrail(duplicateToolSanitizer, 0, func(_ string, args json.RawMessage) json.RawMessage { return args })
	if err == nil {
		t.Fatal("expected duplicate tool sanitizer registration to fail")
	}
	if afterToolDuplicate := registeredClosureCount(); afterToolDuplicate != baseline+1 {
		t.Fatalf("duplicate tool registration leaked callback: baseline=%d current=%d", baseline, afterToolDuplicate)
	}
	if err := DeregisterToolSanitizeRequestGuardrail(duplicateToolSanitizer); err != nil {
		t.Fatal(err)
	}

	for name, register := range map[string]func() error{
		"mark": func() error {
			return ScopeRegisterMarkSanitizeGuardrail(invalidScopeUUID, "go-invalid-mark", 0, passThrough)
		},
		"scope start": func() error {
			return ScopeRegisterScopeSanitizeStartGuardrail(invalidScopeUUID, "go-invalid-start", 0, passThrough)
		},
		"scope end": func() error {
			return ScopeRegisterScopeSanitizeEndGuardrail(invalidScopeUUID, "go-invalid-end", 0, passThrough)
		},
	} {
		err = register()
		if err == nil {
			t.Fatalf("expected invalid UUID for %s registration", name)
		}
	}
	if afterErrors := registeredClosureCount(); afterErrors != baseline {
		t.Fatalf("failed registration leaked callbacks: baseline=%d current=%d", baseline, afterErrors)
	}
}
