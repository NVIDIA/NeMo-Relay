// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
)

const (
	eventMetadataFirstInjector       = "go-event-metadata-first"
	eventMetadataLaterInjector       = "go-event-metadata-later"
	eventMetadataFailureInjector     = "go-event-metadata-failure"
	eventMetadataMixedValuesInjector = "go-event-metadata-mixed-values"
	eventMetadataDuplicateInjector   = "go-event-metadata-duplicate"
	injectedCollisionMetadataKey     = "go.injected.collision"
	injectedLocalMetadataKey         = "go.injected.local"
)

func TestEventMetadataInjectorGlobalScopeLocalAndFailureBehavior(t *testing.T) {
	runTestInIsolatedWorkingDirectory(t, func(t *testing.T) {
		runTestWithScopeStack(t, testEventMetadataInjectorGlobalScopeLocalAndFailureBehavior)
	})
}

func testEventMetadataInjectorGlobalScopeLocalAndFailureBehavior(t *testing.T) {
	var mu sync.Mutex
	var events []Event
	if err := RegisterSubscriber("go-event-metadata-subscriber", func(event Event) {
		mu.Lock()
		events = append(events, event)
		mu.Unlock()
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterSubscriber("go-event-metadata-subscriber") })

	if err := RegisterEventMetadataInjector(eventMetadataFirstInjector, 10, func(event Event) (EventMetadata, error) {
		return EventMetadata{
			"go.injected.global":         event.Kind(),
			injectedCollisionMetadataKey: "first",
			"go.existing":                "replacement",
			"go.injected.integers":       []int64{1, 2},
			"go.injected.doubles":        []float64{1, 2.5},
		}, nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterEventMetadataInjector(eventMetadataFirstInjector) })

	if err := RegisterEventMetadataInjector(eventMetadataLaterInjector, 20, func(Event) (EventMetadata, error) {
		return EventMetadata{injectedCollisionMetadataKey: "later"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterEventMetadataInjector(eventMetadataLaterInjector) })

	if err := RegisterEventMetadataInjector(eventMetadataFailureInjector, 30, func(Event) (EventMetadata, error) {
		return nil, errors.New("expected Go injector failure")
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterEventMetadataInjector(eventMetadataFailureInjector) })

	if err := RegisterEventMetadataInjector(eventMetadataMixedValuesInjector, 40, func(Event) (EventMetadata, error) {
		return EventMetadata{
			"go.invalid.mixed_values": []any{1, "two"},
			"go.invalid.sentinel":     "must-be-omitted",
		}, nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterEventMetadataInjector(eventMetadataMixedValuesInjector) })

	scope, err := PushScope("go-event-metadata-scope", ScopeTypeCustom)
	if err != nil {
		t.Fatal(err)
	}
	if err := ScopeRegisterEventMetadataInjector(scope.UUID(), "go-event-metadata-local", 5, func(Event) (EventMetadata, error) {
		return EventMetadata{injectedLocalMetadataKey: true}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := EmitEvent(
		"go-event-metadata-mark",
		WithEventMetadata(json.RawMessage(`{"go.existing":"original"}`)),
	); err != nil {
		t.Fatal(err)
	}
	if err := PopScope(scope); err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{
		eventMetadataMixedValuesInjector,
		eventMetadataFailureInjector,
		eventMetadataLaterInjector,
		eventMetadataFirstInjector,
	} {
		if err := DeregisterEventMetadataInjector(name); err != nil {
			t.Fatal(err)
		}
	}
	if err := EmitEvent("go-event-metadata-cleanup"); err != nil {
		t.Fatal(err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	assertEventMetadataInjectorEvents(t, events)
}

func assertEventMetadataInjectorEvents(t *testing.T, events []Event) {
	t.Helper()
	if len(events) != 4 {
		t.Fatalf("expected four delivered events, got %d", len(events))
	}
	for index, event := range events[:3] {
		metadata := decodeEventMetadata(t, event)
		if metadata[injectedCollisionMetadataKey] != "first" {
			t.Fatalf("event %d did not preserve first-injector precedence: %#v", index, metadata)
		}
		if _, ok := metadata["go.injected.global"]; !ok {
			t.Fatalf("event %d is missing global metadata: %#v", index, metadata)
		}
	}
	if metadata := decodeEventMetadata(t, events[0]); metadata[injectedLocalMetadataKey] != nil {
		t.Fatalf("scope start unexpectedly contains scope-local metadata: %#v", metadata)
	}
	for _, event := range events[1:3] {
		if metadata := decodeEventMetadata(t, event); metadata[injectedLocalMetadataKey] != true {
			t.Fatalf("event %s is missing scope-local metadata: %#v", event.Name(), metadata)
		}
	}
	if metadata := decodeEventMetadata(t, events[1]); metadata["go.existing"] != "original" {
		t.Fatalf("existing metadata was overwritten: %#v", metadata)
	}
	metadata := decodeEventMetadata(t, events[1])
	if got, want := metadata["go.injected.integers"], []any{float64(1), float64(2)}; !reflect.DeepEqual(got, want) {
		t.Fatalf("homogeneous integer metadata = %#v, want %#v", got, want)
	}
	if got, want := metadata["go.injected.doubles"], []any{float64(1), 2.5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("homogeneous double metadata = %#v, want %#v", got, want)
	}
	if _, ok := metadata["go.invalid.mixed_values"]; ok {
		t.Fatalf("mixed primitive metadata was accepted: %#v", metadata)
	}
	if _, ok := metadata["go.invalid.sentinel"]; ok {
		t.Fatalf("invalid callback output was partially applied: %#v", metadata)
	}
	if metadata := decodeEventMetadata(t, events[3]); len(metadata) != 0 {
		t.Fatalf("cleanup event retained injector metadata: %#v", metadata)
	}
}

func TestPluginContextEventMetadataInjectorLifecycle(t *testing.T) {
	runTestInIsolatedWorkingDirectory(t, func(t *testing.T) {
		runTestWithScopeStack(t, testPluginContextEventMetadataInjectorLifecycle)
	})
}

func testPluginContextEventMetadataInjectorLifecycle(t *testing.T) {
	const kind = "go.event.metadata.plugin"
	var mu sync.Mutex
	var events []Event
	if err := RegisterSubscriber("go-plugin-metadata-subscriber", func(event Event) {
		mu.Lock()
		events = append(events, event)
		mu.Unlock()
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterSubscriber("go-plugin-metadata-subscriber")
	if err := RegisterPlugin(kind, PluginFuncs{RegisterFunc: func(_ map[string]any, ctx *PluginContext) error {
		return ctx.RegisterEventMetadataInjector("configured", 10, func(Event) (EventMetadata, error) {
			return EventMetadata{"go.injected.plugin": true}, nil
		})
	}}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterPlugin(kind)
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{{Kind: kind, Enabled: true}}}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"go-plugin-metadata-active", "go-plugin-metadata-cleanup"} {
		if name == "go-plugin-metadata-cleanup" {
			if err := ClearPluginConfiguration(); err != nil {
				t.Fatal(err)
			}
		}
		if err := EmitEvent(name); err != nil {
			t.Fatal(err)
		}
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(events) != 2 {
		t.Fatalf("expected two delivered events, got %d", len(events))
	}
	if metadata := decodeEventMetadata(t, events[0]); metadata["go.injected.plugin"] != true {
		t.Fatalf("plugin metadata was not injected: %#v", metadata)
	}
	if metadata := decodeEventMetadata(t, events[1]); len(metadata) != 0 {
		t.Fatalf("plugin metadata remained after cleanup: %#v", metadata)
	}
}

func TestEventMetadataInjectorRegistrationErrorsReleaseCallbacks(t *testing.T) {
	baseline := registeredClosureCount()
	callback := func(Event) (EventMetadata, error) { return nil, nil }
	if err := RegisterEventMetadataInjector(eventMetadataDuplicateInjector, 0, callback); err != nil {
		t.Fatal(err)
	}
	if RegisterEventMetadataInjector(eventMetadataDuplicateInjector, 0, callback) == nil {
		t.Fatal("expected duplicate event metadata injector registration to fail")
	}
	current := registeredClosureCount()
	if current != baseline+1 {
		t.Fatalf("duplicate registration leaked callback: baseline=%d current=%d", baseline, current)
	}
	if err := DeregisterEventMetadataInjector(eventMetadataDuplicateInjector); err != nil {
		t.Fatal(err)
	}
}

func TestEventMetadataInjectorNilCallbacksDoNotRegister(t *testing.T) {
	runTestInIsolatedWorkingDirectory(t, func(t *testing.T) {
		runTestWithScopeStack(t, testEventMetadataInjectorNilCallbacksDoNotRegister)
	})
}

func testEventMetadataInjectorNilCallbacksDoNotRegister(t *testing.T) {
	baseline := registeredClosureCount()
	assertNilGlobalEventMetadataInjector(t, baseline)
	assertNilScopeEventMetadataInjector(t, baseline)
	assertNilPluginEventMetadataInjector(t)
}

func assertNilGlobalEventMetadataInjector(t *testing.T, baseline int) {
	t.Helper()
	if err := RegisterEventMetadataInjector("go-event-metadata-nil-global", 0, nil); !errors.Is(err, errEventMetadataInjectorCallbackNil) {
		t.Fatalf("RegisterEventMetadataInjector() error = %v, want %v", err, errEventMetadataInjectorCallbackNil)
	}
	current := registeredClosureCount()
	if current != baseline {
		t.Fatalf("nil global callback changed registry size: baseline=%d current=%d", baseline, current)
	}
}

func assertNilScopeEventMetadataInjector(t *testing.T, baseline int) {
	t.Helper()
	scope, err := PushScope("go-event-metadata-nil-scope", ScopeTypeCustom)
	if err != nil {
		t.Fatal(err)
	}
	if err := ScopeRegisterEventMetadataInjector(scope.UUID(), "go-event-metadata-nil-local", 0, nil); !errors.Is(err, errEventMetadataInjectorCallbackNil) {
		t.Fatalf("ScopeRegisterEventMetadataInjector() error = %v, want %v", err, errEventMetadataInjectorCallbackNil)
	}
	current := registeredClosureCount()
	if current != baseline {
		t.Fatalf("nil scope callback changed registry size: baseline=%d current=%d", baseline, current)
	}
	if err := PopScope(scope); err != nil {
		t.Fatal(err)
	}
}

func assertNilPluginEventMetadataInjector(t *testing.T) {
	t.Helper()
	const kind = "go.event.metadata.nil.plugin"
	if err := RegisterPlugin(kind, PluginFuncs{RegisterFunc: func(_ map[string]any, ctx *PluginContext) error {
		return ctx.RegisterEventMetadataInjector("nil", 0, nil)
	}}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterPlugin(kind) })
	baseline := registeredClosureCount()
	if _, err := InitializePlugins(PluginConfig{Version: 1, Components: []PluginComponentSpec{{Kind: kind, Enabled: true}}}); err == nil {
		t.Fatal("expected nil plugin callback registration to fail")
	}
	current := registeredClosureCount()
	if current != baseline {
		t.Fatalf("nil plugin callback changed registry size: baseline=%d current=%d", baseline, current)
	}
}

func decodeEventMetadata(t *testing.T, event Event) map[string]any {
	t.Helper()
	metadata := map[string]any{}
	if len(event.Metadata()) == 0 {
		return metadata
	}
	if err := json.Unmarshal(event.Metadata(), &metadata); err != nil {
		t.Fatalf("decode event metadata: %v", err)
	}
	return metadata
}
