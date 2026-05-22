// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package scope_test

import (
	"sync"
	"testing"
	"time"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/scope"
)

const (
	scopeTimestampEventName = "scope-ts"
	scopeTimestampMarkName  = "scope-ts-mark"
)

type capturedScopeEvent struct {
	name      string
	timestamp time.Time
}

func TestScopeShorthands(t *testing.T) {
	var sawMark bool
	var mu sync.Mutex

	if err := nemo_relay.RegisterSubscriber("scope_shortcuts_sub", func(event nemo_relay.Event) {
		if event.Kind() == "mark" && event.Name() == "scope-mark" {
			mu.Lock()
			sawMark = true
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer nemo_relay.DeregisterSubscriber("scope_shortcuts_sub")

	handle, err := scope.Push("scope-shortcuts", nemo_relay.ScopeTypeFunction)
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

func TestScopeShorthandsForwardTimestamps(t *testing.T) {
	var (
		events []capturedScopeEvent
		mu     sync.Mutex
	)
	timestamps := []time.Time{
		time.Date(2026, 1, 2, 0, 0, 0, 123456000, time.UTC),
		time.Date(2026, 1, 2, 0, 0, 1, 223456000, time.UTC),
		time.Date(2026, 1, 2, 0, 0, 2, 323456000, time.UTC),
	}
	subscriberName := "scope_timestamp_sub_" + time.Now().Format("150405.000000")
	if err := nemo_relay.RegisterSubscriber(subscriberName, func(event nemo_relay.Event) {
		if event.Name() != scopeTimestampEventName && event.Name() != scopeTimestampMarkName {
			return
		}
		timestamp, err := time.Parse(time.RFC3339Nano, event.Timestamp())
		if err != nil {
			t.Errorf("failed to parse event timestamp %q: %v", event.Timestamp(), err)
			return
		}
		mu.Lock()
		events = append(events, capturedScopeEvent{name: event.Name(), timestamp: timestamp})
		mu.Unlock()
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer nemo_relay.DeregisterSubscriber(subscriberName)

	handle, err := scope.Push(scopeTimestampEventName, nemo_relay.ScopeTypeFunction, nemo_relay.WithScopeTimestamp(timestamps[0]))
	if err != nil {
		t.Fatalf("Push failed: %v", err)
	}
	if err := scope.Event(scopeTimestampMarkName, nemo_relay.WithEventTimestamp(timestamps[1])); err != nil {
		t.Fatalf("Event failed: %v", err)
	}
	if err := scope.Pop(handle, nemo_relay.WithScopeEndTimestamp(timestamps[2])); err != nil {
		t.Fatalf("Pop failed: %v", err)
	}

	expected := []capturedScopeEvent{
		{name: scopeTimestampEventName, timestamp: timestamps[0]},
		{name: scopeTimestampMarkName, timestamp: timestamps[1]},
		{name: scopeTimestampEventName, timestamp: timestamps[2]},
	}
	mu.Lock()
	defer mu.Unlock()
	assertCapturedScopeEvents(t, events, expected)
}

func assertCapturedScopeEvents(t *testing.T, events, expected []capturedScopeEvent) {
	t.Helper()

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d: %#v", len(expected), len(events), events)
	}
	for i, event := range events {
		if event.name != expected[i].name || !event.timestamp.Equal(expected[i].timestamp) {
			t.Fatalf("event %d = %#v, expected %#v", i, event, expected[i])
		}
	}
}
