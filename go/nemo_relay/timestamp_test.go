// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

type capturedTimestampEvent struct {
	name      string
	timestamp time.Time
}

func TestManualLifecycleTimestamps(t *testing.T) {
	var (
		events []capturedTimestampEvent
		mu     sync.Mutex
	)
	timestamps := []time.Time{
		time.Date(2026, 1, 1, 0, 0, 0, 123456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 1, 223456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 2, 323456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 3, 423456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 4, 523456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 5, 623456000, time.UTC),
		time.Date(2026, 1, 1, 0, 0, 6, 723456000, time.UTC),
	}
	subscriberName := "go_timestamp_sub_" + time.Now().Format("150405.000000")
	if err := RegisterSubscriber(subscriberName, func(event Event) {
		if !strings.HasPrefix(event.Name(), "go_ts_") {
			return
		}
		timestamp, err := time.Parse(time.RFC3339Nano, event.Timestamp())
		if err != nil {
			t.Errorf("failed to parse event timestamp %q: %v", event.Timestamp(), err)
			return
		}
		mu.Lock()
		events = append(events, capturedTimestampEvent{name: event.Name(), timestamp: timestamp})
		mu.Unlock()
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer DeregisterSubscriber(subscriberName)

	scopeHandle, err := PushScope("go_ts_scope", ScopeTypeAgent, WithScopeTimestamp(timestamps[0]))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := EmitEvent("go_ts_mark", WithEventParent(scopeHandle), WithEventTimestamp(timestamps[1])); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	toolHandle, err := ToolCall("go_ts_tool", json.RawMessage(`{"x":1}`), WithToolTimestamp(timestamps[2]))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if err := ToolCallEnd(toolHandle, json.RawMessage(`{"ok":true}`), WithToolTimestamp(timestamps[3])); err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}
	llmHandle, err := LlmCall("go_ts_llm", makeRequest(), WithLLMTimestamp(timestamps[4]))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if err := LlmCallEnd(llmHandle, json.RawMessage(`{"ok":true}`), WithLLMTimestamp(timestamps[5])); err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}
	if err := PopScope(scopeHandle, WithScopeEndTimestamp(timestamps[6])); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

	expected := []capturedTimestampEvent{
		{name: "go_ts_scope", timestamp: timestamps[0]},
		{name: "go_ts_mark", timestamp: timestamps[1]},
		{name: "go_ts_tool", timestamp: timestamps[2]},
		{name: "go_ts_tool", timestamp: timestamps[3]},
		{name: "go_ts_llm", timestamp: timestamps[4]},
		{name: "go_ts_llm", timestamp: timestamps[5]},
		{name: "go_ts_scope", timestamp: timestamps[6]},
	}
	mu.Lock()
	defer mu.Unlock()
	assertCapturedTimestampEvents(t, events, expected)
}

func assertCapturedTimestampEvents(t *testing.T, events, expected []capturedTimestampEvent) {
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

func TestManualLifecycleTimestampNormalizesToUnixMicroseconds(t *testing.T) {
	var (
		observed time.Time
		seen     bool
		mu       sync.Mutex
	)
	source := time.Date(2026, 1, 1, 12, 0, 0, 123456789, time.FixedZone("test-offset", -5*60*60))
	expected := time.UnixMicro(source.UTC().UnixMicro()).UTC()
	subscriberName := "go_timestamp_normalize_sub_" + time.Now().Format("150405.000000")

	if err := RegisterSubscriber(subscriberName, func(event Event) {
		if event.Name() != "go_ts_normalized" {
			return
		}
		timestamp, err := time.Parse(time.RFC3339Nano, event.Timestamp())
		if err != nil {
			t.Errorf("failed to parse event timestamp %q: %v", event.Timestamp(), err)
			return
		}
		mu.Lock()
		observed = timestamp.UTC()
		seen = true
		mu.Unlock()
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer DeregisterSubscriber(subscriberName)

	if err := EmitEvent("go_ts_normalized", WithEventTimestamp(source)); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !seen {
		t.Fatal("expected go_ts_normalized event")
	}
	if !observed.Equal(expected) {
		t.Fatalf("timestamp = %s, expected %s", observed.Format(time.RFC3339Nano), expected.Format(time.RFC3339Nano))
	}
}
