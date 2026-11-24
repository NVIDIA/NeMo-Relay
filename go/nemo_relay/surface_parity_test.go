// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestNativeStatusErrors(t *testing.T) {
	const name = "status-error-subscriber"
	if err := RegisterSubscriber(name, func(Event) {}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterSubscriber(name)
	err := RegisterSubscriber(name, func(Event) {})
	wrapped := fmt.Errorf("register: %w", err)
	var native *RelayError
	if !errors.Is(wrapped, ErrAlreadyExists) || !errors.As(wrapped, &native) || native.Code != StatusAlreadyExists || native.Message == "" {
		t.Fatalf("duplicate registration lost status or message: %v", wrapped)
	}
	if errors.Is(wrapped, ErrInvalidJSON) {
		t.Fatal("unrelated status matched")
	}
	if !errors.Is(&RelayError{Code: StatusCancelled}, context.Canceled) {
		t.Fatal("native cancellation must unwrap to context.Canceled")
	}
	codec := NewOpenAIChatCodec()
	codec.Close()
	_, pointerErr := codec.Decode(LLMRequest{})
	var pointerNative *RelayError
	if !errors.Is(pointerErr, ErrUnknownStatus) || !errors.As(pointerErr, &pointerNative) || pointerNative.Message == "" {
		t.Fatalf("pointer error lost unknown status or message: %v", pointerErr)
	}
	runWithTestScopeStack(t, func() {
		const guard = "status-error-guard"
		message := "blocked by status test"
		if err := RegisterLlmConditionalExecutionGuardrail(guard, 1, func(_, _ json.RawMessage) *string { return &message }); err != nil {
			t.Fatal(err)
		}
		defer DeregisterLlmConditionalExecutionGuardrail(guard)
		_, err := LlmCallExecute("status-blocked", requestForTest(makeRequest()), func(json.RawMessage) (json.RawMessage, error) {
			t.Error("rejected call executed")
			return json.RawMessage(`{}`), nil
		})
		if !errors.Is(err, ErrGuardrailRejected) {
			t.Fatalf("expected rejection status, got %v", err)
		}
	})
}

func TestExplicitEventCategoryProfile(t *testing.T) {
	var mu sync.Mutex
	var events []Event
	const name = "explicit-category-profile"
	if err := RegisterSubscriber(name, func(e Event) {
		if e.Name() == name {
			mu.Lock()
			events = append(events, e)
			mu.Unlock()
		}
	}); err != nil {
		t.Fatal(err)
	}
	defer DeregisterSubscriber(name)
	profile := json.RawMessage(`{"model_name":"custom-model","subtype":"producer.mark","producer_field":{"value":42}}`)
	runWithTestScopeStack(t, func() {
		if err := EmitEvent(name, WithEventCategory("producer.category"), WithEventCategoryProfile(profile)); err != nil {
			t.Fatal(err)
		}
		err := EmitEvent(name, WithEventCategoryProfile(json.RawMessage(`{"model_name":123}`)))
		if !errors.Is(err, ErrInvalidJSON) {
			t.Fatalf("invalid typed profile: %v", err)
		}
	})
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(events) != 1 {
		t.Fatalf("expected one valid event, got %d", len(events))
	}
	if events[0].Category() != "producer.category" {
		t.Fatalf("category: %s", events[0].Category())
	}
	assertSemanticJSONEqual(t, events[0].CategoryProfile(), profile)
}

func TestAtifCompleteConstructorOptions(t *testing.T) {
	exporter, err := NewAtifExporter("session-options", "options-agent", "1", "model-options",
		WithAtifTrajectoryID("independent-trajectory"),
		WithAtifToolDefinitions([]json.RawMessage{json.RawMessage(`{"name":"search","description":"Find things"}`)}),
		WithAtifAgentExtra(json.RawMessage(`{"environment":"test"}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer exporter.Close()
	payload, err := exporter.ExportJSON()
	if err != nil {
		t.Fatal(err)
	}
	var trajectory struct {
		Session string `json:"session_id"`
		ID      string `json:"trajectory_id"`
		Agent   struct {
			Tools []json.RawMessage `json:"tool_definitions"`
			Extra json.RawMessage   `json:"extra"`
		} `json:"agent"`
	}
	if err := json.Unmarshal(payload, &trajectory); err != nil {
		t.Fatal(err)
	}
	if trajectory.Session != "session-options" || trajectory.ID != "independent-trajectory" || len(trajectory.Agent.Tools) != 1 {
		t.Fatalf("unexpected trajectory: %s", payload)
	}
	assertSemanticJSONEqual(t, trajectory.Agent.Tools[0], json.RawMessage(`{"name":"search","description":"Find things"}`))
	assertSemanticJSONEqual(t, trajectory.Agent.Extra, json.RawMessage(`{"environment":"test"}`))
	if handle, err := NewAtifExporter("s", "a", "1", "", WithAtifAgentExtra(json.RawMessage(`{`))); !errors.Is(err, ErrInvalidJSON) || handle != nil {
		t.Fatalf("invalid extra: handle=%v err=%v", handle, err)
	}
	if handle, err := NewAtifExporter("s", "a", "1", "", WithAtifToolDefinitions([]json.RawMessage{json.RawMessage(`{`)})); err == nil || handle != nil {
		t.Fatalf("invalid definition: handle=%v err=%v", handle, err)
	}
}
