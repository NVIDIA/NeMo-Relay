// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
	"time"
)

func TestHandleInspectionPreservesLifecycleFields(t *testing.T) {
	runWithTestScopeStack(t, func() {
		timestamp := time.Date(2026, 1, 2, 3, 4, 5, 123456000, time.UTC)
		data := json.RawMessage(`{"payload":42}`)
		metadata := json.RawMessage(`{"source":"inspection"}`)
		scope, err := PushScope("inspection-scope", ScopeTypeAgent, WithScopeTimestamp(timestamp))
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := PopScope(scope); err != nil {
				t.Error(err)
			}
		}()
		assertHandleStartTime(t, scope.StartedAt(), timestamp)
		tool, err := ToolCall("inspection-tool", json.RawMessage(`{}`), WithToolData(data), WithToolMetadata(metadata), WithToolCallID("call-123"), WithToolTimestamp(timestamp))
		if err != nil {
			t.Fatal(err)
		}
		assertHandleStartTime(t, tool.StartedAt(), timestamp)
		assertSemanticJSONEqual(t, data, tool.Data())
		assertSemanticJSONEqual(t, metadata, tool.Metadata())
		if tool.ToolCallID() != "call-123" {
			t.Fatalf("tool call ID: %q", tool.ToolCallID())
		}
		if err := ToolCallEnd(tool, ToolExecutionResult{}); err != nil {
			t.Fatal(err)
		}
		llm, err := LlmCall("inspection-llm", requestForTest(makeRequest()), WithLLMData(data), WithLLMMetadata(metadata), WithLLMModelName("inspection-model"), WithLLMTimestamp(timestamp))
		if err != nil {
			t.Fatal(err)
		}
		assertHandleStartTime(t, llm.StartedAt(), timestamp)
		assertSemanticJSONEqual(t, data, llm.Data())
		assertSemanticJSONEqual(t, metadata, llm.Metadata())
		if llm.ModelName() != "inspection-model" {
			t.Fatalf("model: %q", llm.ModelName())
		}
		if err := LlmCallEnd(llm, json.RawMessage(`{}`)); err != nil {
			t.Fatal(err)
		}
		tool, err = ToolCall("inspection-empty-tool", json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		if tool.Data() != nil || tool.Metadata() != nil || tool.ToolCallID() != "" {
			t.Fatal("absent tool fields must stay absent")
		}
		if err := ToolCallEnd(tool, ToolExecutionResult{}); err != nil {
			t.Fatal(err)
		}
	})
}

func assertHandleStartTime(t *testing.T, value string, expected time.Time) {
	t.Helper()
	actual, err := time.Parse(time.RFC3339Nano, value)
	if err != nil || !actual.Equal(expected) {
		t.Fatalf("start time %q, expected %v (parse error: %v)", value, expected, err)
	}
}
