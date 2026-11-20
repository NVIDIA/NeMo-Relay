// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func executionContextJSON(context *ApplicationExecutionContext) json.RawMessage {
	if context == nil {
		return nil
	}
	value, err := json.Marshal(context)
	if err != nil {
		panic(err)
	}
	return value
}

func TestToolExecutionContextRemainsSeparateAcrossMiddlewareAndNestedCalls(t *testing.T) {
	const marker = "application-private-context-reference"
	context := json.RawMessage(`{"data":{"approval":{"reference":"` + marker + `"}}}`)
	var eventsMu sync.Mutex
	var events []json.RawMessage
	if err := RegisterSubscriber("execution-context-events", func(event Event) {
		if strings.HasPrefix(event.Name(), "execution-context-") {
			eventsMu.Lock()
			events = append(events, bytes.Clone(event.JSON()))
			eventsMu.Unlock()
		}
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterSubscriber("execution-context-events") })

	var nestedCalls, businessCalls int
	if err := RegisterToolExecutionIntercept("execution-context-first", 1,
		func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			if ctx.ToolName == "execution-context-inner" {
				if ctx.ExecutionContext != nil {
					return ToolExecutionInterceptOutcome{}, fmt.Errorf("nested call inherited execution context")
				}
				return toolExecutionOutcome(next(ctx.Args))
			}
			if !bytes.Equal(executionContextJSON(ctx.ExecutionContext), context) {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("first middleware received wrong context: %v", ctx.ExecutionContext)
			}
			if ctx.ToolCallID == nil || *ctx.ToolCallID != "provider-context-call" {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("context lost the provider tool call ID")
			}
			if _, err := ToolCallExecute("execution-context-inner", json.RawMessage(`{}`),
				func(args json.RawMessage) (ToolExecutionResult, error) {
					nestedCalls++
					return ToolExecutionResult{Result: args}, nil
				}); err != nil {
				return ToolExecutionInterceptOutcome{}, err
			}
			// This is callback-owned JSON, not a mutable reference to the context
			// held by the runtime or supplied to the next middleware.
			ctx.ExecutionContext.Data["approval"][0] = '!'
			return toolExecutionOutcome(next(json.RawMessage(`{"value":2}`)))
		}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("execution-context-first") })
	if err := RegisterToolExecutionIntercept("execution-context-second", 2,
		func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			if ctx.ToolName == "execution-context-outer" && !bytes.Equal(executionContextJSON(ctx.ExecutionContext), context) {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("earlier middleware changed held context: %v", ctx.ExecutionContext)
			}
			return toolExecutionOutcome(next(ctx.Args))
		}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("execution-context-second") })

	result, err := ToolCallExecute("execution-context-outer", json.RawMessage(`{"value":1}`),
		func(args json.RawMessage) (ToolExecutionResult, error) {
			businessCalls++
			if string(args) != `{"value":2}` {
				return ToolExecutionResult{}, fmt.Errorf("unexpected business arguments: %s", args)
			}
			return ToolExecutionResult{Result: args, Annotation: json.RawMessage(`{"reviewed":true}`)}, nil
		}, WithToolExecutionContext(context), WithToolCallID("provider-context-call"))
	if err != nil {
		t.Fatal(err)
	}
	if businessCalls != 1 || nestedCalls != 1 || string(result.Annotation) != `{"reviewed":true}` {
		t.Fatalf("unexpected effect counts or annotation: business=%d nested=%d result=%+v", businessCalls, nestedCalls, result)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatal(err)
	}
	eventsMu.Lock()
	defer eventsMu.Unlock()
	if len(events) == 0 {
		t.Fatal("expected lifecycle events")
	}
	for _, event := range events {
		if bytes.Contains(event, []byte(marker)) || bytes.Contains(event, []byte("execution_context")) {
			t.Fatal("execution context was automatically copied into an event")
		}
	}
}

func TestToolExecutionContextOptionSnapshotAbsenceAndReuse(t *testing.T) {
	var seen []json.RawMessage
	if err := RegisterToolExecutionIntercept("execution-context-snapshot", 1,
		func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			seen = append(seen, bytes.Clone(executionContextJSON(ctx.ExecutionContext)))
			return toolExecutionOutcome(next(ctx.Args))
		}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("execution-context-snapshot") })
	input := json.RawMessage(`{"data":{"approved":true}}`)
	snapshot := WithToolExecutionContext(input)
	input[0] = '!'
	execute := func(options ...ToolCallOption) {
		t.Helper()
		if _, err := ToolCallExecute("execution-context-snapshot", json.RawMessage(`{}`),
			func(args json.RawMessage) (ToolExecutionResult, error) { return ToolExecutionResult{Result: args}, nil }, options...); err != nil {
			t.Fatal(err)
		}
	}
	execute(snapshot)
	execute(snapshot)
	execute()
	execute(WithToolExecutionContext(json.RawMessage(`{}`)))
	execute(snapshot, WithToolExecutionContext(nil))
	execute(WithToolExecutionContext(json.RawMessage(`{"data":{"message":"before\u0000after"}}`)))
	depthBoundary := json.RawMessage(`{"data":` + strings.Repeat(`{"nested":`, 63) + `null` + strings.Repeat(`}`, 63) + `}`)
	execute(WithToolExecutionContext(depthBoundary))
	if len(seen) != 7 || string(seen[0]) != `{"data":{"approved":true}}` || !bytes.Equal(seen[0], seen[1]) || seen[2] != nil ||
		string(seen[3]) != `{}` || seen[4] != nil {
		t.Fatalf("unexpected execution contexts: %q", seen)
	}
	var escaped struct{ Data struct{ Message string } }
	if err := json.Unmarshal(seen[5], &escaped); err != nil || escaped.Data.Message != "before\x00after" {
		t.Fatalf("escaped NUL did not round trip: %q, %v", seen[5], err)
	}
	if !bytes.Equal(seen[6], depthBoundary) {
		t.Fatal("valid context at the 64-container boundary did not round trip")
	}
}

func TestToolExecutionContextRejectsInvalidInputBeforeMiddlewareOrEffects(t *testing.T) {
	var middleware, effects int
	if err := RegisterToolConditionalExecutionGuardrail("execution-context-invalid", 1,
		func(string, json.RawMessage) *string { middleware++; return nil }); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolConditionalExecutionGuardrail("execution-context-invalid") })
	invalid := map[string]json.RawMessage{
		"empty":               {},
		"malformed":           json.RawMessage(`{"broken":`),
		"raw-nul":             json.RawMessage("{}\x00{}"),
		"invalid-utf8":        json.RawMessage("{\"name\":\"\xff\"}"),
		"null":                json.RawMessage(`null`),
		"array":               json.RawMessage(`[]`),
		"approval-null":       json.RawMessage(`{"host_approval":null}`),
		"approval-incomplete": json.RawMessage(`{"host_approval":{"version":1}}`),
		"approval-version":    json.RawMessage(`{"host_approval":{"version":2,"source":"test","approval_id":"a","tool_name":"tool","approved_arguments":{}}}`),
		"scalar":              json.RawMessage(`true`),
		"oversized":           json.RawMessage(`{"data":{"value":"` + strings.Repeat("x", 16*1024) + `"}}`),
		"too-deep":            json.RawMessage(`{"data":` + strings.Repeat(`{"nested":`, 64) + `null` + strings.Repeat(`}`, 64) + `}`),
	}
	invalid["data-null"] = json.RawMessage(`{"data":null}`)
	invalid["data-array"] = json.RawMessage(`{"data":[]}`)
	invalid["data-scalar"] = json.RawMessage(`{"data":1}`)

	for name, context := range invalid {
		t.Run(name, func(t *testing.T) {
			_, err := ToolCallExecute("execution-context-invalid", json.RawMessage(`{}`),
				func(args json.RawMessage) (ToolExecutionResult, error) {
					effects++
					return ToolExecutionResult{Result: args}, nil
				},
				WithToolExecutionContext(context))
			if err == nil {
				t.Fatal("invalid context was accepted")
			}
		})
	}
	if middleware != 0 || effects != 0 {
		t.Fatalf("invalid context reached middleware/effect: %d/%d", middleware, effects)
	}
}

func TestToolExecutionContextRequiresManagedExecution(t *testing.T) {
	for _, context := range []json.RawMessage{nil, json.RawMessage(`{}`)} {
		option := WithToolExecutionContext(context)
		if _, err := ToolCall("execution-context-manual", json.RawMessage(`{}`), option); err == nil {
			t.Fatal("manual start silently accepted execution context")
		}
		handle, err := ToolCall("execution-context-manual", json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		if err := ToolCallEnd(handle, ToolExecutionResult{Result: json.RawMessage(`{}`)}, option); err == nil {
			t.Fatal("manual end silently accepted execution context")
		}
		if err := ToolCallEnd(handle, ToolExecutionResult{Result: json.RawMessage(`{}`)}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestApplicationExecutionContextTypedRoundTripAndValidation(t *testing.T) {
	input := []byte(`{"data":{"host_approval":{"decision":"application-owned","version":2},"custom":[1,null]}}`)
	var context ApplicationExecutionContext
	if err := json.Unmarshal(input, &context); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(context)
	if err != nil {
		t.Fatal(err)
	}
	var before, after any
	if err := json.Unmarshal(input, &before); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &after); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("application data lost: %s", encoded)
	}
	option := WithApplicationExecutionContext(&context)
	context.Data["host_approval"][0] = '!'
	var options toolCallOptions
	option(&options)
	if !json.Valid(options.executionContext) {
		t.Fatal("typed option did not snapshot data")
	}
	invalid := WithApplicationExecutionContext(&ApplicationExecutionContext{Data: map[string]json.RawMessage{"invalid": json.RawMessage(`{`)}})
	var calls int
	if _, err := ToolCallExecute("invalid-typed-context", json.RawMessage(`{}`), func(args json.RawMessage) (ToolExecutionResult, error) {
		calls++
		return ToolExecutionResult{Result: args}, nil
	}, invalid); err == nil || calls != 0 {
		t.Fatal("invalid typed data reached effects")
	}
	for _, source := range []string{`{}`, `{"data":{}}`} {
		var context ApplicationExecutionContext
		if err := json.Unmarshal([]byte(source), &context); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(context)
		if err != nil || string(encoded) != source {
			t.Fatalf("absence/empty distinction lost: %s %v", encoded, err)
		}
	}
	for _, source := range []string{`{"unknown":1}`, `{"data":null}`, `{"data":[]}`} {
		var context ApplicationExecutionContext
		if err := json.Unmarshal([]byte(source), &context); err == nil {
			t.Fatal("invalid typed context accepted")
		}
	}
}

func TestApplicationExecutionContextTypedMiddleware(t *testing.T) {
	context := &ApplicationExecutionContext{Data: map[string]json.RawMessage{
		"host_approval": json.RawMessage(`{"decision":"application-owned","version":2}`),
		"custom":        json.RawMessage(`{"retained":true}`),
	}}
	var observed bool
	if err := RegisterToolExecutionIntercept("typed-application-context", 1, func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
		if ctx.ExecutionContext == nil || string(ctx.ExecutionContext.Data["host_approval"]) != `{"decision":"application-owned","version":2}` || string(ctx.ExecutionContext.Data["custom"]) != `{"retained":true}` {
			return ToolExecutionInterceptOutcome{}, fmt.Errorf("typed context lost application fields")
		}
		observed = true
		return toolExecutionOutcome(next(ctx.Args))
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("typed-application-context") })
	if _, err := ToolCallExecute("typed-context", json.RawMessage(`{"value":1}`), func(args json.RawMessage) (ToolExecutionResult, error) { return ToolExecutionResult{Result: args}, nil }, WithApplicationExecutionContext(context)); err != nil {
		t.Fatal(err)
	}
	if !observed {
		t.Fatal("typed middleware was not called")
	}
}
