// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

func TestToolExecutionContextRemainsSeparateAcrossMiddlewareAndSubsequentCalls(t *testing.T) {
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

	var subsequentCalls, businessCalls int
	var outerInterceptOrder []string
	if err := RegisterToolExecutionIntercept("execution-context-first", 1,
		func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			if ctx.ToolName == "execution-context-subsequent" {
				if ctx.ExecutionContext != nil {
					return ToolExecutionInterceptOutcome{}, fmt.Errorf("subsequent call inherited execution context")
				}
				return toolExecutionOutcome(next(ctx.Args))
			}
			if !bytes.Equal(executionContextJSON(ctx.ExecutionContext), context) {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("first middleware received wrong context: %v", ctx.ExecutionContext)
			}
			if ctx.ToolCallID == nil || *ctx.ToolCallID != "provider-context-call" {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("context lost the provider tool call ID")
			}
			// This is callback-owned JSON, not a mutable reference to the context
			// held by the runtime or supplied to the next middleware.
			ctx.ExecutionContext.Data["approval"][0] = '!'
			outerInterceptOrder = append(outerInterceptOrder, "first-mutated")
			return toolExecutionOutcome(next(json.RawMessage(`{"value":2}`)))
		}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("execution-context-first") })
	if err := RegisterToolExecutionIntercept("execution-context-second", 2,
		func(ctx ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			if ctx.ToolName == "execution-context-outer" {
				outerInterceptOrder = append(outerInterceptOrder, "second")
			}
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
	if !reflect.DeepEqual(outerInterceptOrder, []string{"first-mutated", "second"}) {
		t.Fatalf("unexpected outer intercept order: %v", outerInterceptOrder)
	}
	if _, failure := ToolCallExecute("execution-context-subsequent", json.RawMessage(`{}`),
		func(arguments json.RawMessage) (ToolExecutionResult, error) {
			subsequentCalls++
			return ToolExecutionResult{Result: arguments}, nil
		}); failure != nil {
		t.Fatal(failure)
	}
	if businessCalls != 1 || subsequentCalls != 1 || string(result.Annotation) != `{"reviewed":true}` {
		t.Fatalf("unexpected effect counts or annotation: business=%d subsequent=%d result=%+v", businessCalls, subsequentCalls, result)
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

func TestApplicationContextEmptyRawMessageIsNotNull(test *testing.T) {
	invalid := ApplicationExecutionContext{Data: map[string]json.RawMessage{"x": {}}}
	if _, failure := json.Marshal(invalid); failure == nil {
		test.Fatal("empty non-nil JSON was converted to null")
	}
	for _, value := range []json.RawMessage{nil, json.RawMessage("null")} {
		context := ApplicationExecutionContext{Data: map[string]json.RawMessage{"x": value}}
		encoded, failure := json.Marshal(context)
		if failure != nil || string(encoded) != `{"data":{"x":null}}` {
			test.Fatalf("null context changed: %s %v", encoded, failure)
		}
		result, failure := ToolCallExecute("valid-null-context", json.RawMessage(`{}`), func(arguments json.RawMessage) (ToolExecutionResult, error) {
			return ToolExecutionResult{Result: arguments}, nil
		}, WithApplicationExecutionContext(&context))
		if failure != nil || string(result.Result) != "{}" {
			test.Fatalf("null should remain valid: %s %v", result.Result, failure)
		}
	}
}

func TestApplicationContextBudgetRefusesBeforeCallbacksAndRecovers(test *testing.T) {
	middlewareCalls, toolCalls := 0, 0
	name := "review-context-budget"
	if failure := RegisterToolExecutionIntercept(name, 1, func(context ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
		middlewareCalls++
		return toolExecutionOutcome(next(context.Args))
	}); failure != nil {
		test.Fatal(failure)
	}
	test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
	raw := json.RawMessage(`{"data":{"value":"` + strings.Repeat("x", 1_048_576) + `"}}`)
	typed := ApplicationExecutionContext{Data: map[string]json.RawMessage{"value": json.RawMessage(`"` + strings.Repeat("x", 1_048_576) + `"`)}}
	empty := ApplicationExecutionContext{Data: map[string]json.RawMessage{"x": {}}}
	deep := json.RawMessage(`{"data":{"value":` + strings.Repeat("[", 63) + `null` + strings.Repeat("]", 63) + `}}`)
	for _, option := range []ToolCallOption{WithToolExecutionContext(raw), WithApplicationExecutionContext(&typed), WithApplicationExecutionContext(&empty), WithToolExecutionContext(deep)} {
		_, failure := ToolCallExecute("refused-context", json.RawMessage(`{}`), func(arguments json.RawMessage) (ToolExecutionResult, error) {
			toolCalls++
			return ToolExecutionResult{Result: arguments}, nil
		}, option)
		if failure == nil {
			test.Fatal("invalid context accepted")
		}
		if middlewareCalls != 0 || toolCalls != 0 {
			test.Fatal("invalid context reached callbacks")
		}
	}
	if _, failure := ToolCallExecute("recovered-context", json.RawMessage(`{}`), func(arguments json.RawMessage) (ToolExecutionResult, error) {
		toolCalls++
		return ToolExecutionResult{Result: arguments}, nil
	}, WithToolExecutionContext(json.RawMessage(`{"data":{"value":null}}`))); failure != nil {
		test.Fatal(failure)
	}
	if middlewareCalls != 1 || toolCalls != 1 {
		test.Fatalf("valid recovery calls: %d/%d", middlewareCalls, toolCalls)
	}
}

func TestApplicationContextFloatAndEscapedBoundaries(test *testing.T) {
	for _, number := range []string{"1e-7", "1e-6", "0.00001", "1e+16", "-0.0", "5e-324", "18446744073709551615"} {
		test.Run(number, func(test *testing.T) {
			overhead := len(`{"data":{"number":` + number + `,"padding":""}}`)
			input := json.RawMessage(`{"data":{"number":` + number + `,"padding":"` + strings.Repeat("x", 16_384-overhead) + `"}}`)
			var options toolCallOptions
			WithToolExecutionContext(input)(&options)
			if options.executionContextError != nil {
				test.Fatal(options.executionContextError)
			}
			if len(options.executionContext) != 16_384 {
				test.Fatalf("canonical boundary is %d", len(options.executionContext))
			}
		})
	}
	for _, input := range []string{`{"data":{"value":"` + strings.Repeat(`\u0061`, 3_000) + `"}}`, strings.Repeat(" ", 20_000) + `{}`} {
		var options toolCallOptions
		WithToolExecutionContext(json.RawMessage(input))(&options)
		if options.executionContextError != nil {
			test.Fatal(options.executionContextError)
		}
	}
}

var retainedReviewContextOption ToolCallOption

func TestApplicationContextOversizeOptionsDoNotCopyThePayload(test *testing.T) {
	oversized := json.RawMessage(`{"data":{"value":"` + strings.Repeat("x", 1_048_576) + `"}}`)
	typed := ApplicationExecutionContext{Data: map[string]json.RawMessage{"value": json.RawMessage(`"` + strings.Repeat("x", 1_048_576) + `"`)}}
	for name, makeOption := range map[string]func() ToolCallOption{
		"raw":   func() ToolCallOption { return WithToolExecutionContext(oversized) },
		"typed": func() ToolCallOption { return WithApplicationExecutionContext(&typed) },
	} {
		test.Run(name, func(test *testing.T) {
			measured := testing.Benchmark(func(benchmark *testing.B) {
				for benchmark.Loop() {
					retainedReviewContextOption = makeOption()
				}
			})
			if measured.AllocedBytesPerOp() > 64*1024 {
				test.Fatalf("option copied oversized payload: %d bytes/op", measured.AllocedBytesPerOp())
			}
			var options toolCallOptions
			retainedReviewContextOption(&options)
			if options.executionContextError == nil {
				test.Fatal("oversized option retained no refusal")
			}
		})
	}
}

func TestApplicationContextBudgetPreservesRawNumbers(test *testing.T) {
	for _, number := range []string{"18446744073709551616", "1.00000000000000011102230246251565404236316680908203125", "1e-07"} {
		context := ApplicationExecutionContext{Data: map[string]json.RawMessage{"number": json.RawMessage(number)}}
		encoded, failure := json.Marshal(context)
		expected := `{"data":{"number":` + number + `}}`
		if failure != nil || string(encoded) != expected {
			test.Fatalf("raw number changed: %s %v", encoded, failure)
		}
		var options toolCallOptions
		WithToolExecutionContext(json.RawMessage(expected))(&options)
		if options.executionContextError != nil || string(options.executionContext) != expected {
			test.Fatalf("raw option changed numeric payload: %s %v", options.executionContext, options.executionContextError)
		}
	}
}

func TestApplicationContextAndCancellationCompose(test *testing.T) {
	for _, cancelDuringExecution := range []bool{false, true} {
		test.Run(fmt.Sprintf("cancel=%t", cancelDuringExecution), func(test *testing.T) {
			requestContext, cancelRequest := context.WithCancel(context.Background())
			defer cancelRequest()
			applicationContext := ApplicationExecutionContext{Data: map[string]json.RawMessage{
				"decision": json.RawMessage(`"reviewed"`),
			}}
			middlewareCalls, implementationCalls := 0, 0
			name := fmt.Sprintf("application-context-cancellation-%t", cancelDuringExecution)
			if failure := RegisterToolExecutionIntercept(name, 1,
				func(invocation ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
					middlewareCalls++
					if invocation.ExecutionContext == nil || string(invocation.ExecutionContext.Data["decision"]) != `"reviewed"` {
						return ToolExecutionInterceptOutcome{}, fmt.Errorf("application context did not reach middleware")
					}
					if string(invocation.Args) != `{"value":1}` {
						return ToolExecutionInterceptOutcome{}, fmt.Errorf("application context changed tool arguments")
					}
					return toolExecutionOutcome(next(invocation.Args))
				}); failure != nil {
				test.Fatal(failure)
			}
			test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
			result, failure := ToolCallExecuteContext(requestContext, name, json.RawMessage(`{"value":1}`),
				func(executionContext context.Context, arguments json.RawMessage) (ToolExecutionResult, error) {
					implementationCalls++
					if cancelDuringExecution {
						cancelRequest()
						<-executionContext.Done()
						return ToolExecutionResult{}, executionContext.Err()
					}
					return ToolExecutionResult{Result: arguments}, nil
				}, WithApplicationExecutionContext(&applicationContext))
			if cancelDuringExecution {
				if !errors.Is(failure, context.Canceled) {
					test.Fatalf("cancellation was lost: %v", failure)
				}
			} else if failure != nil || string(result.Result) != `{"value":1}` {
				test.Fatalf("valid context/cancellation composition failed: %s %v", result.Result, failure)
			}
			if middlewareCalls != 1 || implementationCalls != 1 {
				test.Fatalf("unexpected execution counts: %d/%d", middlewareCalls, implementationCalls)
			}
			recovered, failure := ToolCallExecute(name+"-recovery", json.RawMessage(`{"value":1}`),
				func(arguments json.RawMessage) (ToolExecutionResult, error) {
					implementationCalls++
					return ToolExecutionResult{Result: arguments}, nil
				}, WithApplicationExecutionContext(&applicationContext))
			if failure != nil || string(recovered.Result) != `{"value":1}` || middlewareCalls != 2 || implementationCalls != 2 {
				test.Fatalf("valid recovery failed: %s %v, calls=%d/%d", recovered.Result, failure, middlewareCalls, implementationCalls)
			}
		})
	}
}

func TestToolExecutionContextDoesNotInheritContextDuringNestedCalls(test *testing.T) {
	supplied := json.RawMessage(`{"data":{"marker":"outer-only"}}`)
	outerIntercepts, innerIntercepts, outerCalls, innerCalls := 0, 0, 0, 0
	name := "execution-context-nested-isolation"
	if failure := RegisterToolExecutionIntercept(name, 1,
		func(invocation ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			switch invocation.ToolName {
			case "execution-context-nested-outer":
				outerIntercepts++
				if !bytes.Equal(executionContextJSON(invocation.ExecutionContext), supplied) {
					return ToolExecutionInterceptOutcome{}, fmt.Errorf("outer intercept received wrong context")
				}
				result, failure := next(invocation.Args)
				if failure != nil {
					return ToolExecutionInterceptOutcome{}, failure
				}
				if !bytes.Equal(executionContextJSON(invocation.ExecutionContext), supplied) {
					return ToolExecutionInterceptOutcome{}, fmt.Errorf("nested execution changed outer context")
				}
				return toolExecutionOutcome(result, nil)
			case "execution-context-nested-inner":
				innerIntercepts++
				if invocation.ExecutionContext != nil {
					return ToolExecutionInterceptOutcome{}, fmt.Errorf("nested call inherited outer context")
				}
			}
			return toolExecutionOutcome(next(invocation.Args))
		}); failure != nil {
		test.Fatal(failure)
	}
	test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
	result, failure := ToolCallExecute("execution-context-nested-outer", json.RawMessage(`{"outer":true}`),
		func(arguments json.RawMessage) (ToolExecutionResult, error) {
			outerCalls++
			nested, failure := ToolCallExecute("execution-context-nested-inner", json.RawMessage(`{"inner":true}`),
				func(arguments json.RawMessage) (ToolExecutionResult, error) {
					innerCalls++
					return ToolExecutionResult{Result: arguments}, nil
				})
			if failure != nil {
				return ToolExecutionResult{}, failure
			}
			if string(nested.Result) != `{"inner":true}` {
				return ToolExecutionResult{}, fmt.Errorf("nested call returned wrong result: %s", nested.Result)
			}
			return ToolExecutionResult{Result: arguments}, nil
		}, WithToolExecutionContext(supplied))
	if failure != nil {
		test.Fatal(failure)
	}
	if string(result.Result) != `{"outer":true}` || outerIntercepts != 1 || innerIntercepts != 1 || outerCalls != 1 || innerCalls != 1 {
		test.Fatalf("unexpected nested execution: result=%s outer intercepts/calls=%d/%d inner intercepts/calls=%d/%d", result.Result, outerIntercepts, outerCalls, innerIntercepts, innerCalls)
	}
}

func TestApplicationContextTypedKeysRequireValidUTF8(test *testing.T) {
	interceptCalls, toolCalls := 0, 0
	name := "execution-context-key-validation"
	if failure := RegisterToolExecutionIntercept(name, 1,
		func(invocation ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			interceptCalls++
			if invocation.ExecutionContext == nil || len(invocation.ExecutionContext.Data) != 1 || string(invocation.ExecutionContext.Data["\uFFFD"]) != "true" {
				return ToolExecutionInterceptOutcome{}, fmt.Errorf("valid replacement-character key changed")
			}
			return toolExecutionOutcome(next(invocation.Args))
		}); failure != nil {
		test.Fatal(failure)
	}
	test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
	implementation := func(arguments json.RawMessage) (ToolExecutionResult, error) {
		toolCalls++
		return ToolExecutionResult{Result: arguments}, nil
	}
	for scenario, fields := range map[string]map[string]json.RawMessage{
		"invalid-key":    {string([]byte{0xff}): json.RawMessage("false")},
		"colliding-keys": {string([]byte{0xff}): json.RawMessage("false"), "\uFFFD": json.RawMessage("true")},
	} {
		test.Run(scenario, func(test *testing.T) {
			context := ApplicationExecutionContext{Data: fields}
			if _, failure := json.Marshal(context); failure == nil || !strings.Contains(failure.Error(), "UTF-8") {
				test.Fatalf("invalid key did not fail marshaling: %v", failure)
			}
			if _, failure := ToolCallExecute(name, json.RawMessage(`{}`), implementation, WithApplicationExecutionContext(&context)); failure == nil || !strings.Contains(failure.Error(), "UTF-8") {
				test.Fatalf("invalid key did not fail managed execution: %v", failure)
			}
			if interceptCalls != 0 || toolCalls != 0 {
				test.Fatalf("invalid key reached callbacks: %d/%d", interceptCalls, toolCalls)
			}
		})
	}
	valid := ApplicationExecutionContext{Data: map[string]json.RawMessage{"\uFFFD": json.RawMessage("true")}}
	encoded, failure := json.Marshal(valid)
	if failure != nil {
		test.Fatal(failure)
	}
	var decoded ApplicationExecutionContext
	if failure := json.Unmarshal(encoded, &decoded); failure != nil || string(decoded.Data["\uFFFD"]) != "true" || len(decoded.Data) != 1 {
		test.Fatalf("valid key did not round trip: %s %v", encoded, failure)
	}
	result, failure := ToolCallExecute(name, json.RawMessage(`{}`), implementation, WithApplicationExecutionContext(&valid))
	if failure != nil || string(result.Result) != `{}` || interceptCalls != 1 || toolCalls != 1 {
		test.Fatalf("valid key did not execute: result=%s error=%v callbacks=%d/%d", result.Result, failure, interceptCalls, toolCalls)
	}
}

func TestApplicationContextDuplicateKeysUseRetainedValueBudget(test *testing.T) {
	oversized := `"` + strings.Repeat(`\u20ac`, 5_500) + `"`
	for _, scenario := range []struct {
		name     string
		object   string
		accepted bool
	}{
		{"overwritten-string", `{"x":` + oversized + `,"x":null}`, true},
		{"retained-string", `{"x":null,"x":` + oversized + `}`, false},
		{"overwritten-object", `{"x":{"child":` + oversized + `},"x":null}`, true},
		{"equivalent-escaped-key", `{"\u0078":` + oversized + `,"x":null}`, true},
	} {
		for _, typed := range []bool{false, true} {
			test.Run(fmt.Sprintf("%s/typed=%t", scenario.name, typed), func(test *testing.T) {
				interceptCalls, toolCalls := 0, 0
				name := "execution-context-duplicate-budget"
				if failure := RegisterToolExecutionIntercept(name, 1,
					func(invocation ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
						interceptCalls++
						if invocation.ExecutionContext == nil {
							return ToolExecutionInterceptOutcome{}, fmt.Errorf("missing execution context")
						}
						observed := invocation.ExecutionContext.Data["x"]
						if typed {
							var retained map[string]json.RawMessage
							if failure := json.Unmarshal(invocation.ExecutionContext.Data["nested"], &retained); failure != nil || len(retained) != 1 {
								return ToolExecutionInterceptOutcome{}, fmt.Errorf("invalid retained object: %v", failure)
							}
							observed = retained["x"]
						}
						if string(observed) != "null" {
							return ToolExecutionInterceptOutcome{}, fmt.Errorf("middleware did not receive retained null: %s", observed)
						}
						return toolExecutionOutcome(next(invocation.Args))
					}); failure != nil {
					test.Fatal(failure)
				}
				test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
				var option ToolCallOption
				if typed {
					context := ApplicationExecutionContext{Data: map[string]json.RawMessage{"nested": json.RawMessage(scenario.object)}}
					option = WithApplicationExecutionContext(&context)
				} else {
					option = WithToolExecutionContext(json.RawMessage(`{"data":` + scenario.object + `}`))
				}
				result, failure := ToolCallExecute(name, json.RawMessage(`{}`), func(arguments json.RawMessage) (ToolExecutionResult, error) {
					toolCalls++
					return ToolExecutionResult{Result: arguments}, nil
				}, option)
				if !scenario.accepted {
					if failure == nil || !strings.Contains(failure.Error(), "16384") || interceptCalls != 0 || toolCalls != 0 {
						test.Fatalf("retained oversize value was not refused before callbacks: %v, %d/%d", failure, interceptCalls, toolCalls)
					}
					return
				}
				if failure != nil || string(result.Result) != `{}` || interceptCalls != 1 || toolCalls != 1 {
					test.Fatalf("valid retained context did not execute: result=%s error=%v callbacks=%d/%d", result.Result, failure, interceptCalls, toolCalls)
				}
			})
		}
	}
}

func TestApplicationContextPaddedOptionsRetainBoundedIndependentSnapshots(test *testing.T) {
	for name, input := range map[string]json.RawMessage{
		"whitespace":   json.RawMessage(strings.Repeat(" ", 1_048_576) + `{}`),
		"long-decimal": json.RawMessage(`{"data":{"number":1.` + strings.Repeat("0", 1_048_576) + `}}`),
	} {
		test.Run(name, func(test *testing.T) {
			option := WithToolExecutionContext(input)
			input[0] = '!'
			var first, second toolCallOptions
			option(&first)
			if first.executionContextError != nil {
				test.Fatal(first.executionContextError)
			}
			if len(first.executionContext) > maximumApplicationContextBytes {
				test.Fatal("unbounded snapshot")
			}
			original := bytes.Clone(first.executionContext)
			first.executionContext[0] = '!'
			option(&second)
			if !bytes.Equal(second.executionContext, original) {
				test.Fatal("reused option shared mutable state")
			}
		})
	}
}

func TestApplicationContextNormalizationPreservesNativeNumbers(test *testing.T) {
	name := "context-native-number-normalization"
	var observed json.RawMessage
	if failure := RegisterToolExecutionIntercept(name, 1,
		func(context ToolExecutionContext, next func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
			observed = bytes.Clone(context.ExecutionContext.Data["number"])
			result, failure := next(context.Args)
			return ToolExecutionInterceptOutcome{Result: result.Result}, failure
		}); failure != nil {
		test.Fatal(failure)
	}
	test.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })
	for _, number := range []string{"18446744073709551615", "18446744073709551616", "1.00000000000000011102230246251565404236316680908203125", "1e-07", "3e+23", "-0.0", "5e-324"} {
		test.Run(number, func(test *testing.T) {
			execute := func(option ToolCallOption) json.RawMessage {
				observed = nil
				_, failure := ToolCallExecute(name, json.RawMessage(`{}`), func(arguments json.RawMessage) (ToolExecutionResult, error) {
					return ToolExecutionResult{Result: arguments}, nil
				}, option)
				if failure != nil {
					test.Fatal(failure)
				}
				if observed == nil {
					test.Fatal("intercept did not run")
				}
				return bytes.Clone(observed)
			}
			raw := `{"data":{"number":` + number + `}}`
			expected := execute(WithToolExecutionContext(json.RawMessage(raw)))
			actual := execute(WithToolExecutionContext(json.RawMessage(strings.Repeat(" ", 1_048_576) + raw)))
			if !bytes.Equal(actual, expected) {
				test.Fatalf("raw normalized number changed: %s != %s", actual, expected)
			}
			typed := ApplicationExecutionContext{Data: map[string]json.RawMessage{
				"number": json.RawMessage(strings.Repeat(" ", 1_048_576) + number),
			}}
			actual = execute(WithApplicationExecutionContext(&typed))
			if !bytes.Equal(actual, expected) {
				test.Fatalf("typed normalized number changed: %s != %s", actual, expected)
			}
		})
	}
}

func TestApplicationContextPaddedConstructionAndApplicationAllocationsStayBounded(test *testing.T) {
	longValue := json.RawMessage("1." + strings.Repeat("0", 1_048_576))
	for name, construct := range map[string]func() ToolCallOption{
		"raw-whitespace": func() ToolCallOption {
			return WithToolExecutionContext(json.RawMessage(strings.Repeat(" ", 1_048_576) + `{}`))
		},
		"typed-number": func() ToolCallOption {
			return WithApplicationExecutionContext(&ApplicationExecutionContext{Data: map[string]json.RawMessage{"number": longValue}})
		},
	} {
		// Construct fixtures before measurement; only option work is charged.
		if name == "raw-whitespace" {
			input := json.RawMessage(strings.Repeat(" ", 1_048_576) + `{}`)
			construct = func() ToolCallOption { return WithToolExecutionContext(input) }
		}
		test.Run(name, func(test *testing.T) {
			measured := testing.Benchmark(func(benchmark *testing.B) {
				for benchmark.Loop() {
					retainedReviewContextOption = construct()
				}
			})
			if measured.AllocedBytesPerOp() > 64*1024 {
				test.Fatalf("construction allocated %d bytes/op", measured.AllocedBytesPerOp())
			}
			option := construct()
			var accepted toolCallOptions
			option(&accepted)
			if accepted.executionContextError != nil {
				test.Fatal(accepted.executionContextError)
			}
			if len(accepted.executionContext) > maximumApplicationContextBytes {
				test.Fatal("typed option retained oversized representation")
			}
			measured = testing.Benchmark(func(benchmark *testing.B) {
				for benchmark.Loop() {
					var options toolCallOptions
					option(&options)
					if options.executionContextError != nil {
						benchmark.Fatal(options.executionContextError)
					}
					if len(options.executionContext) > maximumApplicationContextBytes {
						benchmark.Fatal("unbounded application")
					}
				}
			})
			if measured.AllocedBytesPerOp() > 64*1024 {
				test.Fatalf("application allocated %d bytes/op", measured.AllocedBytesPerOp())
			}
		})
	}
}

func TestApplicationContextOversizedNumericTokensFailWithoutGoCopies(test *testing.T) {
	number := json.RawMessage(strings.Repeat("9", 1_048_576))
	raw := json.RawMessage(`{"data":{"number":` + string(number) + `}}`)
	typed := ApplicationExecutionContext{Data: map[string]json.RawMessage{"number": number}}
	for name, construct := range map[string]func() ToolCallOption{
		"raw":   func() ToolCallOption { return WithToolExecutionContext(raw) },
		"typed": func() ToolCallOption { return WithApplicationExecutionContext(&typed) },
	} {
		test.Run(name, func(test *testing.T) {
			measured := testing.Benchmark(func(benchmark *testing.B) {
				for benchmark.Loop() {
					retainedReviewContextOption = construct()
				}
			})
			if measured.AllocedBytesPerOp() > 64*1024 {
				test.Fatalf("refused number copied input: %d bytes/op", measured.AllocedBytesPerOp())
			}
			var refused toolCallOptions
			retainedReviewContextOption(&refused)
			if refused.executionContextError == nil || refused.executionContext != nil {
				test.Fatal("out-of-range numeric input was retained")
			}
		})
	}
}
