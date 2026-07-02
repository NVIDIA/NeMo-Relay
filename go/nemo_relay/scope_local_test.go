// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
)

const (
	scopeLocalNewScopeStackFailed                  = "NewScopeStack failed: %v"
	scopeLocalPushScopeFailed                      = "PushScope failed: %v"
	scopeLocalToolCallExecuteFailed                = "ToolCallExecute failed: %v"
	scopeLocalRegisterToolSanitizeRequestGuardrail = "ScopeRegisterToolSanitizeRequestGuardrail failed: %v"
	scopeLocalRegisterToolRequestInterceptFailed   = "ScopeRegisterToolRequestIntercept failed: %v"
	scopeLocalRegisterSubscriberFailed             = "ScopeRegisterSubscriber failed: %v"
	scopeLocalRegisterFailed                       = "ScopeRegister failed: %v"
	scopeLocalFlushSubscribersFailed               = "FlushSubscribers failed: %v"
	scopeLocalGuardrailRejected                    = "guardrail rejected"
)

type scopeLocalCapturedEvent struct {
	kind          string
	category      string
	scopeCategory string
	input         json.RawMessage
	output        json.RawMessage
}

func assertCapturedScopeLocalEventHasBoolFlag(t *testing.T, events []scopeLocalCapturedEvent, category, scopeCategory, key string) {
	t.Helper()

	for _, ev := range events {
		if ev.kind != "scope" || ev.category != category || ev.scopeCategory != scopeCategory {
			continue
		}

		payload := ev.input
		if scopeCategory == "end" {
			payload = ev.output
		}
		if payload == nil {
			continue
		}

		var decoded map[string]interface{}
		json.Unmarshal(payload, &decoded)
		if decoded[key] == true {
			return
		}
	}

	t.Fatalf("expected %s %s event payload to contain %s=true", category, scopeCategory, key)
}

func executeScopeLocalToolPassthrough(name, payload string) error {
	_, err := ToolCallExecute(name, json.RawMessage(payload),
		func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
	)
	return err
}

func executeScopeLocalLLMPassthrough(name string, request map[string]interface{}) error {
	_, err := LlmCallExecute(name, request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok":true}`), nil
		},
	)
	return err
}

func assertScopeLocalCallbackDeregisters(
	t *testing.T,
	label string,
	calls *int,
	register func() error,
	deregister func() error,
	runBefore func() error,
	runAfter func() error,
) {
	t.Helper()

	if err := register(); err != nil {
		t.Fatalf("register %s failed: %v", label, err)
	}
	if err := runBefore(); err != nil {
		t.Fatalf("%s before deregister failed: %v", label, err)
	}
	if *calls != 1 {
		t.Fatalf("expected %s callback once, got %d", label, *calls)
	}
	if err := deregister(); err != nil {
		t.Fatalf("deregister %s failed: %v", label, err)
	}
	if err := runAfter(); err != nil {
		t.Fatalf("%s after deregister failed: %v", label, err)
	}
	if *calls != 1 {
		t.Fatalf("%s callback still fired after deregister: %d", label, *calls)
	}
}

func assertScopeLocalWrappedStream(t *testing.T, stream *LlmStream) {
	t.Helper()

	chunk, err := stream.Next()
	if err != nil {
		t.Fatalf("stream.Next() failed: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(chunk, &payload); err != nil {
		t.Fatalf("unmarshal stream chunk: %v", err)
	}
	if payload["scope_intercepted"] != true {
		t.Fatalf("expected scope_intercepted=true, got %v", payload)
	}
	nextPayload, ok := payload["next"].(map[string]interface{})
	if !ok || nextPayload["streamed"] != true {
		t.Fatalf("expected next.streamed=true, got %v", payload["next"])
	}
	if _, err := stream.Next(); err != io.EOF {
		t.Fatalf("expected EOF after single wrapped chunk, got %v", err)
	}
}

func runScopeLocalTaggedToolCall( // NOSONAR(S107)
	t *testing.T,
	stack *ScopeStack,
	scopeName string,
	interceptName string,
	callName string,
	source string,
	tag string,
	result *json.RawMessage,
	errOut *error,
) {
	t.Helper()

	stack.Run(func() {
		handle, err := PushScope(scopeName, ScopeTypeAgent)
		if err != nil {
			*errOut = err
			return
		}
		defer PopScope(handle)

		err = ScopeRegisterToolRequestIntercept(handle.UUID(), interceptName, 1, false, func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m[tag] = true
			resultJSON, _ := json.Marshal(m)
			return resultJSON
		})
		if err != nil {
			*errOut = err
			return
		}
		*result, *errOut = ToolCallExecute(callName, json.RawMessage(`{"source": "`+source+`"}`), func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		})
	})
}

func assertJSONTagState(t *testing.T, raw json.RawMessage, presentTag, absentTag string) {
	t.Helper()

	var decoded map[string]interface{}
	json.Unmarshal(raw, &decoded)
	if decoded[presentTag] != true {
		t.Fatalf("result missing %s", presentTag)
	}
	if _, present := decoded[absentTag]; present {
		t.Fatalf("cross-contamination: result has %s", absentTag)
	}
}

// ============================================================================
// Scope-local guardrail registration
// ============================================================================

func TestScopeLocalToolSanitizeRequestGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		var events []scopeLocalCapturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_req_sub", func(e Event) {
			if e.Kind() != "scope" || e.Category() != "tool" || e.ScopeCategory() != "start" {
				return
			}
			mu.Lock()
			events = append(events, scopeLocalCapturedEvent{
				kind:          e.Kind(),
				category:      e.Category(),
				scopeCategory: e.ScopeCategory(),
				input:         append(json.RawMessage(nil), e.Input()...),
			})
			mu.Unlock()
		})
		if err != nil {
			t.Fatalf("RegisterSubscriber failed: %v", err)
		}
		defer DeregisterSubscriber("scope_san_req_sub")

		handle, err := PushScope("guardrail_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "scope_san_req", 1,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["scope_sanitized"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolSanitizeRequestGuardrail, err)
		}

		_, err = ToolCallExecute("scope_guarded_tool", json.RawMessage(`{"value": 42}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatalf(scopeLocalFlushSubscribersFailed, err)
		}

		mu.Lock()
		defer mu.Unlock()
		assertCapturedScopeLocalEventHasBoolFlag(t, events, "tool", "start", "scope_sanitized")
	})
}

func TestScopeLocalToolSanitizeResponseGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		var events []scopeLocalCapturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_resp_sub", func(e Event) {
			if e.Kind() != "scope" || e.Category() != "tool" || e.ScopeCategory() != "end" {
				return
			}
			mu.Lock()
			events = append(events, scopeLocalCapturedEvent{
				kind:          e.Kind(),
				category:      e.Category(),
				scopeCategory: e.ScopeCategory(),
				output:        append(json.RawMessage(nil), e.Output()...),
			})
			mu.Unlock()
		})
		if err != nil {
			t.Fatalf("RegisterSubscriber failed: %v", err)
		}
		defer DeregisterSubscriber("scope_san_resp_sub")

		handle, err := PushScope("resp_guard_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)

		err = ScopeRegisterToolSanitizeResponseGuardrail(handle.UUID(), "scope_san_resp", 1,
			func(name string, result json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(result, &m)
				m["response_sanitized"] = true
				out, _ := json.Marshal(m)
				return out
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeResponseGuardrail failed: %v", err)
		}

		_, err = ToolCallExecute("resp_tool", json.RawMessage(`{}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"output": "data"}`), nil
			},
		)
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatalf(scopeLocalFlushSubscribersFailed, err)
		}

		mu.Lock()
		defer mu.Unlock()
		assertCapturedScopeLocalEventHasBoolFlag(t, events, "tool", "end", "response_sanitized")
	})
}

func TestScopeLocalToolConditionalExecutionGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("cond_guard_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)

		msg := "scope-local block"
		err = ScopeRegisterToolConditionalExecutionGuardrail(handle.UUID(), "scope_cond", 1,
			func(name string, args json.RawMessage) *string {
				return &msg
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolConditionalExecutionGuardrail failed: %v", err)
		}

		_, err = ToolCallExecute("cond_tool", json.RawMessage(`{}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"should": "not reach"}`), nil
			},
		)
		if err == nil {
			t.Fatal("expected error from scope-local conditional guardrail rejection")
		}
		if !strings.Contains(err.Error(), scopeLocalGuardrailRejected) {
			t.Fatalf("expected 'guardrail rejected' error, got: %v", err)
		}
	})
}

// ============================================================================
// Auto-cleanup on scope pop
// ============================================================================

func TestScopeLocalGuardrailCleanupOnPop(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "cleanup_guard", 1,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["from_popped_scope"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolSanitizeRequestGuardrail, err)
		}
		err = PopScope(handle)
		if err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
		result, err := ToolCallExecute("after_pop_tool", json.RawMessage(`{"original": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if _, present := output["from_popped_scope"]; present {
			t.Fatal("scope-local guardrail should have been cleaned up on pop, but it still ran")
		}
		if output["original"] != true {
			t.Fatalf("expected original=true, got %v", output)
		}
	})
}

func TestScopeLocalInterceptCleanupOnPop(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("intercept_cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		err = ScopeRegisterToolRequestIntercept(handle.UUID(), "cleanup_intercept", 1, false,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["from_popped_intercept"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolRequestInterceptFailed, err)
		}
		PopScope(handle)
		result, err := ToolCallExecute("after_intercept_pop", json.RawMessage(`{"check": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if _, present := output["from_popped_intercept"]; present {
			t.Fatal("scope-local intercept should have been cleaned up on pop")
		}
	})
}

func TestScopeLocalSubscriberCleanupOnPop(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("sub_cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		var eventCount int
		var mu sync.Mutex
		err = ScopeRegisterSubscriber(handle.UUID(), "cleanup_sub", func(event Event) { mu.Lock(); eventCount++; mu.Unlock() })
		if err != nil {
			t.Fatalf(scopeLocalRegisterSubscriberFailed, err)
		}
		PopScope(handle)
		mu.Lock()
		countAfterPop := eventCount
		mu.Unlock()
		EmitEvent("after_pop_event")
		mu.Lock()
		countAfterEmit := eventCount
		mu.Unlock()
		if countAfterEmit != countAfterPop {
			t.Fatalf("scope-local subscriber should not fire after pop; count went from %d to %d", countAfterPop, countAfterEmit)
		}
	})
}

// ============================================================================
// Priority merge: global + scope-local guardrails
// ============================================================================

func TestPriorityMergeGlobalAndScopeLocal(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		var order []string
		var mu sync.Mutex
		err := RegisterToolSanitizeRequestGuardrail("global_priority_guard", 10, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "global_p10")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf("RegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		defer DeregisterToolSanitizeRequestGuardrail("global_priority_guard")
		handle, err := PushScope("priority_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_priority_guard", 5, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "scope_p5")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolSanitizeRequestGuardrail, err)
		}
		_, err = ToolCallExecute("priority_tool", json.RawMessage(`{"input": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(order) != 2 {
			t.Fatalf("expected 2 guardrail executions, got %d", len(order))
		}
		if order[0] != "scope_p5" {
			t.Fatalf("expected scope_p5 to run first, got %s", order[0])
		}
		if order[1] != "global_p10" {
			t.Fatalf("expected global_p10 to run second, got %s", order[1])
		}
	})
}

func TestPriorityMergeGlobalBeforeScopeLocal(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()
	stack.Run(func() {
		var order []string
		var mu sync.Mutex
		err := RegisterToolSanitizeRequestGuardrail("global_first", 1, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "global_p1")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf("RegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		defer DeregisterToolSanitizeRequestGuardrail("global_first")
		handle, err := PushScope("priority_order_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_second", 20, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "scope_p20")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolSanitizeRequestGuardrail, err)
		}
		_, err = ToolCallExecute("order_tool", json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(order) != 2 {
			t.Fatalf("expected 2 guardrail executions, got %d", len(order))
		}
		if order[0] != "global_p1" {
			t.Fatalf("expected global_p1 first, got %s", order[0])
		}
		if order[1] != "scope_p20" {
			t.Fatalf("expected scope_p20 second, got %s", order[1])
		}
	})
}

// ============================================================================
// Isolation: separate goroutines with separate ScopeStacks
// ============================================================================

func TestScopeLocalIsolationBetweenGoroutines(t *testing.T) {
	stack1, _ := NewScopeStack()
	defer stack1.Close()
	stack2, _ := NewScopeStack()
	defer stack2.Close()
	var wg sync.WaitGroup
	var result1, result2 json.RawMessage
	var err1, err2 error
	wg.Add(2)
	go func() {
		defer wg.Done()
		runScopeLocalTaggedToolCall(t, stack1, "iso_scope_1", "iso_intercept_1", "iso_tool_1", "g1", "goroutine1_tag", &result1, &err1)
	}()
	go func() {
		defer wg.Done()
		runScopeLocalTaggedToolCall(t, stack2, "iso_scope_2", "iso_intercept_2", "iso_tool_2", "g2", "goroutine2_tag", &result2, &err2)
	}()
	wg.Wait()
	if err1 != nil {
		t.Fatalf("goroutine 1 failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("goroutine 2 failed: %v", err2)
	}
	assertJSONTagState(t, result1, "goroutine1_tag", "goroutine2_tag")
	assertJSONTagState(t, result2, "goroutine2_tag", "goroutine1_tag")
}

func TestScopeLocalConditionalGuardrailIsolation(t *testing.T) {
	stack1, _ := NewScopeStack()
	defer stack1.Close()
	stack2, _ := NewScopeStack()
	defer stack2.Close()
	var wg sync.WaitGroup
	var err1, err2 error
	var result2 json.RawMessage
	wg.Add(2)
	go func() {
		defer wg.Done()
		stack1.Run(func() {
			handle, err := PushScope("block_scope", ScopeTypeAgent)
			if err != nil {
				t.Errorf(scopeLocalPushScopeFailed, err)
				return
			}
			defer PopScope(handle)
			blockMsg := "blocked in scope 1"
			err = ScopeRegisterToolConditionalExecutionGuardrail(handle.UUID(), "block_guard", 1, func(name string, args json.RawMessage) *string { return &blockMsg })
			if err != nil {
				t.Errorf(scopeLocalRegisterFailed, err)
				return
			}
			_, err1 = ToolCallExecute("blocked_tool", json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"reached": true}`), nil })
		})
	}()
	go func() {
		defer wg.Done()
		stack2.Run(func() {
			handle, err := PushScope("allow_scope", ScopeTypeAgent)
			if err != nil {
				t.Errorf(scopeLocalPushScopeFailed, err)
				return
			}
			defer PopScope(handle)
			result2, err2 = ToolCallExecute("allowed_tool", json.RawMessage(`{"ok": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		})
	}()
	wg.Wait()
	if err1 == nil {
		t.Fatal("expected goroutine 1 to be blocked")
	}
	if !strings.Contains(err1.Error(), scopeLocalGuardrailRejected) {
		t.Fatalf("expected 'guardrail rejected', got: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("goroutine 2 should succeed, got: %v", err2)
	}
	var out2 map[string]interface{}
	json.Unmarshal(result2, &out2)
	if out2["ok"] != true {
		t.Fatalf("goroutine 2 expected ok=true, got %v", out2)
	}
}

// ============================================================================
// Scope-local intercepts
// ============================================================================

func TestScopeLocalToolRequestIntercept(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("req_intercept_scope", ScopeTypeAgent)
		defer PopScope(handle)
		err := ScopeRegisterToolRequestIntercept(handle.UUID(), "scope_req_int", 1, false, func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["scope_intercepted"] = true
			result, _ := json.Marshal(m)
			return result
		})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolRequestInterceptFailed, err)
		}
		result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"data": 1}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if output["scope_intercepted"] != true {
			t.Fatalf("expected scope_intercepted=true, got %v", output)
		}
	})
}

func TestScopeLocalToolExecutionIntercept(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("exec_intercept_scope", ScopeTypeAgent)
		defer PopScope(handle)
		err := ScopeRegisterToolExecutionIntercept(handle.UUID(), "scope_exec_int", 1, func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
			result, err := next(args)
			if err != nil {
				return ToolExecutionInterceptOutcome{}, err
			}
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["exec_intercepted"] = true
			out, _ := json.Marshal(m)
			return ToolExecutionInterceptOutcome{Result: out}, nil
		})
		if err != nil {
			t.Fatalf("ScopeRegisterToolExecutionIntercept failed: %v", err)
		}
		result, err := ToolCallExecute("exec_int_tool", json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"original": true}`), nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if output["original"] != true {
			t.Fatal("expected original=true")
		}
		if output["exec_intercepted"] != true {
			t.Fatal("expected exec_intercepted=true")
		}
	})
}

// ============================================================================
// Scope-local subscriber
// ============================================================================

func TestScopeLocalSubscriberReceivesEvents(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("sub_scope", ScopeTypeAgent)
		var eventNames []string
		var mu sync.Mutex
		err := ScopeRegisterSubscriber(handle.UUID(), "scope_sub", func(event Event) { mu.Lock(); eventNames = append(eventNames, event.Name()); mu.Unlock() })
		if err != nil {
			t.Fatalf(scopeLocalRegisterSubscriberFailed, err)
		}
		child, _ := PushScope("child_scope", ScopeTypeFunction)
		PopScope(child)
		PopScope(handle)
		if err := FlushSubscribers(); err != nil {
			t.Fatalf(scopeLocalFlushSubscribersFailed, err)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(eventNames) < 2 {
			t.Fatalf("expected at least 2 events from child scope start+end, got %d", len(eventNames))
		}
	})
}

// ============================================================================
// Explicit deregistration of scope-local middleware
// ============================================================================

func TestScopeLocalExplicitDeregistration(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("explicit_dereg_scope", ScopeTypeAgent)
		defer PopScope(handle)
		scopeUUID := handle.UUID()
		err := ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "explicit_guard", 1, func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["should_not_appear"] = true
			result, _ := json.Marshal(m)
			return result
		})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolSanitizeRequestGuardrail, err)
		}
		err = ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, "explicit_guard")
		if err != nil {
			t.Fatalf("ScopeDeregisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		result, err := ToolCallExecute("after_dereg_tool", json.RawMessage(`{"test": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if _, present := output["should_not_appear"]; present {
			t.Fatal("guardrail should not run after explicit deregistration")
		}
	})
}

func TestScopeLocalDuplicateRegistrationFails(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("dup_scope", ScopeTypeAgent)
		defer PopScope(handle)
		scopeUUID := handle.UUID()
		guardFn := func(name string, args json.RawMessage) json.RawMessage { return args }
		err := ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "dup_guard", 1, guardFn)
		if err != nil {
			t.Fatalf("first registration should succeed: %v", err)
		}
		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "dup_guard", 1, guardFn)
		if err == nil {
			t.Fatal("expected error for duplicate scope-local guardrail registration")
		}
	})
}

// ============================================================================
// Scope-local request intercept applied within scope (verifiable through callable)
// ============================================================================

func TestScopeLocalInterceptAppliedWithinScope(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("active_scope", ScopeTypeAgent)
		defer PopScope(handle)
		err := ScopeRegisterToolRequestIntercept(handle.UUID(), "active_intercept", 1, false,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["intercepted"] = true
				result, _ := json.Marshal(m)
				return result
			})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolRequestInterceptFailed, err)
		}
		result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"input": 1}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if output["intercepted"] != true {
			t.Fatal("expected intercepted=true, intercept was not applied within scope")
		}
	})
}

// ============================================================================
// Scope-local + global intercept merging
// ============================================================================

func TestScopeLocalAndGlobalInterceptMerging(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		err := RegisterToolRequestIntercept("global_merge_int", 5, false,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["global_applied"] = true
				result, _ := json.Marshal(m)
				return result
			})
		if err != nil {
			t.Fatalf("RegisterToolRequestIntercept failed: %v", err)
		}
		defer DeregisterToolRequestIntercept("global_merge_int")

		handle, _ := PushScope("merge_scope", ScopeTypeAgent)
		defer PopScope(handle)
		err = ScopeRegisterToolRequestIntercept(handle.UUID(), "scope_merge_int", 10, false,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["scope_applied"] = true
				result, _ := json.Marshal(m)
				return result
			})
		if err != nil {
			t.Fatalf(scopeLocalRegisterToolRequestInterceptFailed, err)
		}

		result, err := ToolCallExecute("merge_tool", json.RawMessage(`{"input": "data"}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf(scopeLocalToolCallExecuteFailed, err)
		}
		var output map[string]interface{}
		json.Unmarshal(result, &output)
		if output["global_applied"] != true {
			t.Fatal("expected global_applied=true")
		}
		if output["scope_applied"] != true {
			t.Fatal("expected scope_applied=true")
		}
	})
}

// ============================================================================
// Scope-local LLM guardrails (verified through events)
// ============================================================================

func TestScopeLocalLlmSanitizeRequestGuardrailAffectsEvent(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		var capturedInput json.RawMessage
		var mu sync.Mutex
		RegisterSubscriber("scope_llm_san_sub", func(event Event) {
			if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "start" {
				mu.Lock()
				capturedInput = append(json.RawMessage(nil), event.Input()...)
				mu.Unlock()
			}
		})
		defer DeregisterSubscriber("scope_llm_san_sub")

		handle, _ := PushScope("llm_scope_guard", ScopeTypeAgent)
		defer PopScope(handle)
		err := ScopeRegisterLlmSanitizeRequestGuardrail(handle.UUID(), "scope_llm_san_req", 1,
			func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
				var m map[string]interface{}
				json.Unmarshal(content, &m)
				m["scope_llm_sanitized"] = true
				out, _ := json.Marshal(m)
				return headers, out
			})
		if err != nil {
			t.Fatalf("ScopeRegisterLlmSanitizeRequestGuardrail failed: %v", err)
		}

		request := map[string]interface{}{"headers": map[string]interface{}{}, "content": map[string]interface{}{"model": "test"}}
		_, err = LlmCallExecute("scope_llm_guard_test", request, func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"done": true}`), nil
		})
		if err != nil {
			t.Fatalf("LlmCallExecute failed: %v", err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatalf(scopeLocalFlushSubscribersFailed, err)
		}

		mu.Lock()
		defer mu.Unlock()
		if capturedInput == nil {
			t.Fatal("expected non-nil captured input")
		}
		t.Logf("scope-local LLM sanitize guardrail affected event input: %s", string(capturedInput))
	})
}

func TestScopeLocalLlmConditionalGuardrail(t *testing.T) {
	stack, _ := NewScopeStack()
	defer stack.Close()
	stack.Run(func() {
		handle, _ := PushScope("llm_cond_scope", ScopeTypeAgent)
		defer PopScope(handle)
		msg := "scope-local LLM block"
		err := ScopeRegisterLlmConditionalExecutionGuardrail(handle.UUID(), "scope_llm_cond", 1, func(headers, content json.RawMessage) *string { return &msg })
		if err != nil {
			t.Fatalf("ScopeRegisterLlmConditionalExecutionGuardrail failed: %v", err)
		}
		request := map[string]interface{}{"headers": map[string]interface{}{}, "content": map[string]interface{}{"model": "test"}}
		_, err = LlmCallExecute("blocked_scope_llm", request, func(nativeJSON json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{}`), nil })
		if err == nil {
			t.Fatal("expected guardrail rejection from scope-local LLM conditional")
		}
		if !strings.Contains(err.Error(), scopeLocalGuardrailRejected) {
			t.Fatalf("expected 'guardrail rejected', got: %v", err)
		}
	})
}

func assertScopeLocalToolWrappersDeregister(t *testing.T, scopeUUID string) {
	t.Helper()

	sanitizeResponseCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"tool sanitize response",
		&sanitizeResponseCalls,
		func() error {
			return ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, "tool_scope_san_resp", 1,
				func(name string, result json.RawMessage) json.RawMessage {
					sanitizeResponseCalls++
					return result
				},
			)
		},
		func() error { return ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, "tool_scope_san_resp") },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_san_resp_call", `{"value":1}`) },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_san_resp_after", `{"value":2}`) },
	)

	conditionalCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"tool conditional guardrail",
		&conditionalCalls,
		func() error {
			return ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, "tool_scope_cond", 1,
				func(name string, args json.RawMessage) *string {
					conditionalCalls++
					return nil
				},
			)
		},
		func() error { return ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, "tool_scope_cond") },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_cond_call", `{"value":3}`) },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_cond_after", `{"value":4}`) },
	)

	requestInterceptCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"tool request intercept",
		&requestInterceptCalls,
		func() error {
			return ScopeRegisterToolRequestIntercept(scopeUUID, "tool_scope_req_int", 1, false,
				func(name string, args json.RawMessage) json.RawMessage {
					requestInterceptCalls++
					return args
				},
			)
		},
		func() error { return ScopeDeregisterToolRequestIntercept(scopeUUID, "tool_scope_req_int") },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_req_int_call", `{"value":5}`) },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_req_int_after", `{"value":6}`) },
	)

	executionInterceptCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"tool execution intercept",
		&executionInterceptCalls,
		func() error {
			return ScopeRegisterToolExecutionIntercept(scopeUUID, "tool_scope_exec_int", 1,
				func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
					executionInterceptCalls++
					return toolExecutionOutcome(next(args))
				},
			)
		},
		func() error { return ScopeDeregisterToolExecutionIntercept(scopeUUID, "tool_scope_exec_int") },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_exec_int_call", `{"value":7}`) },
		func() error { return executeScopeLocalToolPassthrough("tool_scope_exec_int_after", `{"value":8}`) },
	)
}

func TestScopeLocalExplicitDeregisterToolWrappers(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("tool_deregister_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)

		assertScopeLocalToolWrappersDeregister(t, handle.UUID())
	})
}

func assertScopeLocalLLMWrappersDeregister(t *testing.T, scopeUUID string, request map[string]interface{}) {
	t.Helper()

	sanitizeRequestCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"llm sanitize request",
		&sanitizeRequestCalls,
		func() error {
			return ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, "llm_scope_san_req", 1,
				func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
					sanitizeRequestCalls++
					return headers, content
				},
			)
		},
		func() error { return ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, "llm_scope_san_req") },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_san_req_call", request) },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_san_req_after", request) },
	)

	sanitizeResponseCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"llm sanitize response",
		&sanitizeResponseCalls,
		func() error {
			return ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, "llm_scope_san_resp", 1,
				func(responseJSON json.RawMessage) json.RawMessage {
					sanitizeResponseCalls++
					return responseJSON
				},
			)
		},
		func() error { return ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, "llm_scope_san_resp") },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_san_resp_call", request) },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_san_resp_after", request) },
	)

	conditionalCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"llm conditional guardrail",
		&conditionalCalls,
		func() error {
			return ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, "llm_scope_cond", 1,
				func(headers, content json.RawMessage) *string {
					conditionalCalls++
					return nil
				},
			)
		},
		func() error { return ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, "llm_scope_cond") },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_cond_call", request) },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_cond_after", request) },
	)

	requestInterceptCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"llm request intercept",
		&requestInterceptCalls,
		func() error {
			return ScopeRegisterLlmRequestIntercept(scopeUUID, "llm_scope_req_int", 1, false,
				func(name string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
					requestInterceptCalls++
					return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotated}, nil
				},
			)
		},
		func() error { return ScopeDeregisterLlmRequestIntercept(scopeUUID, "llm_scope_req_int") },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_req_int_call", request) },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_req_int_after", request) },
	)

	executionInterceptCalls := 0
	assertScopeLocalCallbackDeregisters(
		t,
		"llm execution intercept",
		&executionInterceptCalls,
		func() error {
			return ScopeRegisterLlmExecutionIntercept(scopeUUID, "llm_scope_exec_int", 1,
				func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
					executionInterceptCalls++
					return next(requestJSON)
				},
			)
		},
		func() error { return ScopeDeregisterLlmExecutionIntercept(scopeUUID, "llm_scope_exec_int") },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_exec_int_call", request) },
		func() error { return executeScopeLocalLLMPassthrough("llm_scope_exec_int_after", request) },
	)
}

func assertScopeLocalLLMStreamWrapperDeregisters(t *testing.T, scopeUUID string, request map[string]interface{}) {
	t.Helper()

	err := ScopeRegisterLlmStreamExecutionIntercept(scopeUUID, "llm_scope_stream_int", 1,
		func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			nextResult, err := next(requestJSON)
			if err != nil {
				return nil, err
			}
			return json.RawMessage(`{"scope_intercepted":true,"next":` + string(nextResult) + `}`), nil
		},
	)
	if err != nil {
		t.Fatalf("ScopeRegisterLlmStreamExecutionIntercept failed: %v", err)
	}

	stream, err := LlmStreamCallExecute("llm_scope_stream_int_call", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"streamed":true}`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf("LlmStreamCallExecute with stream intercept failed: %v", err)
	}
	assertScopeLocalWrappedStream(t, stream)
	stream.Close()

	if err := ScopeDeregisterLlmStreamExecutionIntercept(scopeUUID, "llm_scope_stream_int"); err != nil {
		t.Fatalf("ScopeDeregisterLlmStreamExecutionIntercept failed: %v", err)
	}
}

func assertScopeLocalSubscriberDeregisters(t *testing.T, scopeUUID string) {
	t.Helper()

	subscriberCalls := 0
	err := ScopeRegisterSubscriber(scopeUUID, "llm_scope_sub", func(event Event) {
		subscriberCalls++
	})
	if err != nil {
		t.Fatalf(scopeLocalRegisterSubscriberFailed, err)
	}
	if err := EmitEvent("llm_scope_sub_event_before"); err != nil {
		t.Fatalf("EmitEvent before subscriber deregister failed: %v", err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf(scopeLocalFlushSubscribersFailed, err)
	}
	if subscriberCalls == 0 {
		t.Fatal("expected scope-local subscriber to receive an event")
	}
	if err := ScopeDeregisterSubscriber(scopeUUID, "llm_scope_sub"); err != nil {
		t.Fatalf("ScopeDeregisterSubscriber failed: %v", err)
	}
	callsAfterDeregister := subscriberCalls
	if err := EmitEvent("llm_scope_sub_event_after"); err != nil {
		t.Fatalf("EmitEvent after subscriber deregister failed: %v", err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf(scopeLocalFlushSubscribersFailed, err)
	}
	if subscriberCalls != callsAfterDeregister {
		t.Fatalf("scope-local subscriber still fired after deregister: %d -> %d", callsAfterDeregister, subscriberCalls)
	}
}

func TestScopeLocalExplicitDeregisterLlmWrappers(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf(scopeLocalNewScopeStackFailed, err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("llm_deregister_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf(scopeLocalPushScopeFailed, err)
		}
		defer PopScope(handle)

		request := makeRequest()
		scopeUUID := handle.UUID()
		assertScopeLocalLLMWrappersDeregister(t, scopeUUID, request)
		assertScopeLocalLLMStreamWrapperDeregisters(t, scopeUUID, request)
		assertScopeLocalSubscriberDeregisters(t, scopeUUID)
	})
}
