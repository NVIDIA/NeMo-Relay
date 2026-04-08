// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// Scope-local guardrail registration
// ============================================================================

func TestScopeLocalToolSanitizeRequestGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		type capturedEvent struct {
			kind  string
			input json.RawMessage
		}
		var events []capturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_req_sub", func(e Event) {
			mu.Lock()
			events = append(events, capturedEvent{
				kind:  e.Kind(),
				input: append(json.RawMessage(nil), e.Input()...),
			})
			mu.Unlock()
		})
		if err != nil {
			t.Fatalf("RegisterSubscriber failed: %v", err)
		}
		defer DeregisterSubscriber("scope_san_req_sub")

		handle, err := PushScope("guardrail_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
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
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}

		_, err = ToolCallExecute("scope_guarded_tool", json.RawMessage(`{"value": 42}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		var found bool
		for _, ev := range events {
			if ev.kind == "ToolStart" && ev.input != nil {
				var m map[string]interface{}
				json.Unmarshal(ev.input, &m)
				if m["scope_sanitized"] == true {
					found = true
					break
				}
			}
		}
		if !found {
			t.Fatal("expected Start event input to contain scope_sanitized=true")
		}
	})
}

func TestScopeLocalToolSanitizeResponseGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		type capturedEvent struct {
			kind   string
			output json.RawMessage
		}
		var events []capturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_resp_sub", func(e Event) {
			mu.Lock()
			events = append(events, capturedEvent{
				kind:   e.Kind(),
				output: append(json.RawMessage(nil), e.Output()...),
			})
			mu.Unlock()
		})
		if err != nil {
			t.Fatalf("RegisterSubscriber failed: %v", err)
		}
		defer DeregisterSubscriber("scope_san_resp_sub")

		handle, err := PushScope("resp_guard_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
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
			t.Fatalf("ToolCallExecute failed: %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		var found bool
		for _, ev := range events {
			if ev.kind == "ToolEnd" && ev.output != nil {
				var m map[string]interface{}
				json.Unmarshal(ev.output, &m)
				if m["response_sanitized"] == true {
					found = true
					break
				}
			}
		}
		if !found {
			t.Fatal("expected End event output to contain response_sanitized=true")
		}
	})
}

func TestScopeLocalToolConditionalExecutionGuardrail(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("cond_guard_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
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
		if !strings.Contains(err.Error(), "guardrail rejected") {
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
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
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
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		err = PopScope(handle)
		if err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}
		result, err := ToolCallExecute("after_pop_tool", json.RawMessage(`{"original": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("intercept_cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
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
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}
		PopScope(handle)
		result, err := ToolCallExecute("after_intercept_pop", json.RawMessage(`{"check": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()
	stack.Run(func() {
		handle, err := PushScope("sub_cleanup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		var eventCount int
		var mu sync.Mutex
		err = ScopeRegisterSubscriber(handle.UUID(), "cleanup_sub", func(event Event) { mu.Lock(); eventCount++; mu.Unlock() })
		if err != nil {
			t.Fatalf("ScopeRegisterSubscriber failed: %v", err)
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
		t.Fatalf("NewScopeStack failed: %v", err)
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
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_priority_guard", 5, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "scope_p5")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		_, err = ToolCallExecute("priority_tool", json.RawMessage(`{"input": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
		t.Fatalf("NewScopeStack failed: %v", err)
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
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_second", 20, func(name string, args json.RawMessage) json.RawMessage {
			mu.Lock()
			order = append(order, "scope_p20")
			mu.Unlock()
			return args
		})
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		_, err = ToolCallExecute("order_tool", json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
		stack1.Run(func() {
			handle, err := PushScope("iso_scope_1", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope failed: %v", err)
				return
			}
			defer PopScope(handle)
			err = ScopeRegisterToolRequestIntercept(handle.UUID(), "iso_intercept_1", 1, false, func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["goroutine1_tag"] = true
				result, _ := json.Marshal(m)
				return result
			})
			if err != nil {
				t.Errorf("ScopeRegister failed: %v", err)
				return
			}
			result1, err1 = ToolCallExecute("iso_tool_1", json.RawMessage(`{"source": "g1"}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		})
	}()
	go func() {
		defer wg.Done()
		stack2.Run(func() {
			handle, err := PushScope("iso_scope_2", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope failed: %v", err)
				return
			}
			defer PopScope(handle)
			err = ScopeRegisterToolRequestIntercept(handle.UUID(), "iso_intercept_2", 1, false, func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["goroutine2_tag"] = true
				result, _ := json.Marshal(m)
				return result
			})
			if err != nil {
				t.Errorf("ScopeRegister failed: %v", err)
				return
			}
			result2, err2 = ToolCallExecute("iso_tool_2", json.RawMessage(`{"source": "g2"}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		})
	}()
	wg.Wait()
	if err1 != nil {
		t.Fatalf("goroutine 1 failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("goroutine 2 failed: %v", err2)
	}
	var out1 map[string]interface{}
	json.Unmarshal(result1, &out1)
	if out1["goroutine1_tag"] != true {
		t.Fatal("goroutine 1 result missing goroutine1_tag")
	}
	if _, present := out1["goroutine2_tag"]; present {
		t.Fatal("cross-contamination: goroutine 1 has goroutine2_tag")
	}
	var out2 map[string]interface{}
	json.Unmarshal(result2, &out2)
	if out2["goroutine2_tag"] != true {
		t.Fatal("goroutine 2 result missing goroutine2_tag")
	}
	if _, present := out2["goroutine1_tag"]; present {
		t.Fatal("cross-contamination: goroutine 2 has goroutine1_tag")
	}
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
				t.Errorf("PushScope failed: %v", err)
				return
			}
			defer PopScope(handle)
			blockMsg := "blocked in scope 1"
			err = ScopeRegisterToolConditionalExecutionGuardrail(handle.UUID(), "block_guard", 1, func(name string, args json.RawMessage) *string { return &blockMsg })
			if err != nil {
				t.Errorf("ScopeRegister failed: %v", err)
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
				t.Errorf("PushScope failed: %v", err)
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
	if !strings.Contains(err1.Error(), "guardrail rejected") {
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
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}
		result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"data": 1}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
		err := ScopeRegisterToolExecutionIntercept(handle.UUID(), "scope_exec_int", 1, func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(args)
			if err != nil {
				return nil, err
			}
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["exec_intercepted"] = true
			out, _ := json.Marshal(m)
			return out, nil
		})
		if err != nil {
			t.Fatalf("ScopeRegisterToolExecutionIntercept failed: %v", err)
		}
		result, err := ToolCallExecute("exec_int_tool", json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{"original": true}`), nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
			t.Fatalf("ScopeRegisterSubscriber failed: %v", err)
		}
		child, _ := PushScope("child_scope", ScopeTypeFunction)
		PopScope(child)
		PopScope(handle)
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
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		err = ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, "explicit_guard")
		if err != nil {
			t.Fatalf("ScopeDeregisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		result, err := ToolCallExecute("after_dereg_tool", json.RawMessage(`{"test": true}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}
		result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"input": 1}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}

		result, err := ToolCallExecute("merge_tool", json.RawMessage(`{"input": "data"}`), func(args json.RawMessage) (json.RawMessage, error) { return args, nil })
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
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
			if event.Kind() == "LLMStart" {
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
		if !strings.Contains(err.Error(), "guardrail rejected") {
			t.Fatalf("expected 'guardrail rejected', got: %v", err)
		}
	})
}

func TestScopeLocalExplicitDeregisterToolWrappers(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("tool_deregister_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()

		sanitizeResponseCalls := 0
		err = ScopeRegisterToolSanitizeResponseGuardrail(scopeUUID, "tool_scope_san_resp", 1,
			func(name string, result json.RawMessage) json.RawMessage {
				sanitizeResponseCalls++
				return result
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeResponseGuardrail failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_san_resp_call", json.RawMessage(`{"value":1}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute with sanitize response failed: %v", err)
		}
		if sanitizeResponseCalls != 1 {
			t.Fatalf("expected sanitize response callback once, got %d", sanitizeResponseCalls)
		}
		if err := ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, "tool_scope_san_resp"); err != nil {
			t.Fatalf("ScopeDeregisterToolSanitizeResponseGuardrail failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_san_resp_after", json.RawMessage(`{"value":2}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute after sanitize response deregister failed: %v", err)
		}
		if sanitizeResponseCalls != 1 {
			t.Fatalf("sanitize response callback still fired after deregister: %d", sanitizeResponseCalls)
		}

		conditionalCalls := 0
		err = ScopeRegisterToolConditionalExecutionGuardrail(scopeUUID, "tool_scope_cond", 1,
			func(name string, args json.RawMessage) *string {
				conditionalCalls++
				return nil
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolConditionalExecutionGuardrail failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_cond_call", json.RawMessage(`{"value":3}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute with conditional guardrail failed: %v", err)
		}
		if conditionalCalls != 1 {
			t.Fatalf("expected conditional callback once, got %d", conditionalCalls)
		}
		if err := ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, "tool_scope_cond"); err != nil {
			t.Fatalf("ScopeDeregisterToolConditionalExecutionGuardrail failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_cond_after", json.RawMessage(`{"value":4}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute after conditional deregister failed: %v", err)
		}
		if conditionalCalls != 1 {
			t.Fatalf("conditional callback still fired after deregister: %d", conditionalCalls)
		}

		requestInterceptCalls := 0
		err = ScopeRegisterToolRequestIntercept(scopeUUID, "tool_scope_req_int", 1, false,
			func(name string, args json.RawMessage) json.RawMessage {
				requestInterceptCalls++
				return args
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_req_int_call", json.RawMessage(`{"value":5}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute with request intercept failed: %v", err)
		}
		if requestInterceptCalls != 1 {
			t.Fatalf("expected request intercept once, got %d", requestInterceptCalls)
		}
		if err := ScopeDeregisterToolRequestIntercept(scopeUUID, "tool_scope_req_int"); err != nil {
			t.Fatalf("ScopeDeregisterToolRequestIntercept failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_req_int_after", json.RawMessage(`{"value":6}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute after request intercept deregister failed: %v", err)
		}
		if requestInterceptCalls != 1 {
			t.Fatalf("request intercept still fired after deregister: %d", requestInterceptCalls)
		}

		executionInterceptCalls := 0
		err = ScopeRegisterToolExecutionIntercept(scopeUUID, "tool_scope_exec_int", 1,
			func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				executionInterceptCalls++
				return next(args)
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolExecutionIntercept failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_exec_int_call", json.RawMessage(`{"value":7}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute with execution intercept failed: %v", err)
		}
		if executionInterceptCalls != 1 {
			t.Fatalf("expected execution intercept once, got %d", executionInterceptCalls)
		}
		if err := ScopeDeregisterToolExecutionIntercept(scopeUUID, "tool_scope_exec_int"); err != nil {
			t.Fatalf("ScopeDeregisterToolExecutionIntercept failed: %v", err)
		}
		if _, err := ToolCallExecute("tool_scope_exec_int_after", json.RawMessage(`{"value":8}`),
			func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
		); err != nil {
			t.Fatalf("ToolCallExecute after execution intercept deregister failed: %v", err)
		}
		if executionInterceptCalls != 1 {
			t.Fatalf("execution intercept still fired after deregister: %d", executionInterceptCalls)
		}
	})
}

func TestScopeLocalExplicitDeregisterLlmWrappers(t *testing.T) {
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("llm_deregister_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()
		request := makeRequest()

		sanitizeRequestCalls := 0
		err = ScopeRegisterLlmSanitizeRequestGuardrail(scopeUUID, "llm_scope_san_req", 1,
			func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
				sanitizeRequestCalls++
				return headers, content
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterLlmSanitizeRequestGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_san_req_call", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute with sanitize request failed: %v", err)
		}
		if sanitizeRequestCalls != 1 {
			t.Fatalf("expected sanitize request callback once, got %d", sanitizeRequestCalls)
		}
		if err := ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, "llm_scope_san_req"); err != nil {
			t.Fatalf("ScopeDeregisterLlmSanitizeRequestGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_san_req_after", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute after sanitize request deregister failed: %v", err)
		}
		if sanitizeRequestCalls != 1 {
			t.Fatalf("sanitize request callback still fired after deregister: %d", sanitizeRequestCalls)
		}

		sanitizeResponseCalls := 0
		err = ScopeRegisterLlmSanitizeResponseGuardrail(scopeUUID, "llm_scope_san_resp", 1,
			func(responseJSON json.RawMessage) json.RawMessage {
				sanitizeResponseCalls++
				return responseJSON
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterLlmSanitizeResponseGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_san_resp_call", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute with sanitize response failed: %v", err)
		}
		if sanitizeResponseCalls != 1 {
			t.Fatalf("expected sanitize response callback once, got %d", sanitizeResponseCalls)
		}
		if err := ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, "llm_scope_san_resp"); err != nil {
			t.Fatalf("ScopeDeregisterLlmSanitizeResponseGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_san_resp_after", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute after sanitize response deregister failed: %v", err)
		}
		if sanitizeResponseCalls != 1 {
			t.Fatalf("sanitize response callback still fired after deregister: %d", sanitizeResponseCalls)
		}

		conditionalCalls := 0
		err = ScopeRegisterLlmConditionalExecutionGuardrail(scopeUUID, "llm_scope_cond", 1,
			func(headers, content json.RawMessage) *string {
				conditionalCalls++
				return nil
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterLlmConditionalExecutionGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_cond_call", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute with conditional guardrail failed: %v", err)
		}
		if conditionalCalls != 1 {
			t.Fatalf("expected conditional callback once, got %d", conditionalCalls)
		}
		if err := ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, "llm_scope_cond"); err != nil {
			t.Fatalf("ScopeDeregisterLlmConditionalExecutionGuardrail failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_cond_after", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute after conditional deregister failed: %v", err)
		}
		if conditionalCalls != 1 {
			t.Fatalf("conditional callback still fired after deregister: %d", conditionalCalls)
		}

		requestInterceptCalls := 0
		err = ScopeRegisterLlmRequestIntercept(scopeUUID, "llm_scope_req_int", 1, false,
			func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
				requestInterceptCalls++
				return headers, content
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterLlmRequestIntercept failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_req_int_call", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute with request intercept failed: %v", err)
		}
		if requestInterceptCalls != 1 {
			t.Fatalf("expected request intercept once, got %d", requestInterceptCalls)
		}
		if err := ScopeDeregisterLlmRequestIntercept(scopeUUID, "llm_scope_req_int"); err != nil {
			t.Fatalf("ScopeDeregisterLlmRequestIntercept failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_req_int_after", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute after request intercept deregister failed: %v", err)
		}
		if requestInterceptCalls != 1 {
			t.Fatalf("request intercept still fired after deregister: %d", requestInterceptCalls)
		}

		executionInterceptCalls := 0
		err = ScopeRegisterLlmExecutionIntercept(scopeUUID, "llm_scope_exec_int", 1,
			func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				executionInterceptCalls++
				return next(requestJSON)
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterLlmExecutionIntercept failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_exec_int_call", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute with execution intercept failed: %v", err)
		}
		if executionInterceptCalls != 1 {
			t.Fatalf("expected execution intercept once, got %d", executionInterceptCalls)
		}
		if err := ScopeDeregisterLlmExecutionIntercept(scopeUUID, "llm_scope_exec_int"); err != nil {
			t.Fatalf("ScopeDeregisterLlmExecutionIntercept failed: %v", err)
		}
		if _, err := LlmCallExecute("llm_scope_exec_int_after", request,
			func(nativeJSON json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"ok":true}`), nil
			},
		); err != nil {
			t.Fatalf("LlmCallExecute after execution intercept deregister failed: %v", err)
		}
		if executionInterceptCalls != 1 {
			t.Fatalf("execution intercept still fired after deregister: %d", executionInterceptCalls)
		}

		err = ScopeRegisterLlmStreamExecutionIntercept(scopeUUID, "llm_scope_stream_int", 1,
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
		stream.Close()
		if err := ScopeDeregisterLlmStreamExecutionIntercept(scopeUUID, "llm_scope_stream_int"); err != nil {
			t.Fatalf("ScopeDeregisterLlmStreamExecutionIntercept failed: %v", err)
		}

		subscriberCalls := 0
		err = ScopeRegisterSubscriber(scopeUUID, "llm_scope_sub", func(event Event) {
			subscriberCalls++
		})
		if err != nil {
			t.Fatalf("ScopeRegisterSubscriber failed: %v", err)
		}
		if err := EmitEvent("llm_scope_sub_event_before"); err != nil {
			t.Fatalf("EmitEvent before subscriber deregister failed: %v", err)
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
		if subscriberCalls != callsAfterDeregister {
			t.Fatalf("scope-local subscriber still fired after deregister: %d -> %d", callsAfterDeregister, subscriberCalls)
		}
	})
}
