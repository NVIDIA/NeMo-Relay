// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
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
		// Register a global subscriber to capture events
		type capturedEvent struct {
			eventType EventType
			input     json.RawMessage
		}
		var events []capturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_req_sub", func(e *Event) {
			mu.Lock()
			events = append(events, capturedEvent{
				eventType: e.Type(),
				input:     append(json.RawMessage(nil), e.Input()...),
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

		// Verify the sanitize guardrail affected the Start event's input
		mu.Lock()
		defer mu.Unlock()
		var found bool
		for _, ev := range events {
			if ev.eventType == EventTypeStart && ev.input != nil {
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
		// Register a global subscriber to capture events
		type capturedEvent struct {
			eventType EventType
			output    json.RawMessage
		}
		var events []capturedEvent
		var mu sync.Mutex
		err := RegisterSubscriber("scope_san_resp_sub", func(e *Event) {
			mu.Lock()
			events = append(events, capturedEvent{
				eventType: e.Type(),
				output:    append(json.RawMessage(nil), e.Output()...),
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

		// Verify the sanitize guardrail affected the End event's output
		mu.Lock()
		defer mu.Unlock()
		var found bool
		for _, ev := range events {
			if ev.eventType == EventTypeEnd && ev.output != nil {
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
		// Push a scope and register a scope-local guardrail that marks args
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

		// Pop the scope -- this should clean up the scope-local guardrail
		err = PopScope(handle)
		if err != nil {
			t.Fatalf("PopScope failed: %v", err)
		}

		// Execute a tool call; the guardrail should no longer be active
		result, err := ToolCallExecute("after_pop_tool", json.RawMessage(`{"original": true}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
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

		// Pop to trigger cleanup
		PopScope(handle)

		result, err := ToolCallExecute("after_intercept_pop", json.RawMessage(`{"check": true}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
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
		err = ScopeRegisterSubscriber(handle.UUID(), "cleanup_sub",
			func(event *Event) {
				mu.Lock()
				eventCount++
				mu.Unlock()
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterSubscriber failed: %v", err)
		}

		// Pop the scope to clean up the subscriber
		PopScope(handle)

		mu.Lock()
		countAfterPop := eventCount
		mu.Unlock()

		// Emit events after pop; the subscriber should no longer fire
		EmitEvent("after_pop_event")

		mu.Lock()
		countAfterEmit := eventCount
		mu.Unlock()

		if countAfterEmit != countAfterPop {
			t.Fatalf("scope-local subscriber should not fire after pop; count went from %d to %d",
				countAfterPop, countAfterEmit)
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
		// Register a global guardrail at priority 10
		var order []string
		var mu sync.Mutex

		err := RegisterToolSanitizeRequestGuardrail("global_priority_guard", 10,
			func(name string, args json.RawMessage) json.RawMessage {
				mu.Lock()
				order = append(order, "global_p10")
				mu.Unlock()
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["global"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf("RegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		defer DeregisterToolSanitizeRequestGuardrail("global_priority_guard")

		handle, err := PushScope("priority_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		// Register a scope-local guardrail at priority 5 (runs before global p10)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_priority_guard", 5,
			func(name string, args json.RawMessage) json.RawMessage {
				mu.Lock()
				order = append(order, "scope_p5")
				mu.Unlock()
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["scope_local"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}

		_, err = ToolCallExecute("priority_tool", json.RawMessage(`{"input": true}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
		if err != nil {
			t.Fatalf("ToolCallExecute failed: %v", err)
		}

		// Verify execution order: lower priority number runs first
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

		// Global at priority 1 (runs first)
		err := RegisterToolSanitizeRequestGuardrail("global_first", 1,
			func(name string, args json.RawMessage) json.RawMessage {
				mu.Lock()
				order = append(order, "global_p1")
				mu.Unlock()
				return args
			},
		)
		if err != nil {
			t.Fatalf("RegisterToolSanitizeRequestGuardrail failed: %v", err)
		}
		defer DeregisterToolSanitizeRequestGuardrail("global_first")

		handle, err := PushScope("priority_order_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		// Scope-local at priority 20 (runs second)
		err = ScopeRegisterToolSanitizeRequestGuardrail(handle.UUID(), "scope_second", 20,
			func(name string, args json.RawMessage) json.RawMessage {
				mu.Lock()
				order = append(order, "scope_p20")
				mu.Unlock()
				return args
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}

		_, err = ToolCallExecute("order_tool", json.RawMessage(`{}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
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
	stack1, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 1 failed: %v", err)
	}
	defer stack1.Close()

	stack2, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 2 failed: %v", err)
	}
	defer stack2.Close()

	var wg sync.WaitGroup
	var result1, result2 json.RawMessage
	var err1, err2 error

	wg.Add(2)

	// Goroutine 1: scope-local request intercept adds "goroutine1_tag"
	go func() {
		defer wg.Done()
		stack1.Run(func() {
			handle, err := PushScope("iso_scope_1", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope in goroutine 1 failed: %v", err)
				return
			}
			defer PopScope(handle)

			err = ScopeRegisterToolRequestIntercept(handle.UUID(), "iso_intercept_1", 1, false,
				func(name string, args json.RawMessage) json.RawMessage {
					var m map[string]interface{}
					json.Unmarshal(args, &m)
					m["goroutine1_tag"] = true
					result, _ := json.Marshal(m)
					return result
				},
			)
			if err != nil {
				t.Errorf("ScopeRegister in goroutine 1 failed: %v", err)
				return
			}

			result1, err1 = ToolCallExecute("iso_tool_1", json.RawMessage(`{"source": "g1"}`),
				func(args json.RawMessage) (json.RawMessage, error) {
					return args, nil
				},
			)
		})
	}()

	// Goroutine 2: scope-local request intercept adds "goroutine2_tag"
	go func() {
		defer wg.Done()
		stack2.Run(func() {
			handle, err := PushScope("iso_scope_2", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope in goroutine 2 failed: %v", err)
				return
			}
			defer PopScope(handle)

			err = ScopeRegisterToolRequestIntercept(handle.UUID(), "iso_intercept_2", 1, false,
				func(name string, args json.RawMessage) json.RawMessage {
					var m map[string]interface{}
					json.Unmarshal(args, &m)
					m["goroutine2_tag"] = true
					result, _ := json.Marshal(m)
					return result
				},
			)
			if err != nil {
				t.Errorf("ScopeRegister in goroutine 2 failed: %v", err)
				return
			}

			result2, err2 = ToolCallExecute("iso_tool_2", json.RawMessage(`{"source": "g2"}`),
				func(args json.RawMessage) (json.RawMessage, error) {
					return args, nil
				},
			)
		})
	}()

	wg.Wait()

	if err1 != nil {
		t.Fatalf("goroutine 1 ToolCallExecute failed: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("goroutine 2 ToolCallExecute failed: %v", err2)
	}

	// Verify goroutine 1 result: should have goroutine1_tag but NOT goroutine2_tag
	var out1 map[string]interface{}
	json.Unmarshal(result1, &out1)
	if out1["goroutine1_tag"] != true {
		t.Fatal("goroutine 1 result missing goroutine1_tag")
	}
	if _, present := out1["goroutine2_tag"]; present {
		t.Fatal("goroutine 1 result should not have goroutine2_tag (cross-contamination)")
	}

	// Verify goroutine 2 result: should have goroutine2_tag but NOT goroutine1_tag
	var out2 map[string]interface{}
	json.Unmarshal(result2, &out2)
	if out2["goroutine2_tag"] != true {
		t.Fatal("goroutine 2 result missing goroutine2_tag")
	}
	if _, present := out2["goroutine1_tag"]; present {
		t.Fatal("goroutine 2 result should not have goroutine1_tag (cross-contamination)")
	}
}

func TestScopeLocalConditionalGuardrailIsolation(t *testing.T) {
	stack1, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 1 failed: %v", err)
	}
	defer stack1.Close()

	stack2, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack 2 failed: %v", err)
	}
	defer stack2.Close()

	var wg sync.WaitGroup
	var err1, err2 error
	var result2 json.RawMessage

	wg.Add(2)

	// Goroutine 1: scope-local conditional guardrail blocks all tool calls
	go func() {
		defer wg.Done()
		stack1.Run(func() {
			handle, err := PushScope("block_scope", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope in goroutine 1 failed: %v", err)
				return
			}
			defer PopScope(handle)

			blockMsg := "blocked in scope 1"
			err = ScopeRegisterToolConditionalExecutionGuardrail(handle.UUID(), "block_guard", 1,
				func(name string, args json.RawMessage) *string {
					return &blockMsg
				},
			)
			if err != nil {
				t.Errorf("ScopeRegister in goroutine 1 failed: %v", err)
				return
			}

			_, err1 = ToolCallExecute("blocked_tool", json.RawMessage(`{}`),
				func(args json.RawMessage) (json.RawMessage, error) {
					return json.RawMessage(`{"reached": true}`), nil
				},
			)
		})
	}()

	// Goroutine 2: no blocking guardrail, tool calls should succeed
	go func() {
		defer wg.Done()
		stack2.Run(func() {
			handle, err := PushScope("allow_scope", ScopeTypeAgent)
			if err != nil {
				t.Errorf("PushScope in goroutine 2 failed: %v", err)
				return
			}
			defer PopScope(handle)

			result2, err2 = ToolCallExecute("allowed_tool", json.RawMessage(`{"ok": true}`),
				func(args json.RawMessage) (json.RawMessage, error) {
					return args, nil
				},
			)
		})
	}()

	wg.Wait()

	// Goroutine 1 should have been blocked
	if err1 == nil {
		t.Fatal("expected goroutine 1 to be blocked by scope-local conditional guardrail")
	}
	if !strings.Contains(err1.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' error in goroutine 1, got: %v", err1)
	}

	// Goroutine 2 should have succeeded
	if err2 != nil {
		t.Fatalf("goroutine 2 ToolCallExecute should succeed, got: %v", err2)
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
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("req_intercept_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		err = ScopeRegisterToolRequestIntercept(handle.UUID(), "scope_req_int", 1, false,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["scope_intercepted"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolRequestIntercept failed: %v", err)
		}

		result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"data": 1}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
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
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("exec_intercept_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		err = ScopeRegisterToolExecutionIntercept(handle.UUID(), "scope_exec_int", 1,
			func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				// Call original, then augment result
				result, err := next(args)
				if err != nil {
					return nil, err
				}
				var m map[string]interface{}
				json.Unmarshal(result, &m)
				m["exec_intercepted"] = true
				out, _ := json.Marshal(m)
				return out, nil
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolExecutionIntercept failed: %v", err)
		}

		result, err := ToolCallExecute("exec_int_tool", json.RawMessage(`{}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(`{"original": true}`), nil
			},
		)
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
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("sub_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}

		var eventNames []string
		var mu sync.Mutex
		err = ScopeRegisterSubscriber(handle.UUID(), "scope_sub",
			func(event *Event) {
				mu.Lock()
				eventNames = append(eventNames, event.Name())
				mu.Unlock()
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterSubscriber failed: %v", err)
		}

		// Push/pop a child scope to generate events
		child, _ := PushScope("child_scope", ScopeTypeFunction)
		PopScope(child)

		// Pop the parent (this also cleans up the subscriber, but events
		// from the pop itself may or may not be delivered)
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
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("explicit_dereg_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "explicit_guard", 1,
			func(name string, args json.RawMessage) json.RawMessage {
				var m map[string]interface{}
				json.Unmarshal(args, &m)
				m["should_not_appear"] = true
				result, _ := json.Marshal(m)
				return result
			},
		)
		if err != nil {
			t.Fatalf("ScopeRegisterToolSanitizeRequestGuardrail failed: %v", err)
		}

		// Explicitly deregister before pop
		err = ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, "explicit_guard")
		if err != nil {
			t.Fatalf("ScopeDeregisterToolSanitizeRequestGuardrail failed: %v", err)
		}

		result, err := ToolCallExecute("after_dereg_tool", json.RawMessage(`{"test": true}`),
			func(args json.RawMessage) (json.RawMessage, error) {
				return args, nil
			},
		)
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
	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(func() {
		handle, err := PushScope("dup_scope", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("PushScope failed: %v", err)
		}
		defer PopScope(handle)

		scopeUUID := handle.UUID()
		guardFn := func(name string, args json.RawMessage) json.RawMessage { return args }

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "dup_guard", 1, guardFn)
		if err != nil {
			t.Fatalf("first registration should succeed: %v", err)
		}

		err = ScopeRegisterToolSanitizeRequestGuardrail(scopeUUID, "dup_guard", 1, guardFn)
		if err == nil {
			t.Fatal("expected error for duplicate scope-local guardrail registration")
		}
	})
}
