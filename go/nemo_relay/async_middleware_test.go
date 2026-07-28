// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func asyncMiddlewareNoop(context.Context, json.RawMessage) (any, error) {
	return nil, nil
}

func asyncExecutionNoop(context.Context, json.RawMessage, AsyncNext) (any, error) {
	return nil, nil
}

func TestAsyncMiddlewareGlobalRegistrationParity(t *testing.T) {
	registrations := []struct {
		name       string
		register   func(string) error
		deregister func(string) error
	}{
		{"mark", func(name string) error { return RegisterMarkSanitizeGuardrailAsync(name, 0, asyncMiddlewareNoop) }, DeregisterMarkSanitizeGuardrail},
		{"scope-start", func(name string) error { return RegisterScopeSanitizeStartGuardrailAsync(name, 0, asyncMiddlewareNoop) }, DeregisterScopeSanitizeStartGuardrail},
		{"scope-end", func(name string) error { return RegisterScopeSanitizeEndGuardrailAsync(name, 0, asyncMiddlewareNoop) }, DeregisterScopeSanitizeEndGuardrail},
		{"tool-sanitize-request", func(name string) error {
			return RegisterToolSanitizeRequestGuardrailAsync(name, 0, asyncMiddlewareNoop)
		}, DeregisterToolSanitizeRequestGuardrail},
		{"tool-sanitize-response", func(name string) error {
			return RegisterToolSanitizeResponseGuardrailAsync(name, 0, asyncMiddlewareNoop)
		}, DeregisterToolSanitizeResponseGuardrail},
		{"tool-conditional", func(name string) error {
			return RegisterToolConditionalExecutionGuardrailAsync(name, 0, asyncMiddlewareNoop)
		}, DeregisterToolConditionalExecutionGuardrail},
		{"tool-request", func(name string) error { return RegisterToolRequestInterceptAsync(name, 0, false, asyncMiddlewareNoop) }, DeregisterToolRequestIntercept},
		{"tool-execution", func(name string) error { return RegisterToolExecutionInterceptAsync(name, 0, asyncExecutionNoop) }, DeregisterToolExecutionIntercept},
		{"llm-sanitize-request", func(name string) error { return RegisterLlmSanitizeRequestGuardrailAsync(name, 0, asyncMiddlewareNoop) }, DeregisterLlmSanitizeRequestGuardrail},
		{"llm-sanitize-response", func(name string) error {
			return RegisterLlmSanitizeResponseGuardrailAsync(name, 0, asyncMiddlewareNoop)
		}, DeregisterLlmSanitizeResponseGuardrail},
		{"llm-conditional", func(name string) error {
			return RegisterLlmConditionalExecutionGuardrailAsync(name, 0, asyncMiddlewareNoop)
		}, DeregisterLlmConditionalExecutionGuardrail},
		{"llm-request", func(name string) error { return RegisterLlmRequestInterceptAsync(name, 0, false, asyncMiddlewareNoop) }, DeregisterLlmRequestIntercept},
		{"llm-execution", func(name string) error { return RegisterLlmExecutionInterceptAsync(name, 0, asyncExecutionNoop) }, DeregisterLlmExecutionIntercept},
		{"llm-stream-execution", func(name string) error { return RegisterLlmStreamExecutionInterceptAsync(name, 0, asyncExecutionNoop) }, DeregisterLlmStreamExecutionIntercept},
	}

	for _, registration := range registrations {
		t.Run(registration.name, func(t *testing.T) {
			name := "go-async-global-" + registration.name
			if err := registration.register(name); err != nil {
				t.Fatalf("register: %v", err)
			}
			if err := registration.register(name); err == nil {
				t.Fatal("duplicate registration unexpectedly succeeded")
			}
			if err := registration.deregister(name); err != nil {
				t.Fatalf("deregister: %v", err)
			}
			if err := registration.deregister(name); err != nil {
				t.Fatalf("idempotent deregister: %v", err)
			}
		})
	}
}

func TestAsyncMiddlewareScopeLocalRegistrationParity(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		handle, err := PushScope("async-registration-owner", ScopeTypeAgent)
		if err != nil {
			t.Fatalf("push scope: %v", err)
		}
		defer func() {
			if err := PopScope(handle); err != nil {
				t.Fatalf("pop scope: %v", err)
			}
		}()

		scopeUUID := handle.UUID()
		registrations := []struct {
			name       string
			register   func(string) error
			deregister func(string) error
		}{
			{"mark", func(name string) error {
				return ScopeRegisterMarkSanitizeGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterMarkSanitizeGuardrail(scopeUUID, name) }},
			{"scope-start", func(name string) error {
				return ScopeRegisterScopeSanitizeStartGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterScopeSanitizeStartGuardrail(scopeUUID, name) }},
			{"scope-end", func(name string) error {
				return ScopeRegisterScopeSanitizeEndGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterScopeSanitizeEndGuardrail(scopeUUID, name) }},
			{"tool-sanitize-request", func(name string) error {
				return ScopeRegisterToolSanitizeRequestGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterToolSanitizeRequestGuardrail(scopeUUID, name) }},
			{"tool-sanitize-response", func(name string) error {
				return ScopeRegisterToolSanitizeResponseGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterToolSanitizeResponseGuardrail(scopeUUID, name) }},
			{"tool-conditional", func(name string) error {
				return ScopeRegisterToolConditionalExecutionGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterToolConditionalExecutionGuardrail(scopeUUID, name) }},
			{"tool-request", func(name string) error {
				return ScopeRegisterToolRequestInterceptAsync(scopeUUID, name, 0, false, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterToolRequestIntercept(scopeUUID, name) }},
			{"tool-execution", func(name string) error {
				return ScopeRegisterToolExecutionInterceptAsync(scopeUUID, name, 0, asyncExecutionNoop)
			}, func(name string) error { return ScopeDeregisterToolExecutionIntercept(scopeUUID, name) }},
			{"llm-sanitize-request", func(name string) error {
				return ScopeRegisterLlmSanitizeRequestGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterLlmSanitizeRequestGuardrail(scopeUUID, name) }},
			{"llm-sanitize-response", func(name string) error {
				return ScopeRegisterLlmSanitizeResponseGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterLlmSanitizeResponseGuardrail(scopeUUID, name) }},
			{"llm-conditional", func(name string) error {
				return ScopeRegisterLlmConditionalExecutionGuardrailAsync(scopeUUID, name, 0, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterLlmConditionalExecutionGuardrail(scopeUUID, name) }},
			{"llm-request", func(name string) error {
				return ScopeRegisterLlmRequestInterceptAsync(scopeUUID, name, 0, false, asyncMiddlewareNoop)
			}, func(name string) error { return ScopeDeregisterLlmRequestIntercept(scopeUUID, name) }},
			{"llm-execution", func(name string) error {
				return ScopeRegisterLlmExecutionInterceptAsync(scopeUUID, name, 0, asyncExecutionNoop)
			}, func(name string) error { return ScopeDeregisterLlmExecutionIntercept(scopeUUID, name) }},
			{"llm-stream-execution", func(name string) error {
				return ScopeRegisterLlmStreamExecutionInterceptAsync(scopeUUID, name, 0, asyncExecutionNoop)
			}, func(name string) error { return ScopeDeregisterLlmStreamExecutionIntercept(scopeUUID, name) }},
		}

		for _, registration := range registrations {
			name := "go-async-local-" + registration.name
			if err := registration.register(name); err != nil {
				t.Fatalf("register %s: %v", registration.name, err)
			}
			if err := registration.register(name); err == nil {
				t.Fatalf("duplicate registration %s unexpectedly succeeded", registration.name)
			}
			if err := registration.deregister(name); err != nil {
				t.Fatalf("deregister %s: %v", registration.name, err)
			}
			if err := registration.deregister(name); err != nil {
				t.Fatalf("idempotent deregistration %s: %v", registration.name, err)
			}
		}
	})
}

func TestAsyncToolRequestInterceptPriorityOrdering(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		register := func(name string, priority int32, marker string) {
			t.Helper()
			err := RegisterToolRequestInterceptAsync(name, priority, false,
				func(_ context.Context, invocation json.RawMessage) (any, error) {
					var envelope struct {
						Value map[string]any `json:"value"`
					}
					if err := json.Unmarshal(invocation, &envelope); err != nil {
						return nil, err
					}
					order, _ := envelope.Value["order"].(string)
					envelope.Value["order"] = order + marker
					return envelope.Value, nil
				},
			)
			if err != nil {
				t.Fatalf("register %s: %v", name, err)
			}
			t.Cleanup(func() { _ = DeregisterToolRequestIntercept(name) })
		}
		register("go-async-priority-late", 10, "B")
		register("go-async-priority-early", 0, "A")

		result, err := ToolRequestIntercepts("priority", json.RawMessage(`{"order":""}`))
		if err != nil {
			t.Fatalf("tool request intercepts: %v", err)
		}
		if string(result) != `{"order":"AB"}` {
			t.Fatalf("result = %s, want priority order AB", result)
		}
	})
}

func TestAsyncToolMiddlewareCompletionAndNext(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		if err := RegisterToolConditionalExecutionGuardrailAsync("go-async-tool-conditional", 0, asyncMiddlewareNoop); err != nil {
			t.Fatalf("register conditional: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolConditionalExecutionGuardrail("go-async-tool-conditional") })

		if err := RegisterToolExecutionInterceptAsync("go-async-tool-execution", 0,
			func(ctx context.Context, invocation json.RawMessage, next AsyncNext) (any, error) {
				var payload struct {
					Value json.RawMessage `json:"value"`
				}
				if err := json.Unmarshal(invocation, &payload); err != nil {
					return nil, err
				}
				result, err := next(ctx, payload.Value)
				if err != nil {
					return nil, err
				}
				return map[string]json.RawMessage{"result": result}, nil
			},
		); err != nil {
			t.Fatalf("register execution intercept: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolExecutionIntercept("go-async-tool-execution") })

		result, err := ToolCallExecute("go-async-tool", json.RawMessage(`{"value":1}`), func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		})
		if err != nil {
			t.Fatalf("tool call execute: %v", err)
		}
		if string(result) != `{"value":1}` {
			t.Fatalf("tool result = %s, want original result", result)
		}
	})
}

func TestAsyncToolMiddlewarePropagatesCallbackAndNextErrors(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		const conditionalName = "go-async-tool-conditional-error"
		if err := RegisterToolConditionalExecutionGuardrailAsync(conditionalName, 0,
			func(context.Context, json.RawMessage) (any, error) {
				return nil, errors.New("conditional callback failed")
			},
		); err != nil {
			t.Fatalf("register conditional: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolConditionalExecutionGuardrail(conditionalName) })

		_, err := ToolCallExecute("go-async-tool-conditional-error", json.RawMessage(`{}`), func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		})
		if err == nil || !strings.Contains(err.Error(), "conditional callback failed") {
			t.Fatalf("conditional error = %v, want callback failure", err)
		}

		if err := DeregisterToolConditionalExecutionGuardrail(conditionalName); err != nil {
			t.Fatalf("deregister conditional: %v", err)
		}
		const executionName = "go-async-tool-next-error"
		if err := RegisterToolExecutionInterceptAsync(executionName, 0,
			func(ctx context.Context, invocation json.RawMessage, next AsyncNext) (any, error) {
				var payload struct {
					Value json.RawMessage `json:"value"`
				}
				if err := json.Unmarshal(invocation, &payload); err != nil {
					return nil, err
				}
				return next(ctx, payload.Value)
			},
		); err != nil {
			t.Fatalf("register execution intercept: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolExecutionIntercept(executionName) })

		_, err = ToolCallExecute("go-async-tool-next-error", json.RawMessage(`{}`), func(json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("tool implementation failed")
		})
		if err == nil || !strings.Contains(err.Error(), "tool implementation failed") {
			t.Fatalf("next error = %v, want implementation failure", err)
		}
	})
}

func TestAsyncMiddlewarePanicsBecomeInvocationErrors(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		const conditionalName = "go-async-tool-conditional-panic"
		if err := RegisterToolConditionalExecutionGuardrailAsync(conditionalName, 0,
			func(context.Context, json.RawMessage) (any, error) {
				panic("conditional callback panicked")
			},
		); err != nil {
			t.Fatalf("register conditional: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolConditionalExecutionGuardrail(conditionalName) })

		_, err := ToolCallExecute(conditionalName, json.RawMessage(`{}`), func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		})
		if err == nil || !strings.Contains(err.Error(), "conditional callback panicked") {
			t.Fatalf("conditional error = %v, want recovered panic", err)
		}
		if err := DeregisterToolConditionalExecutionGuardrail(conditionalName); err != nil {
			t.Fatalf("deregister conditional: %v", err)
		}

		const executionName = "go-async-tool-execution-panic"
		if err := RegisterToolExecutionInterceptAsync(executionName, 0,
			func(context.Context, json.RawMessage, AsyncNext) (any, error) {
				panic("execution intercept panicked")
			},
		); err != nil {
			t.Fatalf("register execution intercept: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolExecutionIntercept(executionName) })

		_, err = ToolCallExecute(executionName, json.RawMessage(`{}`), func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		})
		if err == nil || !strings.Contains(err.Error(), "execution intercept panicked") {
			t.Fatalf("execution error = %v, want recovered panic", err)
		}
	})
}

func TestAsyncNextObservesOuterCancellationWithDetachedContext(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		const name = "go-async-detached-next"
		nextStarted := make(chan struct{})
		releaseNext := make(chan struct{})
		nextDone := make(chan struct{})
		var releaseOnce sync.Once
		release := func() {
			releaseOnce.Do(func() { close(releaseNext) })
		}
		if err := RegisterToolExecutionInterceptAsync(name, 0,
			func(_ context.Context, invocation json.RawMessage, next AsyncNext) (any, error) {
				go func() {
					defer close(nextDone)
					_, _ = next(context.Background(), invocation)
				}()
				<-nextStarted
				return nil, errors.New("intercept returned early")
			},
		); err != nil {
			t.Fatalf("register execution intercept: %v", err)
		}
		t.Cleanup(func() { _ = DeregisterToolExecutionIntercept(name) })

		result := make(chan error, 1)
		go func() {
			_, err := ToolCallExecute(name, json.RawMessage(`{}`), func(args json.RawMessage) (json.RawMessage, error) {
				close(nextStarted)
				<-releaseNext
				return args, nil
			})
			result <- err
		}()
		defer func() {
			release()
			select {
			case <-nextDone:
			case <-time.After(time.Second):
				t.Error("detached next continuation never settled during cleanup")
			}
		}()

		select {
		case err := <-result:
			if err == nil || !strings.Contains(err.Error(), "intercept returned early") {
				t.Fatalf("execution error = %v, want intercept failure", err)
			}
		case <-time.After(time.Second):
			t.Fatal("detached next context prevented intercept cleanup")
		}
		release()
		select {
		case <-nextDone:
		case <-time.After(time.Second):
			t.Fatal("detached next continuation never settled")
		}
	})
}
