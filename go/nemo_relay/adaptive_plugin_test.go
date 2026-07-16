// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
)

const registerPluginFailed = "RegisterPlugin failed: %v"

func lifecycleDiagnostics(pluginKind string, pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
	threshold, _ := pluginConfig["threshold"].(float64)
	field := "threshold"
	component := pluginKind
	return []ConfigDiagnostic{
		{
			Level:     DiagnosticLevelWarning,
			Code:      "plugin.go_validate",
			Component: &component,
			Field:     &field,
			Message:   fmt.Sprintf("%s:%v", pluginKind, threshold),
		},
	}, nil
}

func decorateJSONPayload(payloadJSON json.RawMessage, key string, value any) (json.RawMessage, error) {
	var payload map[string]any
	if err := json.Unmarshal(payloadJSON, &payload); err != nil {
		return nil, err
	}
	payload[key] = value
	return json.Marshal(payload)
}

func registerLifecycleGuardrails(ctx *PluginContext) error {
	if err := ctx.RegisterSubscriber("subscriber", func(event Event) {
		_ = event
	}); err != nil {
		return err
	}
	if err := ctx.RegisterMarkSanitizeGuardrail("mark_sanitize", 7, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Data = json.RawMessage(`{"plugin":true}`)
		return fields
	}); err != nil {
		return err
	}
	if err := ctx.RegisterScopeSanitizeStartGuardrail("scope_start_sanitize", 7, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Metadata = json.RawMessage(`{"plugin_phase":"start"}`)
		return fields
	}); err != nil {
		return err
	}
	if err := ctx.RegisterScopeSanitizeEndGuardrail("scope_end_sanitize", 7, func(_ Event, fields EventSanitizeFields) EventSanitizeFields {
		fields.Metadata = json.RawMessage(`{"plugin_phase":"end"}`)
		return fields
	}); err != nil {
		return err
	}
	if err := ctx.RegisterToolSanitizeRequestGuardrail(
		"tool_sanitize_request",
		7,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	); err != nil {
		return err
	}
	if err := ctx.RegisterToolSanitizeResponseGuardrail(
		"tool_sanitize_response",
		7,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	); err != nil {
		return err
	}
	if err := ctx.RegisterToolConditionalExecutionGuardrail(
		"tool_conditional",
		7,
		func(name string, args json.RawMessage) *string {
			if name == "blocked-tool" {
				msg := "blocked tool"
				return &msg
			}
			return nil
		},
	); err != nil {
		return err
	}
	if err := ctx.RegisterLlmSanitizeRequestGuardrail(
		"llm_sanitize_request",
		7,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			return headers, content
		},
	); err != nil {
		return err
	}
	if err := ctx.RegisterLlmSanitizeResponseGuardrail(
		"llm_sanitize_response",
		7,
		func(responseJSON json.RawMessage) json.RawMessage { return responseJSON },
	); err != nil {
		return err
	}
	return ctx.RegisterLlmConditionalExecutionGuardrail(
		"llm_conditional",
		7,
		func(headers, content json.RawMessage) *string {
			var payload map[string]any
			_ = json.Unmarshal(headers, &payload)
			if payload["blocked"] == true {
				msg := "blocked llm"
				return &msg
			}
			return nil
		},
	)
}

func registerLifecycleInterceptors(ctx *PluginContext, pluginKind string) error {
	if err := ctx.RegisterLlmRequestIntercept(
		"llm_request",
		7,
		false,
		func(name string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
			out, err := decorateJSONPayload(request.Headers, "x-go-plugin", pluginKind)
			if err != nil {
				return LLMRequestInterceptOutcome{}, err
			}
			request.Headers = out
			return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotated}, nil
		},
	); err != nil {
		return err
	}
	if err := ctx.RegisterToolRequestIntercept(
		"tool_request",
		7,
		false,
		func(name string, args json.RawMessage) json.RawMessage {
			out, _ := decorateJSONPayload(args, "goToolPlugin", pluginKind)
			return out
		},
	); err != nil {
		return err
	}
	if err := ctx.RegisterToolExecutionIntercept(
		"tool_exec",
		7,
		func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
			resultJSON, err := next(args)
			if err != nil {
				return ToolExecutionInterceptOutcome{}, err
			}
			result, err := decorateJSONPayload(resultJSON, "goToolExecPlugin", pluginKind)
			return ToolExecutionInterceptOutcome{Result: result}, err
		},
	); err != nil {
		return err
	}
	return ctx.RegisterLlmExecutionIntercept(
		"llm_exec",
		7,
		func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			responseJSON, err := next(requestJSON)
			if err != nil {
				return nil, err
			}
			return decorateJSONPayload(responseJSON, "goLlmPlugin", pluginKind)
		},
	)
}

func registerLifecyclePlugin(pluginKind string, registerCalls *int) error {
	return RegisterPlugin(pluginKind, PluginFuncs{
		ValidateFunc: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
			return lifecycleDiagnostics(pluginKind, pluginConfig)
		},
		RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
			*registerCalls = *registerCalls + 1
			if err := registerLifecycleGuardrails(ctx); err != nil {
				return err
			}
			return registerLifecycleInterceptors(ctx, pluginKind)
		},
	})
}

func registerLifecycleStreamPlugin(streamPluginKind string) error {
	return RegisterPlugin(streamPluginKind, PluginFuncs{
		RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
			return ctx.RegisterLlmStreamExecutionIntercept(
				"llm_stream_exec",
				7,
				func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
					responseJSON, err := next(requestJSON)
					if err != nil {
						return nil, err
					}
					return decorateJSONPayload(responseJSON, "goLlmStreamPlugin", streamPluginKind)
				},
			)
		},
	})
}

func assertToolPluginPayload(t *testing.T, pluginKind string) {
	t.Helper()
	toolResult, err := ToolCallExecute("search", json.RawMessage(`{"query":"test"}`), func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	})
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}
	var toolPayload map[string]any
	if err := json.Unmarshal(toolResult, &toolPayload); err != nil {
		t.Fatalf("tool result unmarshal failed: %v", err)
	}
	if toolPayload["goToolPlugin"] != pluginKind {
		t.Fatalf("unexpected tool plugin value: %#v", toolPayload)
	}
	if toolPayload["goToolExecPlugin"] != pluginKind {
		t.Fatalf("unexpected tool exec plugin value: %#v", toolPayload)
	}
}

func assertLlmPluginPayload(t *testing.T, pluginKind string) {
	t.Helper()
	llmResult, err := LlmCallExecute("test-model", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"messages": []any{}},
	}, func(request json.RawMessage) (json.RawMessage, error) {
		var payload struct {
			Headers map[string]any `json:"headers"`
		}
		if err := json.Unmarshal(request, &payload); err != nil {
			return nil, err
		}
		return json.Marshal(map[string]any{
			"response":   "ok",
			"seenHeader": payload.Headers["x-go-plugin"],
		})
	})
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}
	var llmPayload map[string]any
	if err := json.Unmarshal(llmResult, &llmPayload); err != nil {
		t.Fatalf("llm result unmarshal failed: %v", err)
	}
	if llmPayload["goLlmPlugin"] != pluginKind {
		t.Fatalf("unexpected llm plugin value: %#v", llmPayload)
	}
	if llmPayload["seenHeader"] != pluginKind {
		t.Fatalf("unexpected llm request intercept header: %#v", llmPayload)
	}
}

func assertGuardrailError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || err.Error() != want {
		t.Fatalf("expected %q, got %v", want, err)
	}
}

func assertStreamPluginPayload(t *testing.T, streamPluginKind string) {
	t.Helper()
	stream, err := LlmStreamCallExecute("test-stream-model", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"messages": []any{}},
	}, func(request json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]any{"response": "ok"})
	}, nil, func() string {
		return `{"final":true}`
	})
	if err != nil {
		t.Fatalf("LlmStreamCallExecute failed: %v", err)
	}
	defer stream.Close()

	chunk, err := stream.Next()
	if err != nil {
		t.Fatalf("stream.Next failed: %v", err)
	}
	var streamPayload map[string]any
	if err := json.Unmarshal(chunk, &streamPayload); err != nil {
		t.Fatalf("stream result unmarshal failed: %v", err)
	}
	if streamPayload["goLlmStreamPlugin"] != streamPluginKind {
		t.Fatalf("unexpected llm stream plugin value: %#v", streamPayload)
	}
}

func assertClosedContextRegistrationFails(t *testing.T, name string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected closed context %s registration to fail", name)
	}
}

func TestTopLevelPluginValidationAndLifecycle(t *testing.T) {
	runTestWithScopeStack(t, testTopLevelPluginValidationAndLifecycle)
}

func testTopLevelPluginValidationAndLifecycle(t *testing.T) {
	pluginKind := "go.test.plugin"
	registerCalls := 0

	if err := registerLifecyclePlugin(pluginKind, &registerCalls); err != nil {
		t.Fatalf(registerPluginFailed, err)
	}

	streamPluginKind := pluginKind + ".stream"
	if err := registerLifecycleStreamPlugin(streamPluginKind); err != nil {
		t.Fatalf(registerPluginFailed, err)
	}
	defer func() {
		_ = ClearPluginConfiguration()
		_ = DeregisterPlugin(pluginKind)
		_ = DeregisterPlugin(streamPluginKind)
	}()
	validateLifecyclePluginConfig(t, pluginKind)
	initializeLifecyclePlugins(t, pluginKind, streamPluginKind, &registerCalls)
	assertToolPluginPayload(t, pluginKind)
	assertLlmPluginPayload(t, pluginKind)
	assertGuardrailError(t, ToolConditionalExecution("blocked-tool", json.RawMessage(`{}`)), "guardrail rejected: blocked tool")
	assertGuardrailError(t, LlmConditionalExecution(json.RawMessage(`{"headers":{"blocked":true},"content":{"messages":[]}}`)), "guardrail rejected: blocked llm")
	assertStreamPluginPayload(t, streamPluginKind)
	exercisePluginEventSanitizers(t)
}

func validateLifecyclePluginConfig(t *testing.T, pluginKind string) {
	t.Helper()
	report, err := ValidatePluginConfig(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			{
				Kind:    pluginKind,
				Enabled: true,
				Config:  map[string]any{"threshold": 7},
			},
		},
	})
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 1 {
		t.Fatalf("expected 1 diagnostic, got %#v", report.Diagnostics)
	}
	if report.Diagnostics[0].Code != "plugin.go_validate" {
		t.Fatalf("unexpected diagnostic code: %#v", report.Diagnostics)
	}
}

func initializeLifecyclePlugins(t *testing.T, pluginKind, streamPluginKind string, registerCalls *int) {
	t.Helper()
	config := NewAdaptiveConfig()
	config.AdaptiveHints = &AdaptiveHintsConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}

	_, err := InitializePlugins(PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			AdaptiveComponent(config),
			{
				Kind:    pluginKind,
				Enabled: true,
				Config:  map[string]any{"threshold": 7},
			},
			{
				Kind:    streamPluginKind,
				Enabled: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}
	if *registerCalls != 1 {
		t.Fatalf("expected plugin register to be called once, got %d", *registerCalls)
	}
}

func exercisePluginEventSanitizers(t *testing.T) {
	t.Helper()
	var mu sync.Mutex
	var pluginEvents []Event
	if err := RegisterSubscriber("go-plugin-event-sanitize-test", func(event Event) {
		mu.Lock()
		pluginEvents = append(pluginEvents, event)
		mu.Unlock()
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer DeregisterSubscriber("go-plugin-event-sanitize-test")
	if err := EmitEvent("plugin-mark", WithEventData(json.RawMessage(`{"raw":true}`))); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	handle, err := PushScope("plugin-generic-scope", ScopeTypeCustom)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if err := PopScope(handle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	if err := EmitEvent("plugin-mark-after-clear", WithEventData(json.RawMessage(`{"raw":true}`))); err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
	if err := FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	assertPluginSanitizerEvents(t, pluginEvents)
}

func assertPluginSanitizerEvents(t *testing.T, pluginEvents []Event) {
	t.Helper()
	var phases []string
	for _, event := range pluginEvents {
		switch event.Name() {
		case "plugin-mark":
			if string(event.Data()) != `{"plugin":true}` {
				t.Fatalf("plugin mark sanitizer did not run: %s", event.Data())
			}
		case "plugin-mark-after-clear":
			if string(event.Data()) != `{"raw":true}` {
				t.Fatalf("plugin mark sanitizer was not rolled back: %s", event.Data())
			}
		case "plugin-generic-scope":
			phases = append(phases, string(event.Metadata()))
		}
	}
	if len(phases) != 2 || phases[0] != `{"plugin_phase":"start"}` || phases[1] != `{"plugin_phase":"end"}` {
		t.Fatalf("unexpected plugin scope sanitizer phases: %v", phases)
	}
}

func TestPluginHelperConstructorsAndRegistryListing(t *testing.T) {
	config := NewPluginConfig()
	if config.Version != 1 || len(config.Components) != 0 {
		t.Fatalf("unexpected plugin config defaults: %#v", config)
	}

	component := NewPluginComponent("go.test.component")
	if component.Kind != "go.test.component" || !component.Enabled {
		t.Fatalf("unexpected plugin component defaults: %#v", component)
	}
	if len(component.Config) != 0 {
		t.Fatalf("expected empty default config, got %#v", component.Config)
	}

	pluginKind := "go.test.list_kinds"
	if err := RegisterPlugin(pluginKind, PluginFuncs{RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
		return nil
	}}); err != nil {
		t.Fatalf(registerPluginFailed, err)
	}
	defer DeregisterPlugin(pluginKind)

	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds failed: %v", err)
	}
	found := false
	for _, kind := range kinds {
		if kind == pluginKind {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected %q in registered kinds: %#v", pluginKind, kinds)
	}
}

func TestPluginFuncsAndClosedContextBranches(t *testing.T) {
	var funcs PluginFuncs
	diagnostics, err := funcs.Validate(map[string]any{"ignored": true})
	if err != nil {
		t.Fatalf("Validate should allow nil callback: %v", err)
	}
	if diagnostics != nil {
		t.Fatalf("expected nil diagnostics for nil validate callback, got %#v", diagnostics)
	}
	if err := funcs.Register(map[string]any{"ignored": true}, nil); err != nil {
		t.Fatalf("Register should allow nil callback: %v", err)
	}

	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration failed: %v", err)
	}
	report, err := ActivePluginReport()
	if err != nil {
		t.Fatalf("ActivePluginReport failed: %v", err)
	}
	if report != nil {
		t.Fatalf("expected nil active plugin report, got %#v", report)
	}

	closed := &PluginContext{}
	cases := []struct {
		name     string
		register func() error
	}{
		{"subscriber", func() error {
			return closed.RegisterSubscriber("subscriber", func(event Event) { _ = event })
		}},
		{"mark sanitizer", func() error {
			return closed.RegisterMarkSanitizeGuardrail("mark", 1, func(_ Event, fields EventSanitizeFields) EventSanitizeFields { return fields })
		}},
		{"scope start sanitizer", func() error {
			return closed.RegisterScopeSanitizeStartGuardrail("scope_start", 1, func(_ Event, fields EventSanitizeFields) EventSanitizeFields { return fields })
		}},
		{"scope end sanitizer", func() error {
			return closed.RegisterScopeSanitizeEndGuardrail("scope_end", 1, func(_ Event, fields EventSanitizeFields) EventSanitizeFields { return fields })
		}},
		{"tool sanitize request", func() error {
			return closed.RegisterToolSanitizeRequestGuardrail("tool_sanitize_request", 1, func(name string, args json.RawMessage) json.RawMessage { return args })
		}},
		{"tool sanitize response", func() error {
			return closed.RegisterToolSanitizeResponseGuardrail("tool_sanitize_response", 1, func(name string, args json.RawMessage) json.RawMessage { return args })
		}},
		{"tool conditional", func() error {
			return closed.RegisterToolConditionalExecutionGuardrail("tool_conditional", 1, func(name string, args json.RawMessage) *string { return nil })
		}},
		{"llm sanitize request", func() error {
			return closed.RegisterLlmSanitizeRequestGuardrail("llm_sanitize_request", 1, func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) { return headers, content })
		}},
		{"llm sanitize response", func() error {
			return closed.RegisterLlmSanitizeResponseGuardrail("llm_sanitize_response", 1, func(response json.RawMessage) json.RawMessage { return response })
		}},
		{"llm conditional", func() error {
			return closed.RegisterLlmConditionalExecutionGuardrail("llm_conditional", 1, func(headers, content json.RawMessage) *string { return nil })
		}},
		{"llm request", func() error {
			return closed.RegisterLlmRequestIntercept("llm_request", 1, false, func(name string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
				return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotated}, nil
			})
		}},
		{"tool request", func() error {
			return closed.RegisterToolRequestIntercept("tool_request", 1, false, func(name string, args json.RawMessage) json.RawMessage { return args })
		}},
		{"llm execution", func() error {
			return closed.RegisterLlmExecutionIntercept("llm_exec", 1, func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				return next(request)
			})
		}},
		{"llm stream", func() error {
			return closed.RegisterLlmStreamExecutionIntercept("llm_stream_exec", 1, func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				return next(request)
			})
		}},
		{"tool execution", func() error {
			return closed.RegisterToolExecutionIntercept("tool_exec", 1, func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (ToolExecutionInterceptOutcome, error) {
				return toolExecutionOutcome(next(args))
			})
		}},
	}
	for _, tc := range cases {
		assertClosedContextRegistrationFails(t, tc.name, tc.register())
	}
}
