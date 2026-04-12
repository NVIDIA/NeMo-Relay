// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_flow

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestTopLevelPluginValidationAndLifecycle(t *testing.T) {
	pluginKind := "go.test.plugin"
	registerCalls := 0

	if err := RegisterPlugin(pluginKind, PluginFuncs{
		ValidateFunc: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
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
		},
		RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
			registerCalls++
			if err := ctx.RegisterSubscriber("subscriber", func(event Event) {}); err != nil {
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
			if err := ctx.RegisterLlmConditionalExecutionGuardrail(
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
			); err != nil {
				return err
			}
			if err := ctx.RegisterLlmRequestIntercept(
				"llm_request",
				7,
				false,
				func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
					var payload map[string]any
					if err := json.Unmarshal(headers, &payload); err != nil {
						return nil, nil, nil, err
					}
					payload["x-go-plugin"] = pluginKind
					out, err := json.Marshal(payload)
					if err != nil {
						return nil, nil, nil, err
					}
					return out, content, annotated, nil
				},
			); err != nil {
				return err
			}
			if err := ctx.RegisterToolRequestIntercept(
				"tool_request",
				7,
				false,
				func(name string, args json.RawMessage) json.RawMessage {
					var payload map[string]any
					_ = json.Unmarshal(args, &payload)
					payload["goToolPlugin"] = pluginKind
					out, _ := json.Marshal(payload)
					return out
				},
			); err != nil {
				return err
			}
			if err := ctx.RegisterToolExecutionIntercept(
				"tool_exec",
				7,
				func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
					resultJSON, err := next(args)
					if err != nil {
						return nil, err
					}
					var payload map[string]any
					if err := json.Unmarshal(resultJSON, &payload); err != nil {
						return nil, err
					}
					payload["goToolExecPlugin"] = pluginKind
					return json.Marshal(payload)
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
					var payload map[string]any
					if err := json.Unmarshal(responseJSON, &payload); err != nil {
						return nil, err
					}
					payload["goLlmPlugin"] = pluginKind
					return json.Marshal(payload)
				},
			)
		},
	}); err != nil {
		t.Fatalf("RegisterPlugin failed: %v", err)
	}

	streamPluginKind := pluginKind + ".stream"
	if err := RegisterPlugin(streamPluginKind, PluginFuncs{
		RegisterFunc: func(pluginConfig map[string]any, ctx *PluginContext) error {
			return ctx.RegisterLlmStreamExecutionIntercept(
				"llm_stream_exec",
				7,
				func(requestJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
					responseJSON, err := next(requestJSON)
					if err != nil {
						return nil, err
					}
					var payload map[string]any
					if err := json.Unmarshal(responseJSON, &payload); err != nil {
						return nil, err
					}
					payload["goLlmStreamPlugin"] = streamPluginKind
					return json.Marshal(payload)
				},
			)
		},
	}); err != nil {
		t.Fatalf("RegisterPlugin failed: %v", err)
	}
	defer func() {
		_ = ClearPluginConfiguration()
		_ = DeregisterPlugin(pluginKind)
		_ = DeregisterPlugin(streamPluginKind)
	}()

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

	config := NewAdaptiveConfig()
	config.AdaptiveHints = &AdaptiveHintsConfig{
		Priority:       100,
		InjectHeader:   true,
		InjectBodyPath: "nvext.agent_hints",
	}

	_, err = InitializePlugins(PluginConfig{
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
	if registerCalls != 1 {
		t.Fatalf("expected hosted plugin register to be called once, got %d", registerCalls)
	}

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

	if err := ToolConditionalExecution("blocked-tool", json.RawMessage(`{}`)); err == nil || err.Error() != "guardrail rejected: blocked tool" {
		t.Fatalf("expected blocked tool guardrail error, got %v", err)
	}
	if err := LlmConditionalExecution(json.RawMessage(`{"headers":{"blocked":true},"content":{"messages":[]}}`)); err == nil || err.Error() != "guardrail rejected: blocked llm" {
		t.Fatalf("expected blocked llm guardrail error, got %v", err)
	}

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
		t.Fatalf("RegisterPlugin failed: %v", err)
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
	if err := closed.RegisterSubscriber("subscriber", func(event Event) {}); err == nil {
		t.Fatalf("expected closed context subscriber registration to fail")
	}
	if err := closed.RegisterToolSanitizeRequestGuardrail("tool_sanitize_request", 1, func(name string, args json.RawMessage) json.RawMessage {
		return args
	}); err == nil {
		t.Fatalf("expected closed context tool sanitize request registration to fail")
	}
	if err := closed.RegisterToolSanitizeResponseGuardrail("tool_sanitize_response", 1, func(name string, args json.RawMessage) json.RawMessage {
		return args
	}); err == nil {
		t.Fatalf("expected closed context tool sanitize response registration to fail")
	}
	if err := closed.RegisterToolConditionalExecutionGuardrail("tool_conditional", 1, func(name string, args json.RawMessage) *string {
		return nil
	}); err == nil {
		t.Fatalf("expected closed context tool conditional registration to fail")
	}
	if err := closed.RegisterLlmSanitizeRequestGuardrail("llm_sanitize_request", 1, func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
		return headers, content
	}); err == nil {
		t.Fatalf("expected closed context llm sanitize request registration to fail")
	}
	if err := closed.RegisterLlmSanitizeResponseGuardrail("llm_sanitize_response", 1, func(response json.RawMessage) json.RawMessage {
		return response
	}); err == nil {
		t.Fatalf("expected closed context llm sanitize response registration to fail")
	}
	if err := closed.RegisterLlmConditionalExecutionGuardrail("llm_conditional", 1, func(headers, content json.RawMessage) *string {
		return nil
	}); err == nil {
		t.Fatalf("expected closed context llm conditional registration to fail")
	}
	if err := closed.RegisterLlmRequestIntercept("llm_request", 1, false, func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
		return headers, content, annotated, nil
	}); err == nil {
		t.Fatalf("expected closed context llm request registration to fail")
	}
	if err := closed.RegisterToolRequestIntercept("tool_request", 1, false, func(name string, args json.RawMessage) json.RawMessage {
		return args
	}); err == nil {
		t.Fatalf("expected closed context tool request registration to fail")
	}
	if err := closed.RegisterLlmExecutionIntercept("llm_exec", 1, func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		return next(request)
	}); err == nil {
		t.Fatalf("expected closed context llm execution registration to fail")
	}
	if err := closed.RegisterLlmStreamExecutionIntercept("llm_stream_exec", 1, func(request json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		return next(request)
	}); err == nil {
		t.Fatalf("expected closed context llm stream registration to fail")
	}
	if err := closed.RegisterToolExecutionIntercept("tool_exec", 1, func(args json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
		return next(args)
	}); err == nil {
		t.Fatalf("expected closed context tool execution registration to fail")
	}
}
