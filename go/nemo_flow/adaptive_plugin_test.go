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

	if err := RegisterPlugin(pluginKind, PluginHandlerFuncs{
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
	if err := RegisterPlugin(streamPluginKind, PluginHandlerFuncs{
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

	llmResult, err := LlmCallExecute("test-model", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"messages": []any{}},
	}, func(request json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]any{"response": "ok"})
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
