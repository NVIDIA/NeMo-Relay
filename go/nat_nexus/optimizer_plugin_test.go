// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nat_nexus

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestOptimizerHostedPluginValidationAndLifecycle(t *testing.T) {
	pluginKind := "go.test.optimizer_plugin"
	registerCalls := 0

	if err := RegisterOptimizerPlugin(pluginKind, OptimizerPluginHandlerFuncs{
		ValidateFunc: func(instanceID string, pluginConfig map[string]any) ([]OptimizerConfigDiagnostic, error) {
			threshold, _ := pluginConfig["threshold"].(float64)
			field := "plugin_config.threshold"
			component := "external_component"
			return []OptimizerConfigDiagnostic{
				{
					Level:     OptimizerDiagnosticLevelWarning,
					Code:      "optimizer.go_plugin_validate",
					Component: &component,
					Field:     &field,
					Message:   fmt.Sprintf("%s:%v", instanceID, threshold),
				},
			}, nil
		},
		RegisterFunc: func(instanceID string, pluginConfig map[string]any, ctx *OptimizerPluginContext) error {
			registerCalls++
			if err := ctx.RegisterSubscriber(instanceID+".subscriber", func(event Event) {}); err != nil {
				return err
			}
			if err := ctx.RegisterToolRequestIntercept(
				instanceID+".tool_request",
				7,
				false,
				func(name string, args json.RawMessage) json.RawMessage {
					var payload map[string]any
					_ = json.Unmarshal(args, &payload)
					payload["goToolPlugin"] = instanceID
					out, _ := json.Marshal(payload)
					return out
				},
			); err != nil {
				return err
			}
			return ctx.RegisterLlmExecutionIntercept(
				instanceID+".llm_exec",
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
					payload["goLlmPlugin"] = instanceID
					return json.Marshal(payload)
				},
			)
		},
	}); err != nil {
		t.Fatalf("RegisterOptimizerPlugin failed: %v", err)
	}

	streamPluginKind := pluginKind + ".stream"
	if err := RegisterOptimizerPlugin(streamPluginKind, OptimizerPluginHandlerFuncs{
		RegisterFunc: func(instanceID string, pluginConfig map[string]any, ctx *OptimizerPluginContext) error {
			return ctx.RegisterLlmStreamExecutionIntercept(
				instanceID+".llm_stream_exec",
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
					payload["goLlmStreamPlugin"] = instanceID
					return json.Marshal(payload)
				},
			)
		},
	}); err != nil {
		t.Fatalf("RegisterOptimizerPlugin failed: %v", err)
	}
	defer func() {
		_ = DeregisterOptimizerPlugin(pluginKind)
		_ = DeregisterOptimizerPlugin(streamPluginKind)
	}()

	report, err := ValidateOptimizerConfig(OptimizerConfig{
		Version: 1,
		Components: []OptimizerComponentSpec{
			ExternalComponent(ExternalComponentConfig{
				PluginKind:   pluginKind,
				InstanceID:   "go-plugin",
				PluginConfig: map[string]any{"threshold": 7},
			}),
		},
	})
	if err != nil {
		t.Fatalf("ValidateOptimizerConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 1 {
		t.Fatalf("expected 1 diagnostic, got %#v", report.Diagnostics)
	}
	if report.Diagnostics[0].Code != "optimizer.go_plugin_validate" {
		t.Fatalf("unexpected diagnostic code: %#v", report.Diagnostics)
	}

	config := NewOptimizerConfig()
	config.State = &OptimizerStateConfig{Backend: NewInMemoryOptimizerBackend()}
	config.Components = []OptimizerComponentSpec{
		ExternalComponent(ExternalComponentConfig{
			PluginKind:   pluginKind,
			InstanceID:   "go-plugin",
			PluginConfig: map[string]any{"threshold": 7},
		}),
		ExternalComponent(ExternalComponentConfig{
			PluginKind: streamPluginKind,
			InstanceID: "go-stream-plugin",
		}),
	}

	runtime, err := NewOptimizerRuntime(config)
	if err != nil {
		t.Fatalf("NewOptimizerRuntime failed: %v", err)
	}
	defer runtime.Close()

	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
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
	if toolPayload["goToolPlugin"] != "go-plugin" {
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
	if llmPayload["goLlmPlugin"] != "go-plugin" {
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
	if streamPayload["goLlmStreamPlugin"] != "go-stream-plugin" {
		t.Fatalf("unexpected llm stream plugin value: %#v", streamPayload)
	}

	if err := runtime.Deregister(); err != nil {
		t.Fatalf("Deregister failed: %v", err)
	}
	if err := runtime.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}
