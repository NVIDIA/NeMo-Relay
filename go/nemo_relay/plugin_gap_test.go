// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

func TestPluginConfigSerializationErrorsSurfaceBeforeFFI(t *testing.T) {
	config := PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			{
				Kind:    "go.invalid.plugin",
				Enabled: true,
				Config: map[string]any{
					"unsupported": make(chan int),
				},
			},
		},
	}

	if cConfig, err := pluginConfigCString(config); err == nil {
		t.Fatalf("expected pluginConfigCString serialization error, got %v", cConfig)
	}

	if _, err := ValidatePluginConfig(config); err == nil {
		t.Fatal("expected ValidatePluginConfig serialization error")
	}

	if _, err := InitializePlugins(config); err == nil {
		t.Fatal("expected InitializePlugins serialization error")
	}
}

func TestClosedPluginContextRejectsEveryRegistrationSurface(t *testing.T) {
	ctx := &PluginContext{}
	request := func(LLMRequestDTO, LLMSanitizeRequestContext) (LLMRequestDTO, bool) {
		return LLMRequestDTO{}, false
	}
	response := func(json.RawMessage, LLMSanitizeResponseContext) (json.RawMessage, bool) {
		return nil, false
	}

	for _, test := range []struct {
		name string
		call func() error
	}{
		{name: "subscriber", call: func() error { return ctx.RegisterSubscriber("closed_subscriber", nil) }},
		{name: "event metadata injector", call: func() error {
			return ctx.RegisterEventMetadataInjector("closed_event_metadata", 0, nil)
		}},
		{name: "mark sanitizer", call: func() error { return ctx.RegisterMarkSanitizeGuardrail("closed_mark", 0, nil) }},
		{name: "scope-start sanitizer", call: func() error { return ctx.RegisterScopeSanitizeStartGuardrail("closed_scope_start", 0, nil) }},
		{name: "scope-end sanitizer", call: func() error { return ctx.RegisterScopeSanitizeEndGuardrail("closed_scope_end", 0, nil) }},
		{name: "tool request sanitizer", call: func() error { return ctx.RegisterToolSanitizeRequestGuardrail("closed_tool_request_sanitize", 0, nil) }},
		{name: "tool response sanitizer", call: func() error {
			return ctx.RegisterToolSanitizeResponseGuardrail("closed_tool_response_sanitize", 0, nil)
		}},
		{name: "tool conditional", call: func() error { return ctx.RegisterToolConditionalExecutionGuardrail("closed_tool_conditional", 0, nil) }},
		{name: "llm request sanitizer", call: func() error {
			return ctx.RegisterLlmSanitizeRequestGuardrail("closed_llm_request_sanitize", 0, request)
		}},
		{name: "llm response sanitizer", call: func() error {
			return ctx.RegisterLlmSanitizeResponseGuardrail("closed_llm_response_sanitize", 0, response)
		}},
		{name: "llm conditional", call: func() error { return ctx.RegisterLlmConditionalExecutionGuardrail("closed_llm_conditional", 0, nil) }},
		{name: "llm request intercept", call: func() error { return ctx.RegisterLlmRequestIntercept("closed_llm_request_intercept", 0, false, nil) }},
		{name: "tool request intercept", call: func() error { return ctx.RegisterToolRequestIntercept("closed_tool_request_intercept", 0, false, nil) }},
		{name: "llm execution intercept", call: func() error { return ctx.RegisterLlmExecutionIntercept("closed_llm_execution_intercept", 0, nil) }},
		{name: "llm stream execution intercept", call: func() error {
			return ctx.RegisterLlmStreamExecutionIntercept("closed_llm_stream_execution_intercept", 0, nil)
		}},
		{name: "tool execution intercept", call: func() error { return ctx.RegisterToolExecutionIntercept("closed_tool_execution_intercept", 0, nil) }},
	} {
		if err := test.call(); err == nil || err.Error() != errPluginContextClosed {
			t.Fatalf("expected closed context %s registration to fail, got %v", test.name, err)
		}
	}
}
