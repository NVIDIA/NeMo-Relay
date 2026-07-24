// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

func toolExecutionOutcome(result json.RawMessage, err error) (ToolExecutionInterceptOutcome, error) {
	return ToolExecutionInterceptOutcome{Result: result}, err
}

func TestRegisterAndUnregisterClosure(t *testing.T) {
	fn := ToolExecutionFunc(func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	})

	userData := registerClosure(fn)
	if userData == nil {
		t.Fatal("registerClosure returned nil")
	}

	if lookupClosure(userData) == nil {
		t.Fatal("lookupClosure returned nil before unregister")
	}

	id := closureID(userData)
	unregisterClosure(userData)

	closureRegistryMu.Lock()
	_, exists := closureRegistry[id]
	closureRegistryMu.Unlock()
	if exists {
		t.Fatal("closure registry still contains callback after unregister")
	}
}

func TestLlmSanitizeContextPreservesEveryCodecIdentity(t *testing.T) {
	openAIChat := "openai_chat"
	openAIResponses := "openai_responses"
	anthropicMessages := "anthropic_messages"
	runtimeCodec := "com.example.chat.v1"

	cases := []struct {
		name string
		kind uint32
		id   *string
		want LLMCodecKind
	}{
		{"none", 0, nil, LLMCodecNone},
		{"openai chat", 1, &openAIChat, LLMCodecBuiltin},
		{"openai responses", 1, &openAIResponses, LLMCodecBuiltin},
		{"anthropic messages", 1, &anthropicMessages, LLMCodecBuiltin},
		{"runtime", 2, &runtimeCodec, LLMCodecRuntime},
		{"opaque", 3, nil, LLMCodecOpaque},
		{"unknown", 99, nil, LLMCodecOpaque},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			context := llmSanitizeContext(test.kind, test.id)
			if context.Codec.CodecKind != test.want {
				t.Fatalf("codec kind = %q, want %q", context.Codec.CodecKind, test.want)
			}
			if context.Codec.CodecID == nil && test.id != nil {
				t.Fatal("codec ID was lost")
			}
			if context.Codec.CodecID != nil && test.id != nil && *context.Codec.CodecID != *test.id {
				t.Fatalf("codec ID = %q, want %q", *context.Codec.CodecID, *test.id)
			}
		})
	}
}
