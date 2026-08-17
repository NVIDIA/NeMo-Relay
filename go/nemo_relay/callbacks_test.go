// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func TestLLMSanitizeCodecInvocationInvalidationWaitsForInflightCall(t *testing.T) {
	invocation := newLLMSanitizeCodecInvocation()
	release, err := invocation.acquire()
	if err != nil {
		t.Fatalf("acquire active invocation: %v", err)
	}

	invalidated := make(chan struct{})
	go func() {
		invocation.invalidate()
		close(invalidated)
	}()

	select {
	case <-invalidated:
		t.Fatal("invalidation returned while a codec operation was in flight")
	case <-time.After(20 * time.Millisecond):
	}
	release()
	select {
	case <-invalidated:
	case <-time.After(time.Second):
		t.Fatal("invalidation did not finish after the codec operation completed")
	}
	if _, err := invocation.acquire(); !errors.Is(err, ErrLLMSanitizeCodecExpired) {
		t.Fatalf("expired invocation acquire returned %v", err)
	}
}

func toolExecutionResult(result json.RawMessage) ToolExecutionResult {
	return ToolExecutionResult{Result: result}
}

func toolExecutionOutcome(result ToolExecutionResult, err error) (ToolExecutionInterceptOutcome, error) {
	return ToolExecutionInterceptOutcome{Result: result.Result, Annotation: result.Annotation}, err
}

func TestRegisterAndUnregisterClosure(t *testing.T) {
	fn := ToolExecutionFunc(func(args json.RawMessage) (ToolExecutionResult, error) {
		return toolExecutionResult(args), nil
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

type codecIdentityTestCase struct {
	name string
	kind uint32
	id   *string
	want LLMCodecKind
}

func assertCodecIdentity(t *testing.T, test codecIdentityTestCase) {
	t.Helper()
	codec := llmCodecIdentity(test.kind, test.id)
	if codec.CodecKind != test.want {
		t.Fatalf("codec kind = %q, want %q", codec.CodecKind, test.want)
	}
	if codec.CodecID == nil && test.id != nil {
		t.Fatal("codec ID was lost")
	}
	if codec.CodecID != nil && test.id == nil {
		t.Fatalf("unexpected codec ID %q", *codec.CodecID)
	}
	if codec.CodecID != nil && test.id != nil && *codec.CodecID != *test.id {
		t.Fatalf("codec ID = %q, want %q", *codec.CodecID, *test.id)
	}
}

func TestLlmSanitizeDirectionalContextsPreserveEveryCodecIdentity(t *testing.T) {
	openAIChat := "openai_chat"
	openAIResponses := "openai_responses"
	anthropicMessages := "anthropic_messages"
	gemini := "gemini_generate_content"
	runtimeCodec := "com.example.chat.v1"

	cases := []codecIdentityTestCase{
		{"none", 0, nil, LLMCodecNone},
		{"openai chat", 1, &openAIChat, LLMCodecBuiltin},
		{"openai responses", 1, &openAIResponses, LLMCodecBuiltin},
		{"anthropic messages", 1, &anthropicMessages, LLMCodecBuiltin},
		{"gemini_generate_content", 1, &gemini, LLMCodecBuiltin},
		{"runtime", 2, &runtimeCodec, LLMCodecRuntime},
		{"opaque", 3, nil, LLMCodecOpaque},
		{"unknown", 99, nil, LLMCodecOpaque},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			assertCodecIdentity(t, test)
		})
	}
}
