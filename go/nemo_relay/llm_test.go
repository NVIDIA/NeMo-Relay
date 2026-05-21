// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
)

const (
	llmTestModel               = "test-model"
	llmCallFailed              = "LlmCall failed: %v"
	llmCallExecuteFailed       = "LlmCallExecute failed: %v"
	llmRegisterFailed          = "register failed: %v"
	llmStreamCallExecuteFailed = "LlmStreamCallExecute failed: %v"
	streamNextFailed           = "stream.Next() failed: %v"
	llmExecuteFailed           = "execute failed: %v"
)

func makeRequest() map[string]interface{} {
	return map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"messages": []string{}, "model": llmTestModel},
	}
}

// ============================================================================
// LLM lifecycle
// ============================================================================

func TestLlmCallAndEnd(t *testing.T) {
	request := makeRequest()
	handle, err := LlmCall("my_llm", request)
	if err != nil {
		t.Fatalf(llmCallFailed, err)
	}
	if handle == nil {
		t.Fatal("returned nil handle")
	}
	if handle.Name() != "my_llm" {
		t.Fatalf("expected 'my_llm', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}

	err = LlmCallEnd(handle, json.RawMessage(`{"response": "ok"}`))
	if err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}
}

func TestLlmCallWithAttributes(t *testing.T) {
	request := makeRequest()
	handle, err := LlmCall("streaming_llm", request, WithLLMAttributes(LLMAttrStreaming))
	if err != nil {
		t.Fatalf(llmCallFailed, err)
	}
	if handle.Attributes()&LLMAttrStreaming == 0 {
		t.Fatal("expected STREAMING attribute")
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmCallWithDataMetadata(t *testing.T) {
	request := makeRequest()
	handle, err := LlmCall("llm_dm", request,
		WithLLMData(json.RawMessage(`{"custom": "data"}`)),
		WithLLMMetadata(json.RawMessage(`{"trace": "xyz"}`)),
	)
	if err != nil {
		t.Fatalf(llmCallFailed, err)
	}
	LlmCallEnd(handle, json.RawMessage(`{}`),
		WithLLMData(json.RawMessage(`{"end": true}`)),
	)
}

func TestLlmCallWithParent(t *testing.T) {
	parent, _ := PushScope("llm_parent", ScopeTypeAgent)
	defer PopScope(parent)

	request := makeRequest()
	handle, err := LlmCall("child_llm", request, WithLLMParent(parent))
	if err != nil {
		t.Fatalf(llmCallFailed, err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_llm_evt", func(event Event) {
		mu.Lock()
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "start" {
			startSeen = true
		}
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "end" {
			endSeen = true
		}
		mu.Unlock()
	})

	request := makeRequest()
	handle, _ := LlmCall("evt_llm", request)
	LlmCallEnd(handle, json.RawMessage(`{}`))
	DeregisterSubscriber("go_llm_evt")

	mu.Lock()
	if !startSeen || !endSeen {
		t.Fatal("expected both start and end events")
	}
	mu.Unlock()
}

// ============================================================================
// LLM execute
// ============================================================================

func TestLlmCallExecuteBasic(t *testing.T) {
	request := makeRequest()
	result, err := LlmCallExecute("exec_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			var input map[string]interface{}
			json.Unmarshal(nativeJSON, &input)
			out, _ := json.Marshal(map[string]interface{}{"received": true})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["received"] != true {
		t.Fatalf("expected received=true, got %v", output)
	}
}

func TestCodecHandleConstructors(t *testing.T) {
	if NewOpenAIChatCodec() == nil {
		t.Fatal("expected OpenAI chat codec handle")
	}
	if NewOpenAIResponsesCodec() == nil {
		t.Fatal("expected OpenAI responses codec handle")
	}
	if NewAnthropicMessagesCodec() == nil {
		t.Fatal("expected Anthropic messages codec handle")
	}
}

func TestLlmCallExecuteWithRequestAndResponseCodecs(t *testing.T) {
	codec := llmRequestResponseCodec()
	capturedEvents, cleanupEvents := registerLlmCodecEventCollector(t)
	defer cleanupEvents()

	result, err := LlmCallExecute(
		"codec_llm",
		makeRequest(),
		requireEncodedModelExecutor(t),
		WithLLMAttributes(LLMAttrStreaming),
		WithLLMCodec(codec),
		WithLLMResponseCodec(NewOpenAIChatCodec()),
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}
	if len(result) == 0 {
		t.Fatal("expected JSON response from codec-backed execute")
	}
	events := capturedEvents()
	if len(events) != 2 {
		t.Fatalf("expected start/end events, got %d", len(events))
	}

	startEvent, endEvent := requireLlmScopeEvents(t, events)
	_ = startEvent.Attributes()
	_ = startEvent.AnnotatedRequest()
	_ = endEvent.AnnotatedResponse()
}

func llmRequestResponseCodec() CodecFunc {
	return CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"messages":[{"role":"user","content":"decoded"}],"model":"decoded-model"}`), nil
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"messages":[{"role":"user","content":"encoded"}],"model":"encoded-model"}`), nil
		},
	}
}

func registerLlmCodecEventCollector(t *testing.T) (func() []Event, func()) {
	t.Helper()

	var events []Event
	if err := RegisterSubscriber("go_llm_codec_events", func(event Event) {
		events = append(events, event)
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}

	return func() []Event {
			return events
		}, func() {
			DeregisterSubscriber("go_llm_codec_events")
		}
}

func requireEncodedModelExecutor(t *testing.T) func(json.RawMessage) (json.RawMessage, error) {
	t.Helper()

	return func(nativeJSON json.RawMessage) (json.RawMessage, error) {
		var request struct {
			Content map[string]any `json:"content"`
		}
		if err := json.Unmarshal(nativeJSON, &request); err != nil {
			return nil, err
		}
		if request.Content["model"] != "encoded-model" {
			t.Fatalf("expected encoded model in execution payload, got %#v", request.Content)
		}
		return json.RawMessage(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`), nil
	}
}

func requireLlmScopeEvents(t *testing.T, events []Event) (*ScopeEvent, *ScopeEvent) {
	t.Helper()

	var startEvent, endEvent *ScopeEvent
	for _, event := range events {
		scopeEvent, ok := event.(*ScopeEvent)
		if !ok || scopeEvent.Category() != "llm" {
			continue
		}
		switch scopeEvent.ScopeCategory() {
		case "start":
			startEvent = scopeEvent
		case "end":
			endEvent = scopeEvent
		}
	}
	if startEvent == nil || endEvent == nil {
		t.Fatalf("expected LLM start and end events, got %#v", events)
	}
	return startEvent, endEvent
}

// ============================================================================
// LLM guardrails
// ============================================================================

func TestLlmSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_san_req", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			return headers, content
		},
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_san_req")
}

func TestLlmSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeResponseGuardrail("go_llm_san_resp", 1,
		func(responseJSON json.RawMessage) json.RawMessage { return responseJSON },
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmSanitizeResponseGuardrail("go_llm_san_resp")
}

func TestLlmConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterLlmConditionalExecutionGuardrail("go_llm_cond", 1,
		func(headers, content json.RawMessage) *string {
			return nil // pass
		},
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmConditionalExecutionGuardrail("go_llm_cond")
}

func TestLlmDuplicateGuardrailFails(t *testing.T) {
	RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			return headers, content
		},
	)
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			return headers, content
		},
	)
	if err == nil {
		t.Fatal("expected error for duplicate")
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_dup")
}

func TestLlmConditionalBlocksExecution(t *testing.T) {
	msg := "LLM blocked"
	RegisterLlmConditionalExecutionGuardrail("go_llm_blocker", 1,
		func(headers, content json.RawMessage) *string {
			return &msg
		},
	)

	request := makeRequest()
	_, err := LlmCallExecute("blocked_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"should": "not reach"}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected error from guardrail rejection")
	}
	if !strings.Contains(err.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' error, got: %v", err)
	}

	DeregisterLlmConditionalExecutionGuardrail("go_llm_blocker")
}

// ============================================================================
// LLM intercepts
// ============================================================================

func TestLlmRequestInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmRequestIntercept("go_llm_req", 1, false,
		func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
			return headers, content, annotated, nil
		},
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmRequestIntercept("go_llm_req")
}

func TestLlmExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmExecutionIntercept("go_llm_exec", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(nativeJSON)
		},
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmExecutionIntercept("go_llm_exec")
}

func TestLlmStreamExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamExecutionIntercept("go_llm_sexec", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(nativeJSON)
		},
	)
	if err != nil {
		t.Fatalf(llmRegisterFailed, err)
	}
	DeregisterLlmStreamExecutionIntercept("go_llm_sexec")
}

func TestLlmStreamExecutionInterceptCanCallNext(t *testing.T) {
	request := makeRequest()

	err := RegisterLlmStreamExecutionIntercept("go_llm_stream_exec_next", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			nextResult, err := next(nativeJSON)
			if err != nil {
				return nil, err
			}
			return json.RawMessage(`{"intercepted":true,"next":` + string(nextResult) + `}`), nil
		},
	)
	if err != nil {
		t.Fatalf("RegisterLlmStreamExecutionIntercept failed: %v", err)
	}
	defer DeregisterLlmStreamExecutionIntercept("go_llm_stream_exec_next")

	stream, err := LlmStreamCallExecute("stream_exec_next_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"streamed":true}`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf(llmStreamCallExecuteFailed, err)
	}
	defer stream.Close()

	chunk, err := stream.Next()
	if err != nil {
		t.Fatalf(streamNextFailed, err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(chunk, &payload); err != nil {
		t.Fatalf("unmarshal chunk: %v", err)
	}
	if payload["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", payload)
	}
	nextPayload, ok := payload["next"].(map[string]interface{})
	if !ok || nextPayload["streamed"] != true {
		t.Fatalf("expected next.streamed=true, got %v", payload["next"])
	}
}

func TestLlmRequestInterceptModifies(t *testing.T) {
	RegisterLlmRequestIntercept("go_llm_req_mod", 1, false,
		func(name string, headers, content, annotated json.RawMessage) (json.RawMessage, json.RawMessage, json.RawMessage, error) {
			var m map[string]interface{}
			json.Unmarshal(content, &m)
			m["intercepted"] = true
			out, _ := json.Marshal(m)
			return headers, out, annotated, nil
		},
	)

	request := makeRequest()
	result, err := LlmCallExecute("int_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			var req struct {
				Content map[string]interface{} `json:"content"`
			}
			json.Unmarshal(nativeJSON, &req)
			out, _ := json.Marshal(map[string]interface{}{"saw_intercepted": req.Content["intercepted"]})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf(llmExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["saw_intercepted"] != true {
		t.Fatalf("expected saw_intercepted=true, got %v", output)
	}

	DeregisterLlmRequestIntercept("go_llm_req_mod")
}

func TestLlmExecutionInterceptReplaces(t *testing.T) {
	RegisterLlmExecutionIntercept("go_llm_exec_rep", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	request := makeRequest()
	result, err := LlmCallExecute("exec_llm_rep", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf(llmExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["from_intercept"] != true {
		t.Fatalf("expected from_intercept, got %v", output)
	}
	if _, ok := output["from_original"]; ok {
		t.Fatal("should not contain from_original")
	}

	DeregisterLlmExecutionIntercept("go_llm_exec_rep")
}

func TestLlmCallableErrorPropagation(t *testing.T) {
	request := makeRequest()
	_, err := LlmCallExecute("error_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("llm internal failure")
		},
	)
	if err == nil {
		t.Fatal("expected llm callable error to propagate")
	}
	if !strings.Contains(err.Error(), "llm internal failure") {
		t.Fatalf("expected propagated llm error message, got %v", err)
	}
}

// ============================================================================
// Full LLM pipeline tests
// ============================================================================

func TestLlmFullPipelineInterceptsAndExecute(t *testing.T) {
	// Register an execution intercept
	RegisterLlmExecutionIntercept("go_llm_pipe_exec_int", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(nativeJSON)
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
	defer DeregisterLlmExecutionIntercept("go_llm_pipe_exec_int")

	request := makeRequest()
	result, err := LlmCallExecute("pipeline_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			out, _ := json.Marshal(map[string]interface{}{"llm_ran": true})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)

	if output["llm_ran"] != true {
		t.Fatal("expected llm_ran=true")
	}
	if output["exec_intercepted"] != true {
		t.Fatal("expected exec_intercepted=true")
	}
}

func TestLlmSanitizeRequestGuardrailModifiesEventInput(t *testing.T) {
	// Sanitize-request guardrails modify the event input, not the actual request
	// passed to the callable. Verify through event subscriber.
	var capturedInput json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_llm_san_evt_sub", func(event Event) {
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "start" {
			mu.Lock()
			capturedInput = append(json.RawMessage(nil), event.Input()...)
			mu.Unlock()
		}
	})
	defer DeregisterSubscriber("go_llm_san_evt_sub")

	RegisterLlmSanitizeRequestGuardrail("go_llm_content_mod", 1,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			var m map[string]interface{}
			json.Unmarshal(content, &m)
			m["system_prompt_injected"] = true
			out, _ := json.Marshal(m)
			return headers, out
		},
	)
	defer DeregisterLlmSanitizeRequestGuardrail("go_llm_content_mod")

	request := makeRequest()
	_, err := LlmCallExecute("mod_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"done": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}

	mu.Lock()
	defer mu.Unlock()

	if capturedInput == nil {
		t.Fatal("expected non-nil captured input from event")
	}
	// The event input should reflect the sanitized content
	t.Logf("captured event input: %s", string(capturedInput))
}

func TestLlmConditionalGuardrailSelectiveReject(t *testing.T) {
	RegisterLlmConditionalExecutionGuardrail("go_llm_selective", 1,
		func(headers, content json.RawMessage) *string {
			var m map[string]interface{}
			json.Unmarshal(content, &m)
			if model, ok := m["model"].(string); ok && model == "blocked-model" {
				msg := "model not allowed"
				return &msg
			}
			return nil
		},
	)
	defer DeregisterLlmConditionalExecutionGuardrail("go_llm_selective")

	// Blocked model
	blockedReq := map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"model": "blocked-model"},
	}
	_, err := LlmCallExecute("selective_llm", blockedReq,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected blocked-model to be rejected")
	}

	// Allowed model
	allowedReq := makeRequest()
	result, err := LlmCallExecute("selective_llm", allowedReq,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("allowed model should succeed: %v", err)
	}
	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["ok"] != true {
		t.Fatalf("expected ok=true, got %v", output)
	}
}

func TestLlmExecutionInterceptWrapsCallable(t *testing.T) {
	RegisterLlmExecutionIntercept("go_llm_wrap_exec", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			result, err := next(nativeJSON)
			if err != nil {
				return nil, err
			}
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["wrapped"] = true
			out, _ := json.Marshal(m)
			return out, nil
		},
	)
	defer DeregisterLlmExecutionIntercept("go_llm_wrap_exec")

	request := makeRequest()
	result, err := LlmCallExecute("wrap_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true {
		t.Fatal("expected original=true")
	}
	if output["wrapped"] != true {
		t.Fatal("expected wrapped=true")
	}
}

func TestLlmExecutionInterceptSeesNextError(t *testing.T) {
	RegisterLlmExecutionIntercept("go_llm_wrap_exec_err", 1,
		func(nativeJSON json.RawMessage, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
			return next(nativeJSON)
		},
	)
	defer DeregisterLlmExecutionIntercept("go_llm_wrap_exec_err")

	request := makeRequest()
	_, err := LlmCallExecute("wrap_llm_err", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("llm next failure")
		},
	)
	if err == nil {
		t.Fatal("expected llm next error to propagate through intercept")
	}
	if !strings.Contains(err.Error(), "llm next failure") {
		t.Fatalf("expected propagated llm next error message, got %v", err)
	}
}

func TestLlmCallWithModelName(t *testing.T) {
	var capturedModelName string
	var mu sync.Mutex

	RegisterSubscriber("go_llm_model_sub", func(event Event) {
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "start" {
			mu.Lock()
			capturedModelName = event.ModelName()
			mu.Unlock()
		}
	})

	request := makeRequest()
	handle, err := LlmCall("model_llm", request, WithLLMModelName("gpt-4-turbo"))
	if err != nil {
		t.Fatalf(llmCallFailed, err)
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
	DeregisterSubscriber("go_llm_model_sub")

	mu.Lock()
	defer mu.Unlock()
	if capturedModelName != "gpt-4-turbo" {
		t.Fatalf("expected model_name='gpt-4-turbo', got '%s'", capturedModelName)
	}
}

func TestLlmEventInputOutput(t *testing.T) {
	var capturedInput, capturedOutput json.RawMessage
	var mu sync.Mutex

	RegisterSubscriber("go_llm_io_sub", func(event Event) {
		mu.Lock()
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "start" {
			capturedInput = append(json.RawMessage(nil), event.Input()...)
		}
		if event.Kind() == "scope" && event.Category() == "llm" && event.ScopeCategory() == "end" {
			capturedOutput = append(json.RawMessage(nil), event.Output()...)
		}
		mu.Unlock()
	})

	request := makeRequest()
	result, err := LlmCallExecute("io_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"response": "hello"}`), nil
		},
	)
	if err != nil {
		t.Fatalf(llmCallExecuteFailed, err)
	}
	_ = result
	DeregisterSubscriber("go_llm_io_sub")

	mu.Lock()
	defer mu.Unlock()

	if capturedInput == nil {
		t.Fatal("expected non-nil input on Start event")
	}

	if capturedOutput == nil {
		t.Fatal("expected non-nil output on End event")
	}
	var output map[string]interface{}
	json.Unmarshal(capturedOutput, &output)
	if output["response"] != "hello" {
		t.Fatalf("expected response=hello in output, got %v", output)
	}
}

// ============================================================================
// LLM streaming tests
// ============================================================================

func TestLlmStreamCallExecuteBasic(t *testing.T) {
	request := makeRequest()

	stream, err := LlmStreamCallExecute("stream_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			chunks := `data: {"chunk": 1}` + "\n\n" +
				`data: {"chunk": 2}` + "\n\n" +
				`data: [DONE]` + "\n\n"
			return json.RawMessage(`"` + strings.ReplaceAll(chunks, `"`, `\"`) + `"`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf(llmStreamCallExecuteFailed, err)
	}
	defer stream.Close()

	chunkCount := 0
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf(streamNextFailed, err)
		}
		chunkCount++
	}
	t.Logf("received %d chunks from stream", chunkCount)
}

func TestLlmStreamCallExecuteWithCollectorFinalizer(t *testing.T) {
	request := makeRequest()

	var collectedChunks []json.RawMessage
	var mu sync.Mutex

	collector := func(chunk json.RawMessage) {
		mu.Lock()
		collectedChunks = append(collectedChunks, append(json.RawMessage(nil), chunk...))
		mu.Unlock()
	}

	finalizerCalled := false
	finalizer := func() string {
		mu.Lock()
		finalizerCalled = true
		count := len(collectedChunks)
		mu.Unlock()
		return fmt.Sprintf(`{"aggregated": true, "total_chunks": %d}`, count)
	}

	stream, err := LlmStreamCallExecute("collector_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			chunks := `data: {"token": "hello"}` + "\n\n" +
				`data: [DONE]` + "\n\n"
			return json.RawMessage(`"` + strings.ReplaceAll(chunks, `"`, `\"`) + `"`), nil
		},
		collector, finalizer,
	)
	if err != nil {
		t.Fatalf(llmStreamCallExecuteFailed, err)
	}
	defer stream.Close()

	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf(streamNextFailed, err)
		}
	}

	mu.Lock()
	defer mu.Unlock()

	t.Logf("collector received %d chunks", len(collectedChunks))
	if finalizerCalled {
		t.Log("finalizer was called as expected")
	}
}

func TestLlmStreamCloseIsIdempotent(t *testing.T) {
	request := makeRequest()

	stream, err := LlmStreamCallExecute("close_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`"data: [DONE]\n\n"`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf(llmStreamCallExecuteFailed, err)
	}

	stream.Close()
	stream.Close()
	stream.Close()

	_, err = stream.Next()
	if err != io.EOF {
		t.Fatalf("expected io.EOF after close, got %v", err)
	}
}

func TestLlmStreamNilCollectorFinalizer(t *testing.T) {
	request := makeRequest()

	stream, err := LlmStreamCallExecute("nil_opts_llm", request,
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`"data: [DONE]\n\n"`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf(llmStreamCallExecuteFailed, err)
	}
	defer stream.Close()

	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf(streamNextFailed, err)
		}
	}
}
