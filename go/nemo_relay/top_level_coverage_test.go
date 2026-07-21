// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"unsafe"
)

type coveragePlugin struct {
	validate func(map[string]any) ([]ConfigDiagnostic, error)
	register func(map[string]any, *PluginContext) error
}

const (
	coverageModelName     = "coverage-model"
	coverageInterceptName = "coverage-intercept"
)

func (p coveragePlugin) Validate(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
	return p.validate(pluginConfig)
}

func (p coveragePlugin) Register(pluginConfig map[string]any, ctx *PluginContext) error {
	return p.register(pluginConfig, ctx)
}

func TestTopLevelNemoRelayCoverage(t *testing.T) {
	runWithTestScopeStack(t, func() {
		assertScopeInputOutputCoverage(t)
	})
	assertLlmStreamExecutionCoverage(t)
	assertCheckedValueFailureCoverage(t)
	assertCheckedJSONStringFailureCoverage(t)
	assertNilLLMRequestWrapperCoverage(t)
	assertOpenTelemetryMarshalFailureCoverage(t)
	assertOpenInferenceMarshalFailureCoverage(t)
}

func TestTopLevelCallbacksCoverage(t *testing.T) {
	assertRegisterClosurePanicCoverage(t)

	request := newCoverageRequest(t)
	assertCodecDecodeCoverage(t, request)
	assertCodecEncodeCoverage(t, request)
	assertLlmRequestInterceptPayloadCoverage(t, request)
	assertPluginValidateCoverage(t)
	assertPluginRegisterCoverage(t)
	assertLlmCodecInterceptCoverage(t)
}

func assertScopeInputOutputCoverage(t *testing.T) {
	t.Helper()

	scope, err := PushScope(
		"coverage_input_output_scope",
		ScopeTypeAgent,
		WithInput(json.RawMessage(`{"step":"start"}`)),
	)
	if err != nil {
		t.Fatalf("PushScope with input failed: %v", err)
	}
	if err := PopScope(scope, WithOutput(json.RawMessage(`{"step":"end"}`))); err != nil {
		t.Fatalf("PopScope with output failed: %v", err)
	}
}

func assertLlmStreamExecutionCoverage(t *testing.T) {
	t.Helper()

	stream, err := LlmStreamCallExecute(
		"coverage_stream_opts",
		map[string]any{
			"headers": map[string]any{},
			"content": map[string]any{"model": coverageModelName},
		},
		func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`"data: [DONE]\n\n"`), nil
		},
		nil,
		nil,
		WithLLMModelName(coverageModelName),
	)
	if err != nil {
		t.Fatalf("LlmStreamCallExecute with opts failed: %v", err)
	}
	defer stream.Close()

	drainLlmStream(t, stream)
}

func drainLlmStream(t *testing.T, stream *LlmStream) {
	t.Helper()

	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("expected stream iteration to succeed, got %v", err)
		}
	}
}

func assertCheckedValueFailureCoverage(t *testing.T) {
	t.Helper()

	setLastErrorMessage("forced checkedValue failure")
	_, err := checkedValue(5, 1)
	assertErrorContains(t, err, "forced checkedValue failure", "checkedValue")
}

func assertCheckedJSONStringFailureCoverage(t *testing.T) {
	t.Helper()

	setLastErrorMessage("forced checkedJSONString failure")
	_, err := checkedJSONString(5, func() string { return "{}" }, noOpCheckedJSONStringFree)
	assertErrorContains(t, err, "forced checkedJSONString failure", "checkedJSONString")
}

func noOpCheckedJSONStringFree() {
	// checkedJSONString always defers the cleanup callback, even when the raw
	// string in this test is Go-owned and does not require freeing.
}

func assertNilLLMRequestWrapperCoverage(t *testing.T) {
	t.Helper()

	if got := newLLMRequestFromPtr(nil); got != nil {
		t.Fatalf("expected nil request wrapper, got %#v", got)
	}
}

func assertOpenTelemetryMarshalFailureCoverage(t *testing.T) {
	t.Helper()

	oldMarshal := jsonMarshal
	t.Cleanup(func() { jsonMarshal = oldMarshal })

	jsonMarshal = func(v any) ([]byte, error) {
		return nil, errors.New("forced otel headers marshal failure")
	}
	_, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{})
	assertErrorContains(t, err, "forced otel headers marshal failure", "OpenTelemetry header marshal")

	callCount := 0
	jsonMarshal = func(v any) ([]byte, error) {
		callCount++
		if callCount == 2 {
			return nil, errors.New("forced otel resource marshal failure")
		}
		return oldMarshal(v)
	}
	_, err = NewOpenTelemetrySubscriber(OpenTelemetryConfig{})
	assertErrorContains(t, err, "forced otel resource marshal failure", "OpenTelemetry resource marshal")
}

func assertOpenInferenceMarshalFailureCoverage(t *testing.T) {
	t.Helper()

	oldMarshal := jsonMarshal
	t.Cleanup(func() { jsonMarshal = oldMarshal })

	jsonMarshal = func(v any) ([]byte, error) {
		return nil, errors.New("forced openinference headers marshal failure")
	}
	_, err := NewOpenInferenceSubscriber(OpenInferenceConfig{})
	assertErrorContains(t, err, "forced openinference headers marshal failure", "OpenInference header marshal")

	callCount := 0
	jsonMarshal = func(v any) ([]byte, error) {
		callCount++
		if callCount == 2 {
			return nil, errors.New("forced openinference resource marshal failure")
		}
		return oldMarshal(v)
	}
	_, err = NewOpenInferenceSubscriber(OpenInferenceConfig{})
	assertErrorContains(t, err, "forced openinference resource marshal failure", "OpenInference resource marshal")
}

func assertRegisterClosurePanicCoverage(t *testing.T) {
	t.Helper()

	oldAlloc := closureTokenAlloc
	defer func() { closureTokenAlloc = oldAlloc }()

	closureTokenAlloc = func() unsafe.Pointer { return nil }
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected registerClosure to panic on nil token allocation")
			}
		}()
		_ = registerClosure("panic")
	}()
}

func newCoverageRequest(t *testing.T) *LLMRequest {
	t.Helper()

	request := NewLLMRequest(
		map[string]any{"x-test": "coverage"},
		map[string]any{"model": coverageModelName},
	)
	if request == nil {
		t.Fatal("expected non-nil request")
	}
	return request
}

func assertCodecDecodeCoverage(t *testing.T, request *LLMRequest) {
	t.Helper()

	decodeCodec := &CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("forced decode failure")
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return annotatedJSON, nil
		},
	}
	_, err := codecDecodeResultForTest(decodeCodec, request)
	assertErrorContains(t, err, "forced decode failure", "codec decode")

	successDecodeCodec := &CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return contentJSON, nil
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return annotatedJSON, nil
		},
	}
	if got, err := codecDecodeResultForTest(successDecodeCodec, request); err != nil || string(got) != string(request.Content()) {
		t.Fatalf("expected codec decode success, got %s %v", got, err)
	}
}

func assertCodecEncodeCoverage(t *testing.T, request *LLMRequest) {
	t.Helper()

	encodeCodec := &CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return contentJSON, nil
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("forced encode failure")
		},
	}
	_, err := codecEncodeResultForTest(encodeCodec, json.RawMessage(`{"annotated":true}`), request)
	assertErrorContains(t, err, "forced encode failure", "codec encode")

	successEncodeCodec := &CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return contentJSON, nil
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return originalContentJSON, nil
		},
	}
	if got, err := codecEncodeResultForTest(successEncodeCodec, json.RawMessage(`{"annotated":true}`), request); err != nil || string(got) != string(request.Content()) {
		t.Fatalf("expected codec encode success, got %s %v", got, err)
	}
}

func assertLlmRequestInterceptPayloadCoverage(t *testing.T, request *LLMRequest) {
	t.Helper()

	outcome, err := llmRequestInterceptPayload(
		func(name string, request LLMRequestDTO, annotatedJSON json.RawMessage) (LLMRequestInterceptOutcome, error) {
			if name != coverageInterceptName {
				t.Fatalf("unexpected intercept name: %q", name)
			}
			if string(annotatedJSON) != `{"annotated":true}` {
				t.Fatalf("unexpected annotated payload: %s", annotatedJSON)
			}
			request.Headers = json.RawMessage(`{"updated":"headers"}`)
			request.Content = json.RawMessage(`{"model":"updated"}`)
			return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: json.RawMessage(`{"annotated":"updated"}`)}, nil
		},
		coverageInterceptName,
		request.Headers(),
		request.Content(),
		json.RawMessage(`{"annotated":true}`),
	)
	if err != nil {
		t.Fatalf("expected intercept success, got %v", err)
	}
	if string(outcome.Request.Headers) != `{"updated":"headers"}` {
		t.Fatalf("unexpected output headers: %s", outcome.Request.Headers)
	}
	if string(outcome.Request.Content) != `{"model":"updated"}` {
		t.Fatalf("unexpected output content: %s", outcome.Request.Content)
	}
	if string(outcome.AnnotatedRequest) != `{"annotated":"updated"}` {
		t.Fatalf("unexpected output annotated json: %s", outcome.AnnotatedRequest)
	}

	_, err = llmRequestInterceptPayload(
		func(name string, request LLMRequestDTO, annotatedJSON json.RawMessage) (LLMRequestInterceptOutcome, error) {
			if annotatedJSON != nil {
				t.Fatalf("expected nil annotated JSON, got %s", annotatedJSON)
			}
			return LLMRequestInterceptOutcome{}, errors.New("forced intercept failure")
		},
		coverageInterceptName,
		request.Headers(),
		request.Content(),
		nil,
	)
	assertErrorContains(t, err, "forced intercept failure", "LLM request intercept")
}

func assertPluginValidateCoverage(t *testing.T) {
	t.Helper()

	if _, err := pluginValidateResultForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) { return nil, nil },
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, json.RawMessage("{")); err == nil {
		t.Fatal("expected invalid JSON validation error")
	}

	_, err := pluginValidateResultForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
			return nil, errors.New("forced validate failure")
		},
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, nil)
	assertErrorContains(t, err, "forced validate failure", "plugin validate")

	oldMarshal := jsonMarshal
	defer func() { jsonMarshal = oldMarshal }()
	jsonMarshal = func(v any) ([]byte, error) {
		switch v.(type) {
		case []ConfigDiagnostic:
			return nil, errors.New("forced diagnostic marshal failure")
		default:
			return oldMarshal(v)
		}
	}
	_, err = pluginValidateResultForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) {
			return []ConfigDiagnostic{{Level: DiagnosticLevelWarning, Code: "warn", Message: "warn"}}, nil
		},
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, nil)
	assertErrorContains(t, err, "forced diagnostic marshal failure", "diagnostic marshal")
	jsonMarshal = oldMarshal

	payload, err := pluginValidateResultForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) { return nil, nil },
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, nil)
	if err != nil {
		t.Fatalf("expected nil-diagnostics validation success, got %v", err)
	}
	if got := string(payload); got != "[]" {
		t.Fatalf("expected empty diagnostics payload, got %q", got)
	}
}

func assertPluginRegisterCoverage(t *testing.T) {
	t.Helper()

	if pluginRegisterErrorForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) { return nil, nil },
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, json.RawMessage("{"), nil) == nil {
		t.Fatal("expected invalid JSON register error")
	}

	err := pluginRegisterErrorForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) { return nil, nil },
		register: func(pluginConfig map[string]any, ctx *PluginContext) error {
			return errors.New("forced register failure")
		},
	}, nil, nil)
	assertErrorContains(t, err, "forced register failure", "plugin register")

	if err := pluginRegisterErrorForTest(coveragePlugin{
		validate: func(pluginConfig map[string]any) ([]ConfigDiagnostic, error) { return nil, nil },
		register: func(pluginConfig map[string]any, ctx *PluginContext) error { return nil },
	}, nil, nil); err != nil {
		t.Fatalf("expected register success, got %v", err)
	}
}

func assertLlmCodecInterceptCoverage(t *testing.T) {
	t.Helper()

	requestCodec := CodecFunc{
		Decode: func(headersJSON, contentJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"messages":[{"role":"user","content":"decoded"}],"model":"decoded-model"}`), nil
		},
		Encode: func(annotatedJSON json.RawMessage, originalHeadersJSON, originalContentJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"messages":[{"role":"user","content":"encoded"}],"model":"encoded-model"}`), nil
		},
	}

	if err := RegisterLlmRequestIntercept("coverage_llm_codec_success", 1, false, func(name string, request LLMRequestDTO, annotatedJSON json.RawMessage) (LLMRequestInterceptOutcome, error) {
		return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: json.RawMessage(`{"messages":[{"role":"user","content":"updated"}],"model":"updated-model"}`)}, nil
	}); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept success case failed: %v", err)
	}
	defer DeregisterLlmRequestIntercept("coverage_llm_codec_success")
	if _, err := LlmCallExecute("coverage_llm_codec_success", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"model": coverageModelName},
	}, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"content":"ok"}`), nil
	}, WithLLMCodec(requestCodec)); err != nil {
		t.Fatalf("expected codec-backed intercept success, got %v", err)
	}

	if err := RegisterLlmRequestIntercept("coverage_llm_codec_raw_content", 1, false, func(name string, request LLMRequestDTO, annotatedJSON json.RawMessage) (LLMRequestInterceptOutcome, error) {
		request.Content = json.RawMessage(`{"model":"raw-model-edit"}`)
		return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotatedJSON}, nil
	}); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept raw content case failed: %v", err)
	}
	t.Cleanup(func() {
		if err := DeregisterLlmRequestIntercept("coverage_llm_codec_raw_content"); err != nil {
			t.Errorf("failed to deregister raw content intercept: %v", err)
		}
	})
	providerCalled := false
	_, err := LlmCallExecute("coverage_llm_codec_raw_content", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"model": coverageModelName},
	}, func(json.RawMessage) (json.RawMessage, error) {
		providerCalled = true
		return json.RawMessage(`{"content":"unexpected"}`), nil
	}, WithLLMCodec(requestCodec))
	assertErrorContains(t, err, "request.content", "codec-backed raw content mutation")
	if providerCalled {
		t.Fatal("provider should not run after a codec-backed raw content mutation")
	}

	if err := RegisterLlmRequestIntercept("coverage_llm_codec_error", 1, false, func(name string, request LLMRequestDTO, annotatedJSON json.RawMessage) (LLMRequestInterceptOutcome, error) {
		return LLMRequestInterceptOutcome{}, errors.New("forced codec-backed intercept failure")
	}); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept error case failed: %v", err)
	}
	defer DeregisterLlmRequestIntercept("coverage_llm_codec_error")
	_, err = LlmCallExecute("coverage_llm_codec_error", map[string]any{
		"headers": map[string]any{},
		"content": map[string]any{"model": coverageModelName},
	}, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"content":"ok"}`), nil
	}, WithLLMCodec(requestCodec))
	assertErrorContains(t, err, "forced codec-backed intercept failure", "codec-backed intercept")
}

func assertErrorContains(t *testing.T, err error, want, context string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("expected %s error containing %q, got %v", context, want, err)
	}
}

func TestTopLevelPluginCoverage(t *testing.T) {
	oldValidate := validatePluginConfigJSON
	oldInitialize := initializePluginsJSON
	oldActive := activePluginReportJSON
	oldKinds := listPluginKindsJSON
	defer func() {
		validatePluginConfigJSON = oldValidate
		initializePluginsJSON = oldInitialize
		activePluginReportJSON = oldActive
		listPluginKindsJSON = oldKinds
	}()

	validatePluginConfigJSON = func(config PluginConfig) (string, error) { return "{", nil }
	if _, err := ValidatePluginConfig(NewPluginConfig()); err == nil {
		t.Fatal("expected ValidatePluginConfig unmarshal error")
	}

	initializePluginsJSON = func(config PluginConfig) (string, error) { return "{", nil }
	if _, err := InitializePlugins(NewPluginConfig()); err == nil {
		t.Fatal("expected InitializePlugins unmarshal error")
	}

	activePluginReportJSON = func() (string, error) { return "{", nil }
	if _, err := ActivePluginReport(); err == nil {
		t.Fatal("expected ActivePluginReport unmarshal error")
	}
	activePluginReportJSON = func() (string, error) { return "", errors.New("forced active report error") }
	if _, err := ActivePluginReport(); err == nil || !strings.Contains(err.Error(), "forced active report error") {
		t.Fatalf("expected ActivePluginReport passthrough error, got %v", err)
	}

	listPluginKindsJSON = func() (string, error) { return "{", nil }
	if _, err := ListPluginKinds(); err == nil {
		t.Fatal("expected ListPluginKinds unmarshal error")
	}
	listPluginKindsJSON = func() (string, error) { return "", errors.New("forced plugin kinds error") }
	if _, err := ListPluginKinds(); err == nil || !strings.Contains(err.Error(), "forced plugin kinds error") {
		t.Fatalf("expected ListPluginKinds passthrough error, got %v", err)
	}
}

func TestTopLevelStreamCoverage(t *testing.T) {
	setLastErrorMessage("forced stream next failure")
	if _, err := llmStreamNextResult(-1, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "forced stream next failure") {
		t.Fatalf("expected llmStreamNextResult error, got %v", err)
	}
}
