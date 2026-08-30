// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

type failingAtofSink struct{}

func (failingAtofSink) atofExporterSink() {
	// This marker method intentionally has no runtime behavior.
}

func (failingAtofSink) MarshalJSON() ([]byte, error) {
	return nil, errors.New("forced ATOF sink marshal failure")
}

func TestEventBaseNilPointerFallbacks(t *testing.T) {
	event := eventBase{}

	assertEmptyEventString(t, "UUID", event.UUID())
	assertEmptyEventString(t, "Name", event.Name())
	assertEmptyEventString(t, "Kind", event.Kind())
	assertEmptyEventString(t, "ATOFVersion", event.ATOFVersion())
	assertEmptyEventString(t, "ScopeType", event.ScopeType())
	assertZeroEventAttributes(t, event.Attributes())
	assertNilEventJSON(t, "AttributesJSON", event.AttributesJSON())
	assertNilEventJSON(t, "Data", event.Data())
	assertNilEventJSON(t, "DataSchema", event.DataSchema())
	assertNilEventJSON(t, "Metadata", event.Metadata())
	assertEmptyEventString(t, "Timestamp", event.Timestamp())
	assertNilEventJSON(t, "Input", event.Input())
	assertNilEventJSON(t, "Output", event.Output())
	assertEmptyEventString(t, "ModelName", event.ModelName())
	assertEmptyEventString(t, "ToolCallID", event.ToolCallID())
	assertEmptyEventString(t, "ParentUUID", event.ParentUUID())
	assertNilEventJSON(t, "AnnotatedRequest", event.AnnotatedRequest())
	assertNilEventJSON(t, "AnnotatedResponse", event.AnnotatedResponse())
	assertEmptyEventString(t, "ScopeCategory", event.ScopeCategory())
	assertEmptyEventString(t, "Category", event.Category())
	assertNilEventJSON(t, "CategoryProfile", event.CategoryProfile())
}

func assertEmptyEventString(t *testing.T, name string, got string) {
	t.Helper()
	if got != "" {
		t.Fatalf("expected empty %s, got %q", name, got)
	}
}

func assertZeroEventAttributes(t *testing.T, got uint32) {
	t.Helper()
	if got != 0 {
		t.Fatalf("expected zero Attributes, got %d", got)
	}
}

func assertNilEventJSON(t *testing.T, name string, got []byte) {
	t.Helper()
	if got != nil {
		t.Fatalf("expected nil %s, got %s", name, got)
	}
}

func TestPublicAPIErrorAndDefaultCoverage(t *testing.T) {
	runTestWithScopeStack(t, testPublicAPIErrorAndDefaultCoverage)
}

func testPublicAPIErrorAndDefaultCoverage(t *testing.T) {
	assertInvalidScopePayloads(t)
	assertInvalidCallPayloads(t)
	assertClosedExporterFails(t)
	assertZeroSubscriberConfigs(t)
	if got := mustConfigMap(nil); len(got) != 0 {
		t.Fatalf("expected empty map for nil config payload, got %#v", got)
	}
}

func assertInvalidScopePayloads(t *testing.T) {
	t.Helper()
	for _, tc := range []struct {
		name string
		opt  ScopeOption
	}{
		{name: "data", opt: WithData(json.RawMessage("{"))},
		{name: "metadata", opt: WithMetadata(json.RawMessage("{"))},
		{name: "input", opt: WithInput(json.RawMessage("{"))},
	} {
		if _, err := PushScope("invalid_scope_json_"+tc.name, ScopeTypeAgent, tc.opt); err == nil {
			t.Fatalf("expected PushScope to fail on invalid JSON %s", tc.name)
		}
	}

	handle, err := PushScope("invalid_scope_end_metadata", ScopeTypeAgent)
	if err != nil {
		t.Fatalf(pushScopeFailed, err)
	}
	if PopScope(handle, WithScopeEndMetadata(json.RawMessage("{"))) == nil {
		t.Fatal("expected PopScope to fail on invalid end metadata JSON")
	}
	if err := PopScope(handle); err != nil {
		t.Fatalf("cleanup PopScope failed: %v", err)
	}
}

func assertInvalidCallPayloads(t *testing.T) {
	t.Helper()
	if _, err := ToolCall("invalid_tool_json", json.RawMessage("{")); err == nil {
		t.Fatal("expected ToolCall to fail on invalid JSON args")
	}

	badMarshal := map[string]interface{}{"ch": make(chan int)}
	if _, err := LlmCall("llm_marshal_error", badMarshal); err == nil {
		t.Fatal("expected LlmCall marshal error")
	}
	if _, err := LlmCallExecute("llm_execute_marshal_error", badMarshal, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}); err == nil {
		t.Fatal("expected LlmCallExecute marshal error")
	}
	if _, err := LlmStreamCallExecute("llm_stream_marshal_error", badMarshal, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}, nil, nil); err == nil {
		t.Fatal("expected LlmStreamCallExecute marshal error")
	}

	malformedRequest := map[string]interface{}{"not": "an LLMRequest"}
	if _, err := LlmCall("llm_invalid_request", malformedRequest); err == nil {
		t.Fatal("expected LlmCall request-shape error")
	}
	if _, err := LlmCallExecute("llm_execute_invalid_request", malformedRequest, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}); err == nil {
		t.Fatal("expected LlmCallExecute request-shape error")
	}
	if _, err := LlmStreamCallExecute("llm_stream_invalid_request", malformedRequest, func(json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`null`), nil
	}, nil, nil); err == nil {
		t.Fatal("expected LlmStreamCallExecute request-shape error")
	}

	if _, err := ToolRequestIntercepts("invalid_tool_request_intercepts", json.RawMessage("{")); err == nil {
		t.Fatal("expected ToolRequestIntercepts to fail on invalid JSON")
	}
	if _, err := LlmRequestIntercepts("invalid_llm_request_intercepts", json.RawMessage("{")); err == nil {
		t.Fatal("expected LlmRequestIntercepts to fail on invalid JSON")
	}
}

func assertClosedExporterFails(t *testing.T) {
	t.Helper()
	exporter, err := NewAtifExporter("session-gap", "agent-gap", "1.0.0", "")
	if err != nil {
		t.Fatalf("NewAtifExporter failed: %v", err)
	}
	exporter.Close()
	if _, err := exporter.ExportJSON(); err == nil {
		t.Fatal("expected ExportJSON to fail after Close")
	}
}

func assertZeroSubscriberConfigs(t *testing.T) {
	t.Helper()
	if _, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{Type: "full"}); err == nil || err.Error() != "endpoint is required" {
		t.Fatalf("expected endpoint is required error, got %v", err)
	}
}

func TestWrapperAndCodecFinalizersRun(t *testing.T) {
	runTestWithScopeStack(t, testWrapperAndCodecFinalizersRun)
}

func TestGoBindingErrorAndDefaultContracts(t *testing.T) {
	assertAdaptiveJSONUnmarshalFailures(t)
	assertOpenTelemetryDefaultConfig(t)
	assertEventSnapshotAccessors(t)
	assertExpiredCodecAndStreamErrors(t)
	assertStreamExporterPath(t)
	assertCallbackSerializationFailures(t)
	assertPluginActivationIncompleteOutputs(t)
	assertAtofExporterSerializationFailures(t)
	assertClosedHandleErrorPaths(t)
	assertOptimizationContributionValidation(t)
}

func TestRemainingPublicDefaultsAndPropagationCoverage(t *testing.T) {
	endpoint := "http://127.0.0.1:4318"
	assertOpenTelemetrySignalDefaults(t, endpoint)
	assertOpenTelemetrySignalMarshalFailures(t, endpoint)
	assertPropagationCoverage(t)
	assertMetricTimestampCoverage(t)
}

func assertOpenTelemetrySignalDefaults(t *testing.T, endpoint string) {
	t.Helper()
	logConfig, err := normalizeOpenTelemetryLogConfig(OpenTelemetryLogConfig{Endpoint: endpoint})
	if err != nil {
		t.Fatalf("normalize log defaults failed: %v", err)
	}
	if logConfig.MinimumSeverity != LogSeverityInfo || logConfig.MaxQueueSize != 2048 ||
		logConfig.MaxExportBatchSize != 512 || logConfig.ScheduledDelay != time.Second ||
		logConfig.CompletedSpanContextTTL != 60*time.Second ||
		logConfig.Headers == nil || logConfig.ResourceAttributes == nil {
		t.Fatalf("log defaults were not applied: %#v", logConfig)
	}

	metricConfig, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{Endpoint: endpoint})
	if err != nil {
		t.Fatalf("normalize metric defaults failed: %v", err)
	}
	if metricConfig.Temporality != OpenTelemetryMetricTemporalityCumulative ||
		metricConfig.MaxInstruments != 256 || metricConfig.CardinalityLimit != 2000 ||
		metricConfig.Headers == nil || metricConfig.ResourceAttributes == nil {
		t.Fatalf("metric defaults were not applied: %#v", metricConfig)
	}

	for name, normalize := range map[string]func() error{
		"negative log duration": func() error {
			_, err := normalizeOpenTelemetryLogConfig(OpenTelemetryLogConfig{
				Endpoint: endpoint,
				Timeout:  -time.Millisecond,
			})
			return err
		},
		"negative metric duration": func() error {
			_, err := normalizeOpenTelemetryMetricConfig(OpenTelemetryMetricConfig{
				Endpoint:       endpoint,
				ExportInterval: -time.Millisecond,
			})
			return err
		},
	} {
		if normalize() == nil {
			t.Fatalf("%s should fail", name)
		}
	}
}

func assertOpenTelemetrySignalMarshalFailures(t *testing.T, endpoint string) {
	t.Helper()
	oldMarshal := jsonMarshal
	t.Cleanup(func() { jsonMarshal = oldMarshal })
	jsonMarshal = func(any) ([]byte, error) {
		return nil, errors.New("forced signal header marshal failure")
	}
	if _, err := newOpenTelemetrySignalCStrings(openTelemetrySignalConfig{endpoint: endpoint}); err == nil {
		t.Fatal("expected signal header marshal failure")
	}
	callCount := 0
	jsonMarshal = func(value any) ([]byte, error) {
		callCount++
		if callCount == 2 {
			return nil, errors.New("forced signal resource marshal failure")
		}
		return oldMarshal(value)
	}
	if _, err := newOpenTelemetrySignalCStrings(openTelemetrySignalConfig{endpoint: endpoint}); err == nil {
		t.Fatal("expected signal resource marshal failure")
	}
	jsonMarshal = oldMarshal
}

func assertPropagationCoverage(t *testing.T) {
	t.Helper()
	rootUUID := propagationRootUUID
	context := PropagationContext{
		Version:    1,
		RootUUID:   &rootUUID,
		ParentUUID: propagationParentUUID,
	}
	traceparent, err := context.ToTraceparent()
	if err != nil {
		t.Fatalf("ToTraceparent failed: %v", err)
	}
	if traceparent == "" {
		t.Fatal("ToTraceparent returned an empty value")
	}
	if _, err := (PropagationContext{Version: 2, ParentUUID: propagationParentUUID}).ToTraceparent(); err == nil {
		t.Fatal("expected invalid propagation context to fail")
	}

	if ScopeDeregisterEventMetadataInjector(propagationParentUUID, "missing-injector") == nil {
		t.Fatal("expected missing scope metadata injector deregistration to fail")
	}
}

func assertMetricTimestampCoverage(t *testing.T) {
	t.Helper()
	expectedMetricTimestamp := time.Unix(1_700_000_000, 0)
	observedMetricTimestamp := make(chan string, 1)
	subscriberName := "go_metric_timestamp_coverage_" + time.Now().Format("150405.000000")
	if err := RegisterSubscriber(subscriberName, func(event Event) {
		if event.Name() == "go_metric_timestamp_coverage" {
			observedMetricTimestamp <- event.Timestamp()
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer DeregisterSubscriber(subscriberName)

	runTestWithScopeStack(t, func(t *testing.T) {
		err := EmitMetric("go_metric_timestamp_coverage", []MetricMeasurement{{
			Name:      "example.go.timestamp",
			Kind:      MetricKindCounter,
			ValueType: MetricValueTypeU64,
			Value:     uint64(1),
		}}, WithMetricTimestamp(expectedMetricTimestamp))
		if err != nil {
			t.Fatalf("EmitMetric with timestamp failed: %v", err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatalf("FlushSubscribers failed: %v", err)
		}
	})

	select {
	case timestamp := <-observedMetricTimestamp:
		observed, err := time.Parse(time.RFC3339Nano, timestamp)
		if err != nil {
			t.Fatalf("failed to parse metric timestamp %q: %v", timestamp, err)
		}
		if !observed.Equal(expectedMetricTimestamp) {
			t.Fatalf("metric timestamp = %s, expected %s", observed, expectedMetricTimestamp)
		}
	default:
		t.Fatal("expected metric event with supplied timestamp")
	}
}

func assertAdaptiveJSONUnmarshalFailures(t *testing.T) {
	t.Helper()
	runtime, err := NewAdaptiveRuntime(testAdaptiveRuntimeConfig("openai"))
	if err != nil {
		t.Fatalf("NewAdaptiveRuntime failed: %v", err)
	}
	defer runtime.Shutdown()
	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	scope, err := PushScope("adaptive_bind_scope", ScopeTypeAgent)
	if err != nil {
		t.Fatalf(pushScopeFailed, err)
	}
	if err := runtime.BindScope(scope); err != nil {
		t.Fatalf("BindScope failed: %v", err)
	}
	if err := PopScope(scope); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}

	oldUnmarshal := jsonUnmarshal
	t.Cleanup(func() { jsonUnmarshal = oldUnmarshal })
	jsonUnmarshal = func([]byte, any) error {
		return errors.New("forced adaptive JSON unmarshal failure")
	}

	if _, err := ValidateAdaptiveConfig(NewAdaptiveConfig()); err == nil {
		t.Fatal("expected ValidateAdaptiveConfig to return the decode failure")
	}
	if _, err := runtime.Report(); err == nil {
		t.Fatal("expected Report to return the decode failure")
	}
	if _, err := runtime.BuildCacheRequestFacts(CacheRequestFactsInput{
		Provider:         "openai",
		RequestID:        "00000000-0000-0000-0000-000000000501",
		AnnotatedRequest: json.RawMessage(`{"model":"test"}`),
		AgentID:          "go-adaptive-openai",
	}); err == nil {
		t.Fatal("expected BuildCacheRequestFacts to return the decode failure")
	}
	if _, err := BuildCacheTelemetryEvent(CacheTelemetryEventInput{
		Provider:  "openai",
		RequestID: "00000000-0000-0000-0000-000000000502",
		Usage:     &CacheUsage{PromptTokens: uint64Ptr(1)},
		AgentID:   "agent", TemplateVersion: "v1", ToolsetHash: "tools", ModelFamily: "model", TenantScope: "tenant",
	}); err == nil {
		t.Fatal("expected BuildCacheTelemetryEvent to return the decode failure")
	}
}

func assertOpenTelemetryDefaultConfig(t *testing.T) {
	t.Helper()
	if _, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{}); err == nil || err.Error() != "type is required" {
		t.Fatalf("expected type is required error, got %v", err)
	}
	if _, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{
		Type: OpenTelemetryType("not_a_projection"), Endpoint: "http://127.0.0.1:4318/v1/traces",
	}); err == nil {
		t.Fatal("expected unsupported OpenTelemetry projection to fail")
	}

	subscriber, err := NewOpenTelemetrySubscriber(OpenTelemetryConfig{
		Type:     OpenTelemetryTypeFull,
		Endpoint: "http://127.0.0.1:4318/v1/traces",
	})
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber failed: %v", err)
	}
	if err := subscriber.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}

func assertEventSnapshotAccessors(t *testing.T) {
	t.Helper()
	event := eventBase{snapshot: &eventSnapshot{
		atofVersion:     "0.1",
		scopeCategory:   "scope",
		category:        "test",
		attributesJSON:  json.RawMessage(`{"parallel":true}`),
		categoryProfile: json.RawMessage(`{"profile":true}`),
		data:            json.RawMessage(`{"data":true}`),
		dataSchema:      json.RawMessage(`{"name":"test"}`),
	}}
	if event.ATOFVersion() != "0.1" || event.ScopeCategory() != "scope" || event.Category() != "test" {
		t.Fatal("event snapshot strings were not returned")
	}
	for _, value := range []json.RawMessage{
		event.AttributesJSON(), event.CategoryProfile(), event.Data(), event.DataSchema(),
	} {
		if value == nil {
			t.Fatal("event snapshot JSON was not returned")
		}
	}
	if event.CategoryProfile() == nil {
		t.Fatal("event snapshot category profile was not returned")
	}
}

func assertExpiredCodecAndStreamErrors(t *testing.T) {
	t.Helper()
	request := LLMRequestDTO{Headers: json.RawMessage(`{}`), Content: json.RawMessage(`{}`)}
	if _, err := (*LLMRequestSanitizeCodec)(nil).Decode(request); !errors.Is(err, ErrLLMSanitizeCodecExpired) {
		t.Fatalf("nil request codec Decode error = %v", err)
	}
	if _, err := (*LLMRequestSanitizeCodec)(nil).Encode(json.RawMessage(`{}`), request); !errors.Is(err, ErrLLMSanitizeCodecExpired) {
		t.Fatalf("nil request codec Encode error = %v", err)
	}
	if _, err := (*LLMResponseSanitizeCodec)(nil).Decode(json.RawMessage(`{}`)); !errors.Is(err, ErrLLMSanitizeCodecExpired) {
		t.Fatalf("nil response codec Decode error = %v", err)
	}
	setLastErrorMessage("forced stream failure")
	if _, err := llmStreamNextResult(-1, nil, nil, nil); err == nil {
		t.Fatal("expected stream error status to propagate")
	}
}

func assertStreamExporterPath(t *testing.T) {
	t.Helper()
	exporter, err := NewAtofExporter(AtofExporterConfig{Sink: NewAtofStreamSinkConfig("http://127.0.0.1:1/events")})
	if err != nil {
		t.Fatalf("NewAtofExporter stream failed: %v", err)
	}
	defer exporter.Close()
	path, err := exporter.Path()
	if err != nil {
		t.Fatalf("stream exporter Path failed: %v", err)
	}
	if path != nil {
		t.Fatalf("stream exporter path = %q, want nil", *path)
	}
}

func assertCallbackSerializationFailures(t *testing.T) {
	t.Helper()
	const toolName = "coverage_tool_intercept_serialize_failure"
	if err := RegisterToolExecutionIntercept(toolName, 1, func(args json.RawMessage, _ func(json.RawMessage) (ToolExecutionResult, error)) (ToolExecutionInterceptOutcome, error) {
		return ToolExecutionInterceptOutcome{Result: args}, nil
	}); err != nil {
		t.Fatalf("RegisterToolExecutionIntercept failed: %v", err)
	}
	defer DeregisterToolExecutionIntercept(toolName)

	const llmName = "coverage_llm_intercept_serialize_failure"
	if err := RegisterLlmRequestIntercept(llmName, 1, false, func(_ string, request LLMRequestDTO, annotated json.RawMessage) (LLMRequestInterceptOutcome, error) {
		return LLMRequestInterceptOutcome{Request: request, AnnotatedRequest: annotated}, nil
	}); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept failed: %v", err)
	}
	defer DeregisterLlmRequestIntercept(llmName)

	oldMarshal := jsonMarshal
	jsonMarshal = func(any) ([]byte, error) { return nil, errors.New("forced callback JSON marshal failure") }
	defer func() { jsonMarshal = oldMarshal }()

	if _, err := ToolCallExecute(toolName, json.RawMessage(`{}`), func(json.RawMessage) (ToolExecutionResult, error) {
		return toolExecutionResult(json.RawMessage(`{}`)), nil
	}); err == nil {
		t.Fatal("expected tool intercept serialization failure")
	}
	if _, err := LlmRequestIntercepts(llmName, json.RawMessage(`{"headers":{},"content":{}}`)); err == nil {
		t.Fatal("expected LLM request intercept serialization failure")
	}
}

func assertPluginActivationIncompleteOutputs(t *testing.T) {
	t.Helper()
	withPluginActivationStubs(t)
	initializeWithDynamicPluginsJSON = func(string, string) (unsafe.Pointer, string, error) {
		return nil, "", nil
	}
	if _, _, err := InitializeWithDynamicPlugins(NewPluginConfig(), fixtureDynamicPluginSpecs()); err == nil {
		t.Fatal("expected incomplete dynamic plugin activation outputs to fail")
	}
	if err := newPluginActivation(nil).Close(); err != nil {
		t.Fatalf("nil plugin activation close failed: %v", err)
	}
}

func assertAtofExporterSerializationFailures(t *testing.T) {
	t.Helper()
	if _, err := json.Marshal(AtofExporterConfig{}); err != nil {
		t.Fatalf("marshal default ATOF config failed: %v", err)
	}
	if _, err := NewAtofExporter(AtofExporterConfig{Sink: failingAtofSink{}}); err == nil {
		t.Fatal("expected ATOF exporter serialization failure")
	}
}

func assertClosedHandleErrorPaths(t *testing.T) {
	t.Helper()
	if _, err := NewAdaptiveRuntime(AdaptiveConfig{Acg: &AcgConfig{Provider: "unsupported"}}); err == nil {
		t.Fatal("expected unsupported adaptive provider to fail runtime construction")
	}

	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	stack.Close()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected closed scope stack Run to panic")
			}
		}()
		stack.Run(func() {
			// The empty callback isolates the closed-stack panic path.
		})
	}()

	exporter, err := NewAtofExporter(NewAtofExporterConfig())
	if err != nil {
		t.Fatalf("NewAtofExporter failed: %v", err)
	}
	exporter.Close()
	if _, err := exporter.Path(); err == nil {
		t.Fatal("expected closed ATOF exporter Path to fail")
	}
}

func assertOptimizationContributionValidation(t *testing.T) {
	t.Helper()
	if _, err := json.Marshal(LLMOptimizationContribution{
		Producer: "test", Kind: "custom", Payload: json.RawMessage("{"),
		PayloadSchema: &LLMOptimizationDataSchema{Name: "test", Version: "v1"},
	}); err == nil {
		t.Fatal("expected invalid optimization payload marshal failure")
	}

	for _, raw := range []string{
		`[]`,
		`null`,
		`{"kind":"custom"}`,
		`{"producer":"test"}`,
	} {
		var contribution LLMOptimizationContribution
		if json.Unmarshal([]byte(raw), &contribution) == nil {
			t.Fatalf("expected invalid optimization contribution %s to fail", raw)
		}
	}
}

func testWrapperAndCodecFinalizersRun(t *testing.T) {
	scopeHandle, err := PushScope("finalizer_scope", ScopeTypeAgent)
	if err != nil {
		t.Fatalf(pushScopeFailed, err)
	}
	if err := PopScope(scopeHandle); err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}

	toolHandle, err := ToolCall("finalizer_tool", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if err := ToolCallEnd(toolHandle, toolExecutionResult(json.RawMessage(`{}`))); err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}

	llmHandle, err := LlmCall("finalizer_llm", map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"model": "test-model"},
	})
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if err := LlmCallEnd(llmHandle, json.RawMessage(`{"content":"ok"}`)); err != nil {
		t.Fatalf("LlmCallEnd failed: %v", err)
	}

	request := NewLLMRequest(
		map[string]interface{}{"x-test": "finalizer"},
		map[string]interface{}{"model": "test-model"},
	)
	if request == nil {
		t.Fatal("expected non-nil LLMRequest")
	}

	chatCodec := NewOpenAIChatCodec()
	responsesCodec := NewOpenAIResponsesCodec()
	anthropicCodec := NewAnthropicMessagesCodec()
	geminiCodec := NewGeminiGenerateContentCodec()
	if chatCodec == nil || responsesCodec == nil || anthropicCodec == nil || geminiCodec == nil {
		t.Fatal("expected non-nil codec handles")
	}

	scopeHandle = nil
	toolHandle = nil
	llmHandle = nil
	request = nil
	chatCodec = nil
	responsesCodec = nil
	anthropicCodec = nil
	geminiCodec = nil

	for i := 0; i < 8; i++ {
		runtime.GC()
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
	}
}

func TestGeminiGenerateContentCodecFunctionCallID(t *testing.T) {
	runTestWithScopeStack(t, testGeminiGenerateContentCodecFunctionCallID)
}

func testGeminiGenerateContentCodecFunctionCallID(t *testing.T) {
	// Verify that NewGeminiGenerateContentCodec returns a usable handle and that,
	// when processing a Gemini response with an explicit functionCall.id,
	// the annotated response carries the actual id (not the function name).
	geminiResp := json.RawMessage(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [{"functionCall": {"id": "call_abc123", "name": "my_fn", "args": {"x": 1}}}]
			},
			"finishReason": "STOP",
			"index": 0
		}],
		"usageMetadata": {}
	}`)

	executor := func(_ json.RawMessage) (json.RawMessage, error) {
		return geminiResp, nil
	}

	capturedEvents, cleanup := registerLlmCodecEventCollector(t)
	defer cleanup()

	_, err := LlmCallExecute(
		"gemini_fn_id_test",
		map[string]interface{}{
			"headers": map[string]interface{}{},
			"content": map[string]interface{}{
				"contents": []interface{}{
					map[string]interface{}{
						"role": "user",
						"parts": []interface{}{
							map[string]interface{}{"text": "call my_fn"},
						},
					},
				},
			},
		},
		executor,
		WithLLMResponseCodec(NewGeminiGenerateContentCodec()),
	)
	if err != nil {
		t.Fatalf("LlmCallExecute with GeminiGenerateContentCodec failed: %v", err)
	}

	events := capturedEvents()
	_, endEvent := requireLlmScopeEvents(t, events)

	var annotated map[string]interface{}
	if err := json.Unmarshal(endEvent.AnnotatedResponse(), &annotated); err != nil {
		t.Fatalf("AnnotatedResponse not valid JSON: %v", err)
	}

	toolCalls, ok := annotated["tool_calls"].([]interface{})
	if !ok || len(toolCalls) == 0 {
		t.Fatalf("expected tool_calls in annotated response, got %#v", annotated)
	}
	tc, ok := toolCalls[0].(map[string]interface{})
	if !ok {
		t.Fatalf("expected tool_calls[0] to be a map, got %T", toolCalls[0])
	}
	id, _ := tc["id"].(string)
	if id != "call_abc123" {
		t.Errorf("functionCall id must be 'call_abc123', got %q", id)
	}
	name, _ := tc["name"].(string)
	if name != "my_fn" {
		t.Errorf("functionCall name must be 'my_fn', got %q", name)
	}
	if id == name {
		t.Error("id must differ from name — id must not be the function name")
	}
}
