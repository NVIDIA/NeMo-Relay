// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestConcreteMetricNumbersPreserveNativeRange(t *testing.T) {
	runWithTestScopeStack(t, func() {
		for _, measurement := range []MetricMeasurement{
			{Name: "unsigned", Kind: MetricKindGauge, ValueType: MetricValueTypeU64, Value: json.Number("9223372036854775807")},
			{Name: "signed", Kind: MetricKindGauge, ValueType: MetricValueTypeI64, Value: json.Number("-9223372036854775808")},
			{Name: "fraction", Kind: MetricKindGauge, ValueType: MetricValueTypeF64, Value: json.Number("0.125")},
		} {
			if err := EmitMetric("typed-numbers", []MetricMeasurement{measurement}); err != nil {
				t.Fatalf("%s: %v", measurement.Name, err)
			}
		}
		overflow := MetricMeasurement{Name: "overflow", Kind: MetricKindGauge, ValueType: MetricValueTypeU64, Value: json.Number("18446744073709551615")}
		payload, err := json.Marshal(overflow)
		if err != nil || !bytes.Contains(payload, []byte(`"value":18446744073709551615`)) {
			t.Fatalf("integer precision lost: %s (%v)", payload, err)
		}
		if err := EmitMetric("typed-overflow", []MetricMeasurement{overflow}); err == nil {
			t.Fatal("values above the runtime instrument limit must fail")
		}
		invalid := MetricMeasurement{Name: "invalid", Kind: MetricKindGauge, ValueType: MetricValueTypeU64, Value: json.Number("not-a-number")}
		if err := EmitMetric("typed-invalid", []MetricMeasurement{invalid}); err == nil {
			t.Fatal("invalid numeric JSON must fail")
		}
	})
}

func TestTypedStreamAdaptersAndNilCallbacks(t *testing.T) {
	var execution LLMExecutionFunc
	var interceptor LLMExecutionInterceptFunc
	if execution.AsStream() != nil || interceptor.AsStream() != nil {
		t.Fatal("nil callbacks must remain nil")
	}
	request := LLMRequest{Headers: json.RawMessage(`{}`), Content: json.RawMessage(`{}`)}
	if _, err := LlmStreamCallExecute("nil-provider", request, nil, nil, nil); err == nil {
		t.Fatal("nil provider must fail")
	}
	if err := RegisterLlmStreamExecutionIntercept("nil-interceptor", 0, nil); err == nil {
		t.Fatal("nil interceptor must fail")
	}
}
