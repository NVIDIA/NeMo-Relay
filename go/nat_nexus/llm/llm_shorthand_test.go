// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
	llmpkg "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/llm"
)

func makeRequest() map[string]interface{} {
	return map[string]interface{}{
		"headers": map[string]interface{}{},
		"content": map[string]interface{}{"messages": []interface{}{}, "model": "test-model"},
	}
}

func TestLlmShorthands(t *testing.T) {
	handle, err := llmpkg.Call("llm_call", makeRequest())
	if err != nil {
		t.Fatalf("Call failed: %v", err)
	}
	if err := llmpkg.CallEnd(handle, json.RawMessage(`{"ok": true}`)); err != nil {
		t.Fatalf("CallEnd failed: %v", err)
	}

	response, err := llmpkg.Execute("llm_execute", makeRequest(),
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"ok": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	var executeResult map[string]interface{}
	if err := json.Unmarshal(response, &executeResult); err != nil {
		t.Fatalf("unmarshal execute result: %v", err)
	}
	if executeResult["ok"] != true {
		t.Fatalf("expected ok=true, got %v", executeResult)
	}

	if err := nat_nexus.RegisterLlmRequestIntercept("llm_req_int", 1, false,
		func(headers, content json.RawMessage) (json.RawMessage, json.RawMessage) {
			var payload map[string]interface{}
			_ = json.Unmarshal(content, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return headers, out
		},
	); err != nil {
		t.Fatalf("RegisterLlmRequestIntercept failed: %v", err)
	}
	t.Cleanup(func() {
		_ = nat_nexus.DeregisterLlmRequestIntercept("llm_req_int")
	})

	request, err := llmpkg.RequestIntercepts("llm_req", json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`))
	if err != nil {
		t.Fatalf("RequestIntercepts failed: %v", err)
	}

	var intercepted struct {
		Content map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(request, &intercepted); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if intercepted.Content["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", intercepted.Content)
	}

	if err := nat_nexus.RegisterLlmConditionalExecutionGuardrail("llm_cond", 1,
		func(headers, content json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterLlmConditionalExecutionGuardrail failed: %v", err)
	}
	t.Cleanup(func() {
		_ = nat_nexus.DeregisterLlmConditionalExecutionGuardrail("llm_cond")
	})

	if err := llmpkg.ConditionalExecution(json.RawMessage(`{"headers":{},"content":{"model":"test-model"}}`)); err != nil {
		t.Fatalf("ConditionalExecution failed: %v", err)
	}

	stream, err := llmpkg.StreamExecute("llm_stream", makeRequest(),
		func(nativeJSON json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`"` + strings.ReplaceAll("data: {\"chunk\": 1}\n\ndata: [DONE]\n\n", `"`, `\"`) + `"`), nil
		},
		nil, nil,
	)
	if err != nil {
		t.Fatalf("StreamExecute failed: %v", err)
	}
	defer stream.Close()

	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream.Next failed: %v", err)
		}
	}
}
