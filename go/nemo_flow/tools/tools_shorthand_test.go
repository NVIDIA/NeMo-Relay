// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tools_test

import (
	"encoding/json"
	"testing"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
	toolspkg "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/tools"
)

func TestToolShorthands(t *testing.T) {
	handle, err := toolspkg.Call("tools_call", json.RawMessage(`{"value": 1}`))
	if err != nil {
		t.Fatalf("Call failed: %v", err)
	}
	if err := toolspkg.CallEnd(handle, json.RawMessage(`{"ok": true}`)); err != nil {
		t.Fatalf("CallEnd failed: %v", err)
	}

	result, err := toolspkg.Execute("tools_execute", json.RawMessage(`{"value": 2}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		},
	)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	var executed map[string]interface{}
	if err := json.Unmarshal(result, &executed); err != nil {
		t.Fatalf("unmarshal execute result: %v", err)
	}
	if executed["value"] != float64(2) {
		t.Fatalf("expected value=2, got %v", executed)
	}

	if err := nemo_flow.RegisterToolRequestIntercept("tools_req_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var payload map[string]interface{}
			_ = json.Unmarshal(args, &payload)
			payload["intercepted"] = true
			out, _ := json.Marshal(payload)
			return out
		},
	); err != nil {
		t.Fatalf("RegisterToolRequestIntercept failed: %v", err)
	}
	t.Cleanup(func() {
		_ = nemo_flow.DeregisterToolRequestIntercept("tools_req_int")
	})

	transformedArgs, err := toolspkg.RequestIntercepts("tools_req", json.RawMessage(`{"value": 3}`))
	if err != nil {
		t.Fatalf("RequestIntercepts failed: %v", err)
	}

	var intercepted map[string]interface{}
	if err := json.Unmarshal(transformedArgs, &intercepted); err != nil {
		t.Fatalf("unmarshal intercepted args: %v", err)
	}
	if intercepted["intercepted"] != true {
		t.Fatalf("expected intercepted=true, got %v", intercepted)
	}

	if err := nemo_flow.RegisterToolConditionalExecutionGuardrail("tools_cond", 1,
		func(name string, args json.RawMessage) *string { return nil },
	); err != nil {
		t.Fatalf("RegisterToolConditionalExecutionGuardrail failed: %v", err)
	}
	t.Cleanup(func() {
		_ = nemo_flow.DeregisterToolConditionalExecutionGuardrail("tools_cond")
	})

	if err := toolspkg.ConditionalExecution("tools_conditional", json.RawMessage(`{"value": 4}`)); err != nil {
		t.Fatalf("ConditionalExecution failed: %v", err)
	}
}
