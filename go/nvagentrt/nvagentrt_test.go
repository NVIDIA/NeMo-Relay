package nvagentrt

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// Note: These tests require the nvagentrt-ffi shared library to be built
// and available in the library search path. Build with:
//   cargo build --release -p nvagentrt-ffi
// Then run tests with:
//   CGO_LDFLAGS="-L../../target/release" go test -v ./...

// ============================================================================
// Types and Constants
// ============================================================================

func TestScopeTypeConstants(t *testing.T) {
	types := []ScopeType{
		ScopeTypeAgent, ScopeTypeFunction, ScopeTypeTool, ScopeTypeLlm,
		ScopeTypeRetriever, ScopeTypeEmbedder, ScopeTypeReranker,
		ScopeTypeGuardrail, ScopeTypeEvaluator, ScopeTypeCustom, ScopeTypeUnknown,
	}
	if len(types) != 11 {
		t.Fatalf("expected 11 scope types, got %d", len(types))
	}
	// Verify sequential values
	for i, st := range types {
		if int(st) != i {
			t.Fatalf("ScopeType at index %d has value %d", i, int(st))
		}
	}
}

func TestEventTypeConstants(t *testing.T) {
	if EventTypeStart != 0 || EventTypeEnd != 1 || EventTypeMark != 2 {
		t.Fatal("unexpected EventType values")
	}
}

func TestScopeAttributeConstants(t *testing.T) {
	if ScopeAttrParallel != 0b01 || ScopeAttrRelocatable != 0b10 {
		t.Fatal("unexpected ScopeAttr values")
	}
	combined := ScopeAttrParallel | ScopeAttrRelocatable
	if combined != 0b11 {
		t.Fatal("combined scope attributes incorrect")
	}
}

func TestToolAttributeConstants(t *testing.T) {
	if ToolAttrLocal != 0b01 {
		t.Fatal("unexpected ToolAttr value")
	}
}

func TestLLMAttributeConstants(t *testing.T) {
	if LLMAttrStateless != 0b01 || LLMAttrStreaming != 0b10 {
		t.Fatal("unexpected LLMAttr values")
	}
}

// ============================================================================
// LLMRequest
// ============================================================================

func TestNewLLMRequest(t *testing.T) {
	req := NewLLMRequest("POST", "https://api.example.com",
		map[string]interface{}{"Authorization": "Bearer token"},
		map[string]interface{}{"messages": []string{}},
	)
	if req == nil {
		t.Fatal("NewLLMRequest returned nil")
	}
	if req.Method() != "POST" {
		t.Fatalf("expected POST, got %s", req.Method())
	}
	if req.URL() != "https://api.example.com" {
		t.Fatalf("expected URL, got %s", req.URL())
	}
	if req.Headers() == nil {
		t.Fatal("headers is nil")
	}
	if req.Body() == nil {
		t.Fatal("body is nil")
	}
}

func TestNewLLMRequestEmptyHeaders(t *testing.T) {
	req := NewLLMRequest("GET", "https://api.example.com", map[string]interface{}{}, map[string]interface{}{})
	if req == nil {
		t.Fatal("returned nil")
	}
	if req.Method() != "GET" {
		t.Fatalf("expected GET, got %s", req.Method())
	}
}

// ============================================================================
// Scope operations
// ============================================================================

func TestPushPopScope(t *testing.T) {
	handle, err := PushScope("test_scope", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	if handle == nil {
		t.Fatal("PushScope returned nil handle")
	}
	if handle.Name() != "test_scope" {
		t.Fatalf("expected name 'test_scope', got '%s'", handle.Name())
	}

	current, err := GetHandle()
	if err != nil {
		t.Fatalf("GetHandle failed: %v", err)
	}
	if current == nil {
		t.Fatal("GetHandle returned nil")
	}
	if current.Name() != "test_scope" {
		t.Fatalf("expected current to be 'test_scope', got '%s'", current.Name())
	}

	err = PopScope(handle)
	if err != nil {
		t.Fatalf("PopScope failed: %v", err)
	}
}

func TestScopeHandleProperties(t *testing.T) {
	handle, err := PushScope("props_test", ScopeTypeRetriever)
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}
	if handle.Name() != "props_test" {
		t.Fatalf("expected 'props_test', got '%s'", handle.Name())
	}
	if handle.Type() != ScopeTypeRetriever {
		t.Fatalf("expected ScopeTypeRetriever, got %d", handle.Type())
	}
}

func TestPushScopeWithAttributes(t *testing.T) {
	handle, err := PushScope("parallel", ScopeTypeFunction, WithScopeAttributes(ScopeAttrParallel))
	if err != nil {
		t.Fatalf("PushScope failed: %v", err)
	}
	defer PopScope(handle)

	if handle.Attributes()&ScopeAttrParallel == 0 {
		t.Fatal("expected PARALLEL attribute to be set")
	}
}

func TestPushScopeWithParent(t *testing.T) {
	parent, err := PushScope("parent", ScopeTypeAgent)
	if err != nil {
		t.Fatalf("PushScope parent failed: %v", err)
	}
	defer PopScope(parent)

	child, err := PushScope("child", ScopeTypeFunction, WithParent(parent))
	if err != nil {
		t.Fatalf("PushScope child failed: %v", err)
	}
	defer PopScope(child)

	if child.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), child.ParentUUID())
	}
}

func TestNestedScopes(t *testing.T) {
	s1, _ := PushScope("level1", ScopeTypeAgent)
	s2, _ := PushScope("level2", ScopeTypeFunction)
	s3, _ := PushScope("level3", ScopeTypeTool)

	current, _ := GetHandle()
	if current.Name() != "level3" {
		t.Fatalf("expected 'level3', got '%s'", current.Name())
	}

	PopScope(s3)
	current, _ = GetHandle()
	if current.Name() != "level2" {
		t.Fatalf("expected 'level2', got '%s'", current.Name())
	}

	PopScope(s2)
	current, _ = GetHandle()
	if current.Name() != "level1" {
		t.Fatalf("expected 'level1', got '%s'", current.Name())
	}

	PopScope(s1)
}

func TestPopInvalidScopeErrors(t *testing.T) {
	handle, _ := PushScope("once", ScopeTypeAgent)
	PopScope(handle)
	err := PopScope(handle)
	if err == nil {
		t.Fatal("expected error when popping already-popped scope")
	}
}

func TestAllScopeTypes(t *testing.T) {
	types := []ScopeType{
		ScopeTypeAgent, ScopeTypeFunction, ScopeTypeTool, ScopeTypeLlm,
		ScopeTypeRetriever, ScopeTypeEmbedder, ScopeTypeReranker,
		ScopeTypeGuardrail, ScopeTypeEvaluator, ScopeTypeCustom, ScopeTypeUnknown,
	}
	for _, st := range types {
		handle, err := PushScope("type_test", st)
		if err != nil {
			t.Fatalf("PushScope with type %d failed: %v", st, err)
		}
		PopScope(handle)
	}
}

// ============================================================================
// Events
// ============================================================================

func TestEmitEvent(t *testing.T) {
	err := EmitEvent("my_mark")
	if err != nil {
		t.Fatalf("EmitEvent failed: %v", err)
	}
}

func TestEmitEventWithData(t *testing.T) {
	err := EmitEvent("data_mark",
		WithEventData(json.RawMessage(`{"key": "value"}`)),
		WithEventMetadata(json.RawMessage(`{"version": 1}`)),
	)
	if err != nil {
		t.Fatalf("EmitEvent with data failed: %v", err)
	}
}

func TestEmitEventWithParent(t *testing.T) {
	handle, _ := PushScope("evt_scope", ScopeTypeAgent)
	defer PopScope(handle)

	err := EmitEvent("scoped_mark", WithEventParent(handle))
	if err != nil {
		t.Fatalf("EmitEvent with parent failed: %v", err)
	}
}

// ============================================================================
// Subscribers
// ============================================================================

func TestSubscriberRegistration(t *testing.T) {
	count := 0
	var mu sync.Mutex
	err := RegisterSubscriber("go_test_sub", func(event *Event) {
		mu.Lock()
		count++
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}

	// Push scope emits start event
	handle, _ := PushScope("s", ScopeTypeFunction)
	PopScope(handle)

	mu.Lock()
	c := count
	mu.Unlock()
	if c < 2 {
		t.Fatalf("expected at least 2 events (start+end), got %d", c)
	}

	err = DeregisterSubscriber("go_test_sub")
	if err != nil {
		t.Fatalf("DeregisterSubscriber failed: %v", err)
	}
}

func TestDuplicateSubscriberFails(t *testing.T) {
	RegisterSubscriber("go_dup_sub", func(event *Event) {})
	err := RegisterSubscriber("go_dup_sub", func(event *Event) {})
	if err == nil {
		t.Fatal("expected error for duplicate subscriber")
	}
	DeregisterSubscriber("go_dup_sub")
}

func TestSubscriberEventProperties(t *testing.T) {
	var events []struct {
		uuid      string
		name      string
		eventType EventType
		timestamp string
	}
	var mu sync.Mutex

	RegisterSubscriber("go_evt_props", func(event *Event) {
		mu.Lock()
		events = append(events, struct {
			uuid      string
			name      string
			eventType EventType
			timestamp string
		}{
			uuid:      event.UUID(),
			name:      event.Name(),
			eventType: event.Type(),
			timestamp: event.Timestamp(),
		})
		mu.Unlock()
	})

	handle, _ := PushScope("prop_test", ScopeTypeAgent)
	PopScope(handle)
	DeregisterSubscriber("go_evt_props")

	mu.Lock()
	defer mu.Unlock()
	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}
	if events[0].eventType != EventTypeStart {
		t.Fatalf("expected Start event, got %d", events[0].eventType)
	}
	if events[0].uuid == "" {
		t.Fatal("event UUID is empty")
	}
	if events[0].timestamp == "" {
		t.Fatal("event timestamp is empty")
	}
}

func TestMarkEvent(t *testing.T) {
	var markSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_mark_sub", func(event *Event) {
		if event.Type() == EventTypeMark {
			mu.Lock()
			markSeen = true
			mu.Unlock()
		}
	})

	EmitEvent("test_mark", WithEventData(json.RawMessage(`{"info": "test"}`)))
	DeregisterSubscriber("go_mark_sub")

	mu.Lock()
	if !markSeen {
		t.Fatal("mark event was not received")
	}
	mu.Unlock()
}

// ============================================================================
// Tool lifecycle
// ============================================================================

func TestToolCallAndEnd(t *testing.T) {
	handle, err := ToolCall("my_tool", json.RawMessage(`{"input": "data"}`))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle == nil {
		t.Fatal("returned nil handle")
	}
	if handle.Name() != "my_tool" {
		t.Fatalf("expected 'my_tool', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("UUID is empty")
	}

	err = ToolCallEnd(handle, json.RawMessage(`{"output": "result"}`))
	if err != nil {
		t.Fatalf("ToolCallEnd failed: %v", err)
	}
}

func TestToolCallWithAttributes(t *testing.T) {
	handle, err := ToolCall("local_tool", json.RawMessage(`{}`), WithToolAttributes(ToolAttrLocal))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle.Attributes()&ToolAttrLocal == 0 {
		t.Fatal("expected LOCAL attribute")
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolCallWithDataMetadata(t *testing.T) {
	handle, err := ToolCall("tool_dm", json.RawMessage(`{"arg": 1}`),
		WithToolData(json.RawMessage(`{"custom": "info"}`)),
		WithToolMetadata(json.RawMessage(`{"trace_id": "abc123"}`)),
	)
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	ToolCallEnd(handle, json.RawMessage(`{}`),
		WithToolData(json.RawMessage(`{"end_data": true}`)),
		WithToolMetadata(json.RawMessage(`{"end_meta": true}`)),
	)
}

func TestToolCallWithParent(t *testing.T) {
	parent, _ := PushScope("tool_parent", ScopeTypeAgent)
	defer PopScope(parent)

	handle, err := ToolCall("child_tool", json.RawMessage(`{}`), WithToolParent(parent))
	if err != nil {
		t.Fatalf("ToolCall failed: %v", err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	ToolCallEnd(handle, json.RawMessage(`{}`))
}

func TestToolEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_tool_evt", func(event *Event) {
		mu.Lock()
		if event.Type() == EventTypeStart {
			startSeen = true
		}
		if event.Type() == EventTypeEnd {
			endSeen = true
		}
		mu.Unlock()
	})

	handle, _ := ToolCall("evt_tool", json.RawMessage(`{}`))
	ToolCallEnd(handle, json.RawMessage(`{}`))
	DeregisterSubscriber("go_tool_evt")

	mu.Lock()
	if !startSeen {
		t.Fatal("start event not seen")
	}
	if !endSeen {
		t.Fatal("end event not seen")
	}
	mu.Unlock()
}

// ============================================================================
// Tool execute
// ============================================================================

func TestToolCallExecuteBasic(t *testing.T) {
	fn := func(args json.RawMessage) (json.RawMessage, error) {
		var input map[string]interface{}
		json.Unmarshal(args, &input)
		x := input["x"].(float64)
		result, _ := json.Marshal(map[string]interface{}{"result": x * 2})
		return result, nil
	}

	result, err := ToolCallExecute("double", json.RawMessage(`{"x": 5}`), fn)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["result"].(float64) != 10 {
		t.Fatalf("expected 10, got %v", output["result"])
	}
}

func TestToolCallExecuteWithAttributes(t *testing.T) {
	fn := func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	}

	result, err := ToolCallExecute("attr_tool", json.RawMessage(`{"test": true}`), fn,
		WithToolAttributes(ToolAttrLocal),
	)
	if err != nil {
		t.Fatalf("ToolCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["test"] != true {
		t.Fatalf("expected test=true, got %v", output["test"])
	}
}

// ============================================================================
// Tool guardrails
// ============================================================================

func TestToolSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterToolSanitizeRequestGuardrail("go_san_req", 1,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["sanitized"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolSanitizeRequestGuardrail("go_san_req")
}

func TestToolSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterToolSanitizeResponseGuardrail("go_san_resp", 1,
		func(name string, result json.RawMessage) json.RawMessage {
			return result
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolSanitizeResponseGuardrail("go_san_resp")
}

func TestToolConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterToolConditionalExecutionGuardrail("go_cond", 1,
		func(name string, args json.RawMessage) *string {
			return nil // pass
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolConditionalExecutionGuardrail("go_cond")
}

func TestDuplicateGuardrailFails(t *testing.T) {
	RegisterToolSanitizeRequestGuardrail("go_dup_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	err := RegisterToolSanitizeRequestGuardrail("go_dup_guard", 1,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err == nil {
		t.Fatal("expected error for duplicate guardrail")
	}
	DeregisterToolSanitizeRequestGuardrail("go_dup_guard")
}

func TestToolConditionalBlocksExecution(t *testing.T) {
	msg := "blocked by policy"
	RegisterToolConditionalExecutionGuardrail("go_blocker", 1,
		func(name string, args json.RawMessage) *string {
			return &msg
		},
	)

	_, err := ToolCallExecute("blocked_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"should": "not reach"}`), nil
		},
	)
	if err == nil {
		t.Fatal("expected error from guardrail rejection")
	}
	if !strings.Contains(err.Error(), "guardrail rejected") {
		t.Fatalf("expected 'guardrail rejected' error, got: %v", err)
	}

	DeregisterToolConditionalExecutionGuardrail("go_blocker")
}

// ============================================================================
// Tool intercepts
// ============================================================================

func TestToolRequestInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolRequestIntercept("go_req_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolRequestIntercept("go_req_int")
}

func TestToolResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolResponseIntercept("go_resp_int", 1, false,
		func(name string, result json.RawMessage) json.RawMessage { return result },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolResponseIntercept("go_resp_int")
}

func TestToolExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterToolExecutionIntercept("go_exec_int", 1,
		func(name string, args json.RawMessage) bool { return false },
		func(args json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{}`), nil },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterToolExecutionIntercept("go_exec_int")
}

func TestDuplicateInterceptFails(t *testing.T) {
	RegisterToolRequestIntercept("go_dup_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	err := RegisterToolRequestIntercept("go_dup_int", 1, false,
		func(name string, args json.RawMessage) json.RawMessage { return args },
	)
	if err == nil {
		t.Fatal("expected error for duplicate intercept")
	}
	DeregisterToolRequestIntercept("go_dup_int")
}

func TestToolRequestInterceptModifiesArgs(t *testing.T) {
	RegisterToolRequestIntercept("go_req_mod", 1, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["intercepted"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)

	result, err := ToolCallExecute("intercepted_tool", json.RawMessage(`{"original": true}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return args, nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true || output["intercepted"] != true {
		t.Fatalf("expected both original and intercepted, got %v", output)
	}

	DeregisterToolRequestIntercept("go_req_mod")
}

func TestToolResponseInterceptModifiesResult(t *testing.T) {
	RegisterToolResponseIntercept("go_resp_mod", 1, false,
		func(name string, result json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(result, &m)
			m["post_processed"] = true
			out, _ := json.Marshal(m)
			return out
		},
	)

	result, err := ToolCallExecute("resp_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"output": "raw"}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["output"] != "raw" || output["post_processed"] != true {
		t.Fatalf("expected output + post_processed, got %v", output)
	}

	DeregisterToolResponseIntercept("go_resp_mod")
}

func TestToolExecutionInterceptReplacesFunc(t *testing.T) {
	RegisterToolExecutionIntercept("go_exec_replace", 1,
		func(name string, args json.RawMessage) bool { return true },
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	result, err := ToolCallExecute("replaced_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["from_intercept"] != true {
		t.Fatalf("expected from_intercept, got %v", output)
	}
	if _, ok := output["from_original"]; ok {
		t.Fatal("should not contain from_original")
	}

	DeregisterToolExecutionIntercept("go_exec_replace")
}

func TestToolRequestInterceptBreakChain(t *testing.T) {
	RegisterToolRequestIntercept("go_chain1", 1, true, // break_chain=true
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["from_first"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)
	RegisterToolRequestIntercept("go_chain2", 2, false,
		func(name string, args json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(args, &m)
			m["from_second"] = true
			result, _ := json.Marshal(m)
			return result
		},
	)

	result, err := ToolCallExecute("chain_tool", json.RawMessage(`{}`),
		func(args json.RawMessage) (json.RawMessage, error) { return args, nil },
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["from_first"] != true {
		t.Fatal("expected from_first")
	}
	if _, ok := output["from_second"]; ok {
		t.Fatal("should not contain from_second (chain was broken)")
	}

	DeregisterToolRequestIntercept("go_chain1")
	DeregisterToolRequestIntercept("go_chain2")
}

// ============================================================================
// LLM lifecycle
// ============================================================================

func makeRequest() *LLMRequest {
	return NewLLMRequest("POST", "https://api.example.com",
		map[string]interface{}{}, map[string]interface{}{"messages": []string{}},
	)
}

func TestLlmCallAndEnd(t *testing.T) {
	req := makeRequest()
	handle, err := LlmCall("my_llm", req)
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
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
	req := makeRequest()
	handle, err := LlmCall("streaming_llm", req, WithLLMAttributes(LLMAttrStreaming))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle.Attributes()&LLMAttrStreaming == 0 {
		t.Fatal("expected STREAMING attribute")
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmCallWithDataMetadata(t *testing.T) {
	req := makeRequest()
	handle, err := LlmCall("llm_dm", req,
		WithLLMData(json.RawMessage(`{"custom": "data"}`)),
		WithLLMMetadata(json.RawMessage(`{"trace": "xyz"}`)),
	)
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	LlmCallEnd(handle, json.RawMessage(`{}`),
		WithLLMData(json.RawMessage(`{"end": true}`)),
	)
}

func TestLlmCallWithParent(t *testing.T) {
	parent, _ := PushScope("llm_parent", ScopeTypeAgent)
	defer PopScope(parent)

	req := makeRequest()
	handle, err := LlmCall("child_llm", req, WithLLMParent(parent))
	if err != nil {
		t.Fatalf("LlmCall failed: %v", err)
	}
	if handle.ParentUUID() != parent.UUID() {
		t.Fatalf("expected parent UUID %s, got %s", parent.UUID(), handle.ParentUUID())
	}
	LlmCallEnd(handle, json.RawMessage(`{}`))
}

func TestLlmEvents(t *testing.T) {
	var startSeen, endSeen bool
	var mu sync.Mutex

	RegisterSubscriber("go_llm_evt", func(event *Event) {
		mu.Lock()
		if event.Type() == EventTypeStart {
			startSeen = true
		}
		if event.Type() == EventTypeEnd {
			endSeen = true
		}
		mu.Unlock()
	})

	req := makeRequest()
	handle, _ := LlmCall("evt_llm", req)
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
	req := makeRequest()
	result, err := LlmCallExecute("exec_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			out, _ := json.Marshal(map[string]interface{}{"model": url})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("LlmCallExecute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["model"] != "https://api.example.com" {
		t.Fatalf("expected url, got %v", output["model"])
	}
}

// ============================================================================
// LLM guardrails
// ============================================================================

func TestLlmSanitizeRequestGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_san_req", 1,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeRequestGuardrail("go_llm_san_req")
}

func TestLlmSanitizeResponseGuardrail(t *testing.T) {
	err := RegisterLlmSanitizeResponseGuardrail("go_llm_san_resp", 1,
		func(value json.RawMessage) json.RawMessage { return value },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmSanitizeResponseGuardrail("go_llm_san_resp")
}

func TestLlmConditionalExecutionGuardrail(t *testing.T) {
	err := RegisterLlmConditionalExecutionGuardrail("go_llm_cond", 1,
		func(method, url string, headers, body json.RawMessage) *string {
			return nil // pass
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmConditionalExecutionGuardrail("go_llm_cond")
}

func TestLlmDuplicateGuardrailFails(t *testing.T) {
	RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	err := RegisterLlmSanitizeRequestGuardrail("go_llm_dup", 1,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
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
		func(method, url string, headers, body json.RawMessage) *string {
			return &msg
		},
	)

	req := makeRequest()
	_, err := LlmCallExecute("blocked_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
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
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, url, headers, body
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmRequestIntercept("go_llm_req")
}

func TestLlmResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmResponseIntercept("go_llm_resp", 1, false,
		func(value json.RawMessage) json.RawMessage { return value },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmResponseIntercept("go_llm_resp")
}

func TestLlmStreamResponseInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamResponseIntercept("go_llm_sr", 1, false,
		func(sseJSON json.RawMessage) json.RawMessage { return sseJSON },
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamResponseIntercept("go_llm_sr")
}

func TestLlmExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmExecutionIntercept("go_llm_exec", 1,
		func(method, url string, headers, body json.RawMessage) bool { return false },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmExecutionIntercept("go_llm_exec")
}

func TestLlmStreamExecutionInterceptRegisterDeregister(t *testing.T) {
	err := RegisterLlmStreamExecutionIntercept("go_llm_sexec", 1,
		func(method, url string, headers, body json.RawMessage) bool { return false },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		},
	)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	DeregisterLlmStreamExecutionIntercept("go_llm_sexec")
}

func TestLlmRequestInterceptModifies(t *testing.T) {
	RegisterLlmRequestIntercept("go_llm_req_mod", 1, false,
		func(method, url string, headers, body json.RawMessage) (string, string, json.RawMessage, json.RawMessage) {
			return method, "https://intercepted.com", headers, body
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("int_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			out, _ := json.Marshal(map[string]interface{}{"called_url": url})
			return out, nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["called_url"] != "https://intercepted.com" {
		t.Fatalf("expected intercepted URL, got %v", output["called_url"])
	}

	DeregisterLlmRequestIntercept("go_llm_req_mod")
}

func TestLlmResponseInterceptModifies(t *testing.T) {
	RegisterLlmResponseIntercept("go_llm_resp_mod", 1, false,
		func(value json.RawMessage) json.RawMessage {
			var m map[string]interface{}
			json.Unmarshal(value, &m)
			m["modified"] = true
			out, _ := json.Marshal(m)
			return out
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("resp_llm", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
	}

	var output map[string]interface{}
	json.Unmarshal(result, &output)
	if output["original"] != true || output["modified"] != true {
		t.Fatalf("expected both original and modified, got %v", output)
	}

	DeregisterLlmResponseIntercept("go_llm_resp_mod")
}

func TestLlmExecutionInterceptReplaces(t *testing.T) {
	RegisterLlmExecutionIntercept("go_llm_exec_rep", 1,
		func(method, url string, headers, body json.RawMessage) bool { return true },
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_intercept": true}`), nil
		},
	)

	req := makeRequest()
	result, err := LlmCallExecute("exec_llm_rep", req,
		func(method, url string, headers, body json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"from_original": true}`), nil
		},
	)
	if err != nil {
		t.Fatalf("execute failed: %v", err)
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

// ============================================================================
// Deregister nonexistent
// ============================================================================

func TestDeregisterNonexistentGuardrails(t *testing.T) {
	// These should not panic, but they may return errors
	DeregisterToolSanitizeRequestGuardrail("nonexistent")
	DeregisterToolSanitizeResponseGuardrail("nonexistent")
	DeregisterToolConditionalExecutionGuardrail("nonexistent")
	DeregisterLlmSanitizeRequestGuardrail("nonexistent")
	DeregisterLlmSanitizeResponseGuardrail("nonexistent")
	DeregisterLlmConditionalExecutionGuardrail("nonexistent")
}

func TestDeregisterNonexistentIntercepts(t *testing.T) {
	DeregisterToolRequestIntercept("nonexistent")
	DeregisterToolResponseIntercept("nonexistent")
	DeregisterToolExecutionIntercept("nonexistent")
	DeregisterLlmRequestIntercept("nonexistent")
	DeregisterLlmResponseIntercept("nonexistent")
	DeregisterLlmStreamResponseIntercept("nonexistent")
	DeregisterLlmExecutionIntercept("nonexistent")
	DeregisterLlmStreamExecutionIntercept("nonexistent")
}
