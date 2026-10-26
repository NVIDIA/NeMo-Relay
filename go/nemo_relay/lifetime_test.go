// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"runtime"
	"testing"
	"time"
)

// Observe finalization without freeing the native pointer during a failing
// regression test. Cleanup transfers ownership back to the normal finalizer.
func watchedScopeHandle(t *testing.T) (*ScopeHandle, <-chan struct{}) {
	t.Helper()
	h, err := GetHandle()
	if err != nil || h == nil {
		t.Fatalf("GetHandle: %v", err)
	}
	ptr := h.ptr
	t.Cleanup(func() { runtime.KeepAlive(newScopeHandle(ptr)) })
	finalized := make(chan struct{})
	runtime.SetFinalizer(h, nil)
	runtime.SetFinalizer(h, func(*ScopeHandle) { close(finalized) })
	return h, finalized
}

func forceLifetimeGC(t *testing.T, finalized <-chan struct{}) {
	t.Helper()
	for i := 0; i < 3; i++ {
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	select {
	case <-finalized:
		t.Error("wrapper finalized before native call completed")
	default:
	}
}

func awaitLifetimeFinalizer(t *testing.T, finalized <-chan struct{}) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		select {
		case <-finalized:
			return
		case <-time.After(time.Millisecond):
		}
	}
	t.Fatal("wrapper was not released after native call completed")
}

func TestParentHandleLifetimeDuringNativeCallbacks(t *testing.T) {
	for _, operation := range []string{"scope", "event", "tool", "tool_execute", "llm", "llm_execute", "llm_stream"} {
		t.Run(operation, func(t *testing.T) {
			runTestWithScopeStack(t, func(t *testing.T) {
				root, err := PushScope("lifetime-root", ScopeTypeAgent)
				if err != nil {
					t.Fatal(err)
				}
				defer PopScope(root)
				parent, finalized := watchedScopeHandle(t)
				calls := 0
				check := func() { calls++; forceLifetimeGC(t, finalized) }
				err = callWithEphemeralParent(operation, parent, check)
				parent = nil
				if err != nil {
					t.Fatal(err)
				}
				if calls == 0 {
					t.Fatal("native callback was not invoked")
				}
				if operation == "tool_execute" || operation == "llm_execute" || operation == "llm_stream" {
					awaitLifetimeFinalizer(t, finalized)
				}
			})
		})
	}
}

// Each option drops its captured wrapper after application. This exercises the
// lifetime of the materialized options, independent of the option closure.
func callWithEphemeralParent(operation string, parent *ScopeHandle, check func()) error {
	scopeOpt := func(o *scopeOptions) { WithParent(parent)(o); parent = nil; check() }
	eventOpt := func(o *eventOptions) { WithEventParent(parent)(o); parent = nil; check() }
	toolOpt := func(o *toolCallOptions) { WithToolParent(parent)(o); parent = nil; check() }
	llmOpt := func(o *llmCallOptions) { WithLLMParent(parent)(o); parent = nil; check() }
	request := json.RawMessage(`{"headers":{},"content":{}}`)
	switch operation {
	case "scope":
		h, err := PushScope("lifetime-child", ScopeTypeFunction, scopeOpt)
		if err != nil {
			return err
		}
		return PopScope(h)
	case "event":
		return EmitEvent("lifetime-mark", eventOpt)
	case "tool":
		h, err := ToolCall("lifetime-tool", json.RawMessage(`{}`), toolOpt)
		if err != nil {
			return err
		}
		return ToolCallEnd(h, toolExecutionResult(json.RawMessage(`{}`)))
	case "tool_execute":
		_, err := ToolCallExecute("lifetime-tool", json.RawMessage(`{}`), func(json.RawMessage) (ToolExecutionResult, error) {
			check()
			return toolExecutionResult(json.RawMessage(`{}`)), nil
		}, toolOpt)
		return err
	case "llm":
		h, err := LlmCall("lifetime-llm", request, llmOpt)
		if err != nil {
			return err
		}
		return LlmCallEnd(h, json.RawMessage(`{}`))
	case "llm_execute":
		_, err := LlmCallExecute("lifetime-llm", request, func(json.RawMessage) (json.RawMessage, error) {
			check()
			return json.RawMessage(`{}`), nil
		}, llmOpt)
		return err
	default:
		s, err := LlmStreamCallExecute("lifetime-llm", request, func(json.RawMessage) (json.RawMessage, error) {
			check()
			return json.RawMessage(`"data: {}\n\n"`), nil
		}, nil, nil, llmOpt)
		if err != nil {
			return err
		}
		return s.Close()
	}
}

func TestNativeWrapperAccessorsUnderGC(t *testing.T) {
	runTestWithScopeStack(t, func(t *testing.T) {
		stop := make(chan struct{})
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				select {
				case <-stop:
					return
				default:
					runtime.GC()
				}
			}
		}()
		defer func() { close(stop); <-done }()
		for i := 0; i < 100; i++ {
			h, err := PushScope("gc-scope", ScopeTypeFunction)
			if err != nil {
				t.Fatal(err)
			}
			if h.UUID() == "" || h.Name() != "gc-scope" || h.Type() != ScopeTypeFunction {
				t.Fatal("invalid scope accessors")
			}
			_ = h.Attributes()
			_ = h.ParentUUID()
			_ = h.Data()
			_ = h.Metadata()
			tool, err := ToolCall("gc-tool", json.RawMessage(`{}`))
			if err != nil {
				t.Fatal(err)
			}
			if tool.UUID() == "" || tool.Name() != "gc-tool" {
				t.Fatal("invalid tool accessors")
			}
			_ = tool.Attributes()
			_ = tool.ParentUUID()
			if err := ToolCallEnd(tool, toolExecutionResult(json.RawMessage(`{}`))); err != nil {
				t.Fatal(err)
			}
			llm, err := LlmCall("gc-llm", json.RawMessage(`{"headers":{},"content":{}}`))
			if err != nil {
				t.Fatal(err)
			}
			if llm.UUID() == "" || llm.Name() != "gc-llm" {
				t.Fatal("invalid LLM accessors")
			}
			_ = llm.Attributes()
			_ = llm.ParentUUID()
			if err := LlmCallEnd(llm, json.RawMessage(`{}`)); err != nil {
				t.Fatal(err)
			}
			if string(NewLLMRequest(map[string]interface{}{}, map[string]interface{}{}).Headers()) != `{}` {
				t.Fatal("invalid request headers")
			}
			if string(NewLLMRequest(map[string]interface{}{}, map[string]interface{}{}).Content()) != `{}` {
				t.Fatal("invalid request content")
			}
			r, err := NewAdaptiveRuntime(testAdaptiveRuntimeConfig("openai"))
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Register(); err != nil {
				t.Fatal(err)
			}
			if err := r.BindScope(h); err != nil {
				t.Fatal(err)
			}
			if _, err := r.BuildCacheRequestFacts(CacheRequestFactsInput{
				Provider: "openai", RequestID: "00000000-0000-0000-0000-000000000403",
				AnnotatedRequest: json.RawMessage(`{"model":"gpt-4.1-mini","messages":[]}`), AgentID: "go-adaptive-openai",
			}); err != nil {
				t.Fatal(err)
			}
			if err := r.WaitForIdle(); err != nil {
				t.Fatal(err)
			}
			if _, err := r.Report(); err != nil {
				t.Fatal(err)
			}
			if err := r.Deregister(); err != nil {
				t.Fatal(err)
			}
			if err := r.Shutdown(); err != nil {
				t.Fatal(err)
			}
			if err := PopScope(h); err != nil {
				t.Fatal(err)
			}
		}
	})
}
