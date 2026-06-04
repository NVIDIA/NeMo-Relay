// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package scope_test

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay/scope"
)

const (
	getHandleAfterFailed = "GetHandle after: %v"
	getHandleFailed      = "GetHandle: %v"
	expectedNonNilHandle = "expected non-nil handle"
)

func metadataStringField(t *testing.T, raw json.RawMessage, field string) string {
	t.Helper()

	var decoded map[string]interface{}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal metadata failed: %v; raw=%s", err, raw)
	}
	value, ok := decoded[field]
	if !ok {
		t.Fatalf("expected metadata field %q, got %v", field, decoded)
	}
	text, ok := value.(string)
	if !ok {
		t.Fatalf("expected metadata field %q to be string, got %T", field, value)
	}
	return text
}

func captureScopeEndMetadata(t *testing.T, subscriberName, scopeName string, fn func()) json.RawMessage {
	t.Helper()

	var captured json.RawMessage
	var mu sync.Mutex
	_ = nemo_relay.DeregisterSubscriber(subscriberName)
	if err := nemo_relay.RegisterSubscriber(subscriberName, func(event nemo_relay.Event) {
		if event.Kind() == "scope" && event.ScopeCategory() == "end" && event.Name() == scopeName {
			mu.Lock()
			captured = append(json.RawMessage(nil), event.Metadata()...)
			mu.Unlock()
		}
	}); err != nil {
		t.Fatalf("RegisterSubscriber failed: %v", err)
	}
	defer nemo_relay.DeregisterSubscriber(subscriberName)

	fn()

	if err := nemo_relay.FlushSubscribers(); err != nil {
		t.Fatalf("FlushSubscribers failed: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if captured == nil {
		t.Fatalf("expected end metadata for scope %q", scopeName)
	}
	return captured
}

// ============================================================================
// WithScope
// ============================================================================

func TestWithScopeNormalReturn(t *testing.T) {
	// Capture the current top-of-stack before pushing.
	before, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle before: %v", err)
	}

	// WithScope pushes and returns a cleanup function.
	cleanup := scope.WithScope("with_scope_test", nemo_relay.ScopeTypeAgent)
	defer cleanup()

	// While inside the scope, the top-of-stack should be our new scope.
	during, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle during: %v", err)
	}
	if during.Name() != "with_scope_test" {
		t.Fatalf("expected 'with_scope_test', got '%s'", during.Name())
	}

	// Call cleanup explicitly to verify double-cleanup is safe (defer will call it again).
	cleanup()

	// After cleanup the scope should be popped.
	after, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleAfterFailed, err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("expected stack to return to %s, got %s", before.UUID(), after.UUID())
	}
}

func TestWithScopeDeferCleanup(t *testing.T) {
	before, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleFailed, err)
	}

	func() {
		defer scope.WithScope("deferred_scope", nemo_relay.ScopeTypeFunction)()

		current, err := nemo_relay.GetHandle()
		if err != nil {
			t.Fatalf("GetHandle inside: %v", err)
		}
		if current.Name() != "deferred_scope" {
			t.Fatalf("expected 'deferred_scope', got '%s'", current.Name())
		}
	}()

	after, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleAfterFailed, err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not popped after defer")
	}
}

func TestWithScopeRecordsOKStatusMetadata(t *testing.T) {
	metadata := captureScopeEndMetadata(t, "go_scope_with_status_ok_sub", "with_scope_ok_status", func() {
		cleanup := scope.WithScope("with_scope_ok_status", nemo_relay.ScopeTypeFunction)
		cleanup()
	})

	if got := metadataStringField(t, metadata, "otel.status_code"); got != "OK" {
		t.Fatalf("expected otel.status_code=OK, got %q", got)
	}
}

func TestWithScopeCleanupOnPanic(t *testing.T) {
	before, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleFailed, err)
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic, got none")
			}
		}()
		defer scope.WithScope("panic_scope", nemo_relay.ScopeTypeTool)()

		// Verify the scope is pushed.
		current, _ := nemo_relay.GetHandle()
		if current.Name() != "panic_scope" {
			t.Fatalf("expected 'panic_scope', got '%s'", current.Name())
		}

		panic("test panic")
	}()

	// After recovering from panic, scope should be popped.
	after, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after panic: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not popped after panic")
	}
}

func TestWithScopeRecordsErrorStatusMetadataOnPanic(t *testing.T) {
	metadata := captureScopeEndMetadata(t, "go_scope_with_status_error_sub", "with_scope_error_status", func() {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			defer scope.WithScope("with_scope_error_status", nemo_relay.ScopeTypeTool)()
			panic("scope status failure")
		}()
	})

	if got := metadataStringField(t, metadata, "otel.status_code"); got != "ERROR" {
		t.Fatalf("expected otel.status_code=ERROR, got %q", got)
	}
	if got := metadataStringField(t, metadata, "otel.status_message"); !strings.Contains(got, "scope status failure") {
		t.Fatalf("expected status message to mention panic, got %q", got)
	}
}

// ============================================================================
// WithScopeHandle
// ============================================================================

func TestWithScopeHandleNormalReturn(t *testing.T) {
	handle, cleanup := scope.WithScopeHandle("handle_test", nemo_relay.ScopeTypeAgent)
	defer cleanup()

	if handle == nil {
		t.Fatal(expectedNonNilHandle)
	}
	if handle.Name() != "handle_test" {
		t.Fatalf("expected 'handle_test', got '%s'", handle.Name())
	}
	if handle.UUID() == "" {
		t.Fatal("expected non-empty UUID")
	}
	if handle.Type() != nemo_relay.ScopeTypeAgent {
		t.Fatalf("expected ScopeTypeAgent, got %d", handle.Type())
	}
}

func TestWithScopeHandleCleanupOnPanic(t *testing.T) {
	before, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleFailed, err)
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic")
			}
		}()
		handle, cleanup := scope.WithScopeHandle("panic_handle", nemo_relay.ScopeTypeFunction)
		defer cleanup()

		if handle == nil {
			t.Fatal(expectedNonNilHandle)
		}

		panic("test panic")
	}()

	after, err := nemo_relay.GetHandle()
	if err != nil {
		t.Fatalf(getHandleAfterFailed, err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("scope not cleaned up after panic")
	}
}

func TestWithScopeWithOptions(t *testing.T) {
	handle, cleanup := scope.WithScopeHandle(
		"opts_test",
		nemo_relay.ScopeTypeFunction,
		nemo_relay.WithScopeAttributes(nemo_relay.ScopeAttrParallel),
	)
	defer cleanup()

	if handle == nil {
		t.Fatal(expectedNonNilHandle)
	}
	if handle.Attributes()&nemo_relay.ScopeAttrParallel == 0 {
		t.Fatal("expected PARALLEL attribute to be set")
	}
}

func TestWithScopeNested(t *testing.T) {
	before, _ := nemo_relay.GetHandle()

	h1, cleanup1 := scope.WithScopeHandle("outer", nemo_relay.ScopeTypeAgent)
	defer cleanup1()

	h2, cleanup2 := scope.WithScopeHandle("inner", nemo_relay.ScopeTypeFunction)
	defer cleanup2()

	current, _ := nemo_relay.GetHandle()
	if current.Name() != "inner" {
		t.Fatalf("expected 'inner', got '%s'", current.Name())
	}

	// Pop inner
	cleanup2()
	current, _ = nemo_relay.GetHandle()
	if current.Name() != "outer" {
		t.Fatalf("expected 'outer', got '%s'", current.Name())
	}

	// Pop outer
	cleanup1()
	current, _ = nemo_relay.GetHandle()
	if current.UUID() != before.UUID() {
		t.Fatalf("expected root scope")
	}

	_ = h1
	_ = h2
}
