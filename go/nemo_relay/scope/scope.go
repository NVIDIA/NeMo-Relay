// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package scope provides shorthand access to NeMo Relay scope operations.
//
// It re-exports the core scope management functions (GetHandle, PushScope,
// PopScope, EmitEvent) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/scope"
//
//	// Push a new agent scope onto the stack.
//	handle, err := scope.Push("my-agent", nemo_relay.ScopeTypeAgent)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer scope.Pop(handle)
//
//	// Emit a mark event within the current scope.
//	_ = scope.Event("checkpoint-reached")
package scope

import (
	"encoding/json"
	"fmt"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

func statusMetadata(statusCode, statusMessage string) nemo_relay.ScopeEndOption {
	metadata := map[string]string{"otel.status_code": statusCode}
	if statusMessage != "" {
		metadata["otel.status_description"] = statusMessage
	}
	raw, _ := json.Marshal(metadata)
	return nemo_relay.WithScopeEndMetadata(raw)
}

func cleanupScope(handle *nemo_relay.ScopeHandle) func() {
	cleaned := false
	return func() {
		if cleaned {
			return
		}
		cleaned = true
		if recovered := recover(); recovered != nil {
			_ = Pop(handle, statusMetadata("ERROR", fmt.Sprint(recovered)))
			panic(recovered)
		}
		_ = Pop(handle, statusMetadata("OK", ""))
	}
}

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty. This is a shorthand for
// [nemo_relay.GetHandle].
func GetHandle() (*nemo_relay.ScopeHandle, error) {
	return nemo_relay.GetHandle()
}

// Push creates a new scope and pushes it onto the hierarchical scope stack,
// emitting a Start event to all registered subscribers. Use [Pop] to end the
// scope. Optional arguments, including [nemo_relay.WithScopeTimestamp], are
// forwarded to [nemo_relay.PushScope].
func Push(name string, scopeType nemo_relay.ScopeType, opts ...nemo_relay.ScopeOption) (*nemo_relay.ScopeHandle, error) {
	return nemo_relay.PushScope(name, scopeType, opts...)
}

// Pop removes the given scope from the scope stack and emits an End event to
// all registered subscribers. Optional arguments, including
// [nemo_relay.WithScopeEndMetadata] and [nemo_relay.WithScopeEndTimestamp],
// are forwarded to [nemo_relay.PopScope].
func Pop(handle *nemo_relay.ScopeHandle, opts ...nemo_relay.ScopeEndOption) error {
	return nemo_relay.PopScope(handle, opts...)
}

// Event emits an instantaneous Mark event within the current scope. This is a
// shorthand for [nemo_relay.EmitEvent]. Optional arguments, including
// [nemo_relay.WithEventTimestamp], are forwarded to [nemo_relay.EmitEvent].
func Event(name string, opts ...nemo_relay.EventOption) error {
	return nemo_relay.EmitEvent(name, opts...)
}

// WithScope pushes a new scope and returns a cleanup function that pops it.
// The cleanup function is safe to call even if the push failed (it becomes a
// no-op). Use with defer for automatic scope cleanup:
//
//	defer scope.WithScope("name", nemo_relay.ScopeTypeAgent)()
//
// Or capture the cleanup explicitly:
//
//	cleanup := scope.WithScope("name", nemo_relay.ScopeTypeAgent)
//	defer cleanup()
func WithScope(name string, scopeType nemo_relay.ScopeType, opts ...nemo_relay.ScopeOption) func() {
	handle, err := Push(name, scopeType, opts...)
	if err != nil {
		return func() {
			// Push failed, so cleanup is intentionally a no-op.
		}
	}
	return cleanupScope(handle)
}

// WithScopeHandle pushes a new scope and returns both the scope handle and a
// cleanup function. If the push fails, handle is nil and the cleanup function
// is a no-op. Use with defer for automatic scope cleanup when you also need
// access to the scope handle:
//
//	handle, cleanup := scope.WithScopeHandle("name", nemo_relay.ScopeTypeAgent)
//	defer cleanup()
//	if handle != nil {
//	    // use handle
//	}
func WithScopeHandle(name string, scopeType nemo_relay.ScopeType, opts ...nemo_relay.ScopeOption) (*nemo_relay.ScopeHandle, func()) {
	handle, err := Push(name, scopeType, opts...)
	if err != nil {
		return nil, func() {
			// Push failed, so cleanup is intentionally a no-op.
		}
	}
	return handle, cleanupScope(handle)
}
