// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package scope provides shorthand access to NeMo Flow scope operations.
//
// It re-exports the core scope management functions (GetHandle, PushScope,
// PopScope, EmitEvent) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/scope"
//
//	// Push a new agent scope onto the stack.
//	handle, err := scope.Push("my-agent", nemo_flow.ScopeTypeAgent)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer scope.Pop(handle)
//
//	// Emit a mark event within the current scope.
//	_ = scope.Event("checkpoint-reached")
package scope

import (
	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty. This is a shorthand for
// [nemo_flow.GetHandle].
func GetHandle() (*nemo_flow.ScopeHandle, error) {
	return nemo_flow.GetHandle()
}

// Push creates a new scope and pushes it onto the hierarchical scope stack,
// emitting a Start event to all registered subscribers. Use [Pop] to end the
// scope. This is a shorthand for [nemo_flow.PushScope].
func Push(name string, scopeType nemo_flow.ScopeType, opts ...nemo_flow.ScopeOption) (*nemo_flow.ScopeHandle, error) {
	return nemo_flow.PushScope(name, scopeType, opts...)
}

// Pop removes the given scope from the scope stack and emits an End event to
// all registered subscribers. This is a shorthand for [nemo_flow.PopScope].
func Pop(handle *nemo_flow.ScopeHandle) error {
	return nemo_flow.PopScope(handle)
}

// Event emits an instantaneous Mark event within the current scope. This is a
// shorthand for [nemo_flow.EmitEvent].
func Event(name string, opts ...nemo_flow.EventOption) error {
	return nemo_flow.EmitEvent(name, opts...)
}

// WithScope pushes a new scope and returns a cleanup function that pops it.
// The cleanup function is safe to call even if the push failed (it becomes a
// no-op). Use with defer for automatic scope cleanup:
//
//	defer scope.WithScope("name", nemo_flow.ScopeTypeAgent)()
//
// Or capture the cleanup explicitly:
//
//	cleanup := scope.WithScope("name", nemo_flow.ScopeTypeAgent)
//	defer cleanup()
func WithScope(name string, scopeType nemo_flow.ScopeType, opts ...nemo_flow.ScopeOption) func() {
	handle, err := Push(name, scopeType, opts...)
	if err != nil {
		return func() {} // noop cleanup on push failure
	}
	return func() {
		Pop(handle)
	}
}

// WithScopeHandle pushes a new scope and returns both the scope handle and a
// cleanup function. If the push fails, handle is nil and the cleanup function
// is a no-op. Use with defer for automatic scope cleanup when you also need
// access to the scope handle:
//
//	handle, cleanup := scope.WithScopeHandle("name", nemo_flow.ScopeTypeAgent)
//	defer cleanup()
//	if handle != nil {
//	    // use handle
//	}
func WithScopeHandle(name string, scopeType nemo_flow.ScopeType, opts ...nemo_flow.ScopeOption) (*nemo_flow.ScopeHandle, func()) {
	handle, err := Push(name, scopeType, opts...)
	if err != nil {
		return nil, func() {} // noop cleanup on push failure
	}
	return handle, func() {
		Pop(handle)
	}
}
