// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package scope provides shorthand access to Nexus scope operations.
//
// It re-exports the core scope management functions (GetHandle, PushScope,
// PopScope, EmitEvent) under shorter names for convenience.
//
// Example usage:
//
//	import "gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus/scope"
//
//	// Push a new agent scope onto the stack.
//	handle, err := scope.Push("my-agent", nat_nexus.ScopeTypeAgent)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer scope.Pop(handle)
//
//	// Emit a mark event within the current scope.
//	_ = scope.Event("checkpoint-reached")
package scope

import (
	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty. This is a shorthand for
// [nat_nexus.GetHandle].
func GetHandle() (*nat_nexus.ScopeHandle, error) {
	return nat_nexus.GetHandle()
}

// Push creates a new scope and pushes it onto the hierarchical scope stack,
// emitting a Start event to all registered subscribers. Use [Pop] to end the
// scope. This is a shorthand for [nat_nexus.PushScope].
func Push(name string, scopeType nat_nexus.ScopeType, opts ...nat_nexus.ScopeOption) (*nat_nexus.ScopeHandle, error) {
	return nat_nexus.PushScope(name, scopeType, opts...)
}

// Pop removes the given scope from the scope stack and emits an End event to
// all registered subscribers. This is a shorthand for [nat_nexus.PopScope].
func Pop(handle *nat_nexus.ScopeHandle) error {
	return nat_nexus.PopScope(handle)
}

// Event emits an instantaneous Mark event within the current scope. This is a
// shorthand for [nat_nexus.EmitEvent].
func Event(name string, opts ...nat_nexus.EventOption) error {
	return nat_nexus.EmitEvent(name, opts...)
}

// WithScope pushes a new scope and returns a cleanup function that pops it.
// The cleanup function is safe to call even if the push failed (it becomes a
// no-op). Use with defer for automatic scope cleanup:
//
//	defer scope.WithScope("name", nat_nexus.ScopeTypeAgent)()
//
// Or capture the cleanup explicitly:
//
//	cleanup := scope.WithScope("name", nat_nexus.ScopeTypeAgent)
//	defer cleanup()
func WithScope(name string, scopeType nat_nexus.ScopeType, opts ...nat_nexus.ScopeOption) func() {
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
//	handle, cleanup := scope.WithScopeHandle("name", nat_nexus.ScopeTypeAgent)
//	defer cleanup()
//	if handle != nil {
//	    // use handle
//	}
func WithScopeHandle(name string, scopeType nat_nexus.ScopeType, opts ...nat_nexus.ScopeOption) (*nat_nexus.ScopeHandle, func()) {
	handle, err := Push(name, scopeType, opts...)
	if err != nil {
		return nil, func() {} // noop cleanup on push failure
	}
	return handle, func() {
		Pop(handle)
	}
}
