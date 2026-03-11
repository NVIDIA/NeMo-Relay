// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package scope provides shorthand access to NVMagic scope operations.
//
// It re-exports the core scope management functions (GetHandle, PushScope,
// PopScope, EmitEvent) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvmagic/go/nvmagic/scope"
//
//	// Push a new agent scope onto the stack.
//	handle, err := scope.Push("my-agent", nvmagic.ScopeTypeAgent)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer scope.Pop(handle)
//
//	// Emit a mark event within the current scope.
//	_ = scope.Event("checkpoint-reached")
package scope

import (
	"github.com/nvidia/nvmagic/go/nvmagic"
)

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty. This is a shorthand for
// [nvmagic.GetHandle].
func GetHandle() (*nvmagic.ScopeHandle, error) {
	return nvmagic.GetHandle()
}

// Push creates a new scope and pushes it onto the hierarchical scope stack,
// emitting a Start event to all registered subscribers. Use [Pop] to end the
// scope. This is a shorthand for [nvmagic.PushScope].
func Push(name string, scopeType nvmagic.ScopeType, opts ...nvmagic.ScopeOption) (*nvmagic.ScopeHandle, error) {
	return nvmagic.PushScope(name, scopeType, opts...)
}

// Pop removes the given scope from the scope stack and emits an End event to
// all registered subscribers. This is a shorthand for [nvmagic.PopScope].
func Pop(handle *nvmagic.ScopeHandle) error {
	return nvmagic.PopScope(handle)
}

// Event emits an instantaneous Mark event within the current scope. This is a
// shorthand for [nvmagic.EmitEvent].
func Event(name string, opts ...nvmagic.EventOption) error {
	return nvmagic.EmitEvent(name, opts...)
}
