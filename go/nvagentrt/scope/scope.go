// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package scope provides shorthand access to NVAgentRT scope operations.
//
// It re-exports the core scope management functions (GetHandle, PushScope,
// PopScope, EmitEvent) under shorter names for convenience.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/scope"
//
//	// Push a new agent scope onto the stack.
//	handle, err := scope.Push("my-agent", nvagentrt.ScopeTypeAgent)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer scope.Pop(handle)
//
//	// Emit a mark event within the current scope.
//	_ = scope.Event("checkpoint-reached")
package scope

import (
	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// GetHandle returns the handle for the scope currently at the top of the scope
// stack. Returns an error if the scope stack is empty. This is a shorthand for
// [nvagentrt.GetHandle].
func GetHandle() (*nvagentrt.ScopeHandle, error) {
	return nvagentrt.GetHandle()
}

// Push creates a new scope and pushes it onto the hierarchical scope stack,
// emitting a Start event to all registered subscribers. Use [Pop] to end the
// scope. This is a shorthand for [nvagentrt.PushScope].
func Push(name string, scopeType nvagentrt.ScopeType, opts ...nvagentrt.ScopeOption) (*nvagentrt.ScopeHandle, error) {
	return nvagentrt.PushScope(name, scopeType, opts...)
}

// Pop removes the given scope from the scope stack and emits an End event to
// all registered subscribers. This is a shorthand for [nvagentrt.PopScope].
func Pop(handle *nvagentrt.ScopeHandle) error {
	return nvagentrt.PopScope(handle)
}

// Event emits an instantaneous Mark event within the current scope. This is a
// shorthand for [nvagentrt.EmitEvent].
func Event(name string, opts ...nvagentrt.EventOption) error {
	return nvagentrt.EmitEvent(name, opts...)
}
