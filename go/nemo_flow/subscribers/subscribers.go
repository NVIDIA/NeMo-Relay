// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package subscribers provides shorthand access to NeMo Flow event subscriber
// registration.
//
// Subscribers receive discriminated lifecycle events emitted by the runtime as
// scopes, tool calls, and LLM calls progress. Each subscriber is identified by
// a unique name.
//
// Example usage:
//
//	import "github.com/NVIDIA/NeMo-Flow/go/nemo_flow/subscribers"
//
//	// Register a subscriber that logs every event.
//	err := subscribers.Register("logger", func(event nemo_flow.Event) {
//	    fmt.Printf("[%s] %s: %s\n", event.Timestamp(), event.Kind(), event.Name())
//	})
//
//	// Later, remove it.
//	_ = subscribers.Deregister("logger")
package subscribers

import (
	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

// Register registers a named event subscriber that will be called for every
// lifecycle event emitted by the runtime. The name must be unique;
// registering a duplicate returns an AlreadyExists error. The callback
// receives an [nemo_flow.Event] that is only valid for the duration of the
// call. This is a shorthand for [nemo_flow.RegisterSubscriber].
func Register(name string, fn nemo_flow.EventSubscriberFunc) error {
	return nemo_flow.RegisterSubscriber(name, fn)
}

// Deregister removes a named event subscriber. Returns a NotFound error if no
// subscriber with the given name is registered. This is a shorthand for
// [nemo_flow.DeregisterSubscriber].
func Deregister(name string) error {
	return nemo_flow.DeregisterSubscriber(name)
}

// ScopeRegister registers a scope-local event subscriber that will be called
// for lifecycle events within the given scope. This is a shorthand for
// [nemo_flow.ScopeRegisterSubscriber].
func ScopeRegister(scopeUUID string, name string, fn nemo_flow.EventSubscriberFunc) error {
	return nemo_flow.ScopeRegisterSubscriber(scopeUUID, name, fn)
}

// ScopeDeregister removes a scope-local event subscriber by name. This is a
// shorthand for [nemo_flow.ScopeDeregisterSubscriber].
func ScopeDeregister(scopeUUID string, name string) error {
	return nemo_flow.ScopeDeregisterSubscriber(scopeUUID, name)
}
