// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package subscribers provides shorthand access to NeMo Relay event subscriber
// registration.
//
// Subscribers receive discriminated lifecycle events emitted by the runtime as
// scopes, tool calls, and LLM calls progress. Subscribers can be named for
// explicit deregistration or anonymous with a closeable subscription handle.
//
// Example usage:
//
//	import (
//	    "fmt"
//
//	    "github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
//	    "github.com/NVIDIA/NeMo-Relay/go/nemo_relay/subscribers"
//	)
//
//	// Subscribe logs every event until the handle is closed.
//	sub, err := subscribers.Subscribe(func(event nemo_relay.Event) {
//	    fmt.Printf("[%s] %s: %s\n", event.Timestamp(), event.Kind(), event.Name())
//	})
//
//	// Later, remove it.
//	_, _ = sub.Close()
package subscribers

import (
	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

// Subscribe registers an anonymous event subscriber and returns a closeable
// handle. This is a shorthand for [nemo_relay.Subscribe].
func Subscribe(fn nemo_relay.EventSubscriberFunc) (*nemo_relay.Subscription, error) {
	return nemo_relay.Subscribe(fn)
}

// Register registers a named event subscriber that will be called for every
// lifecycle event emitted by the runtime. The name must be unique;
// registering a duplicate returns an AlreadyExists error. The callback
// receives an owned [nemo_relay.Event] snapshot that is safe to retain after
// the callback returns. This is a shorthand for
// [nemo_relay.RegisterSubscriber].
func Register(name string, fn nemo_relay.EventSubscriberFunc) error {
	return nemo_relay.RegisterSubscriber(name, fn)
}

// Deregister removes a named event subscriber. Returns a NotFound error if no
// subscriber with the given name is registered. This is a shorthand for
// [nemo_relay.DeregisterSubscriber].
func Deregister(name string) error {
	return nemo_relay.DeregisterSubscriber(name)
}

// ScopeSubscribe registers an anonymous scope-local event subscriber and
// returns a closeable handle. This is a shorthand for
// [nemo_relay.ScopeSubscribe].
func ScopeSubscribe(scopeUUID string, fn nemo_relay.EventSubscriberFunc) (*nemo_relay.Subscription, error) {
	return nemo_relay.ScopeSubscribe(scopeUUID, fn)
}

// ScopeRegister registers a scope-local event subscriber that will be called
// for lifecycle events within the given scope. This is a shorthand for
// [nemo_relay.ScopeRegisterSubscriber].
func ScopeRegister(scopeUUID, name string, fn nemo_relay.EventSubscriberFunc) error {
	return nemo_relay.ScopeRegisterSubscriber(scopeUUID, name, fn)
}

// ScopeDeregister removes a scope-local event subscriber by name. This is a
// shorthand for [nemo_relay.ScopeDeregisterSubscriber].
func ScopeDeregister(scopeUUID, name string) error {
	return nemo_relay.ScopeDeregisterSubscriber(scopeUUID, name)
}
