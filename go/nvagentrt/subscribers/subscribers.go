// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package subscribers provides shorthand access to NVAgentRT event subscriber
// registration.
//
// Subscribers receive lifecycle events (Start, End, Mark) emitted by the
// runtime as scopes, tool calls, and LLM calls progress. Each subscriber is
// identified by a unique name.
//
// Example usage:
//
//	import "github.com/nvidia/nvagentrt/go/nvagentrt/subscribers"
//
//	// Register a subscriber that logs every event.
//	err := subscribers.Register("logger", func(event *nvagentrt.Event) {
//	    fmt.Printf("[%s] %s: %s\n", event.Timestamp(), event.Type(), event.Name())
//	})
//
//	// Later, remove it.
//	_ = subscribers.Deregister("logger")
package subscribers

import (
	"github.com/nvidia/nvagentrt/go/nvagentrt"
)

// Register registers a named event subscriber that will be called for every
// lifecycle event (Start, End, Mark) emitted by the runtime. The name must be
// unique; registering a duplicate returns an AlreadyExists error. The callback
// receives an [nvagentrt.Event] pointer that is only valid for the duration of
// the call. This is a shorthand for [nvagentrt.RegisterSubscriber].
func Register(name string, fn nvagentrt.EventSubscriberFunc) error {
	return nvagentrt.RegisterSubscriber(name, fn)
}

// Deregister removes a named event subscriber. Returns a NotFound error if no
// subscriber with the given name is registered. This is a shorthand for
// [nvagentrt.DeregisterSubscriber].
func Deregister(name string) error {
	return nvagentrt.DeregisterSubscriber(name)
}
