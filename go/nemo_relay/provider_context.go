// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"context"
	"sync"
	"time"
	"weak"
)

// Keep provider hooks separate from the cancellation signal. GC cleanup owns
// the signal, but must not keep hooks that can refer back to the stream alive.
type providerContext struct {
	context.Context
	signal      context.Context
	hooks       *providerHooks
	deadline    time.Time
	hasDeadline bool
}

type providerHooks struct {
	mu        sync.Mutex
	cancelled bool
	next      uint64
	callbacks map[uint64]func()
}

func newProviderContext(parent context.Context) (context.Context, context.CancelFunc) {
	signal, cancelSignal := context.WithCancelCause(context.Background())
	hooks := &providerHooks{callbacks: make(map[uint64]func())}
	ref := weak.Make(hooks)
	cancelWithError := func(reason error) {
		cancelSignal(reason)
		if hooks := ref.Value(); hooks != nil {
			hooks.mu.Lock()
			hooks.cancelled = true
			callbacks := hooks.callbacks
			hooks.callbacks = nil
			hooks.mu.Unlock()
			for _, callback := range callbacks {
				go callback()
			}
		}
	}
	cancel := func() { cancelWithError(context.Canceled) }
	stopParent := context.AfterFunc(parent, func() { cancelWithError(parent.Err()) })
	deadline, hasDeadline := parent.Deadline()
	ctx := &providerContext{Context: context.WithoutCancel(parent), signal: signal,
		hooks: hooks, deadline: deadline, hasDeadline: hasDeadline}
	if parent.Err() != nil {
		cancelWithError(parent.Err())
	}
	return ctx, func() { stopParent(); cancel() }
}

func (ctx *providerContext) Done() <-chan struct{}       { return ctx.signal.Done() }
func (ctx *providerContext) Err() error                  { return context.Cause(ctx.signal) }
func (ctx *providerContext) Deadline() (time.Time, bool) { return ctx.deadline, ctx.hasDeadline }

func (ctx *providerContext) AfterFunc(callback func()) func() bool {
	hooks := ctx.hooks
	hooks.mu.Lock()
	defer hooks.mu.Unlock()
	if hooks.cancelled {
		go callback()
		return func() bool { return false }
	}
	hooks.next++
	id := hooks.next
	hooks.callbacks[id] = callback
	return func() bool {
		hooks.mu.Lock()
		defer hooks.mu.Unlock()
		_, pending := hooks.callbacks[id]
		delete(hooks.callbacks, id)
		return pending
	}
}
