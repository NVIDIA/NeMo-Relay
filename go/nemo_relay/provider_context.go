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
	signal      *providerSignal
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

// The separate Done channel prevents context from attaching child contexts to
// the cause carrier. Children use AfterFunc instead, keeping GC cleanup free
// of callbacks that can retain the managed stream.
type providerSignal struct {
	context.Context
	done chan struct{}
	once sync.Once
	err  error
}

func (signal *providerSignal) Err() error {
	select {
	case <-signal.done:
		return signal.err
	default:
		return nil
	}
}

func newProviderContext(parent context.Context) (context.Context, context.CancelFunc) {
	carrier, cancelCause := context.WithCancelCause(context.Background())
	signal := &providerSignal{Context: carrier, done: make(chan struct{})}
	hooks := &providerHooks{callbacks: make(map[uint64]func())}
	ref := weak.Make(hooks)
	cancelWithError := func(err, cause error) {
		signal.once.Do(func() {
			signal.err = err
			cancelCause(cause)
			close(signal.done)
		})
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
	cancel := func() { cancelWithError(context.Canceled, context.Canceled) }
	stopParent := context.AfterFunc(parent, func() { cancelWithError(parent.Err(), context.Cause(parent)) })
	deadline, hasDeadline := parent.Deadline()
	ctx := &providerContext{Context: context.WithoutCancel(parent), signal: signal,
		hooks: hooks, deadline: deadline, hasDeadline: hasDeadline}
	if parent.Err() != nil {
		cancelWithError(parent.Err(), context.Cause(parent))
	}
	return ctx, func() { stopParent(); cancel() }
}

func (ctx *providerContext) Done() <-chan struct{}       { return ctx.signal.done }
func (ctx *providerContext) Err() error                  { return ctx.signal.Err() }
func (ctx *providerContext) Deadline() (time.Time, bool) { return ctx.deadline, ctx.hasDeadline }
func (ctx *providerContext) Value(key any) any {
	if value := ctx.signal.Value(key); value != nil {
		return value
	}
	return ctx.Context.Value(key)
}

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
