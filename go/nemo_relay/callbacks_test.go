// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"sync"
	"testing"
)

func TestRegisterAndUnregisterClosure(t *testing.T) {
	fn := ToolExecutionFunc(func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	})

	userData := registerClosure(fn)
	if userData == nil {
		t.Fatal("registerClosure returned nil")
	}

	if lookupClosure(userData) == nil {
		t.Fatal("lookupClosure returned nil before unregister")
	}

	id := closureID(userData)
	unregisterClosure(userData)

	closureRegistryMu.Lock()
	_, exists := closureRegistry[id]
	closureRegistryMu.Unlock()
	if exists {
		t.Fatal("closure registry still contains callback after unregister")
	}
}

func closureRegistryLen() int {
	closureRegistryMu.Lock()
	defer closureRegistryMu.Unlock()
	return len(closureRegistry)
}

func TestScopeSubscribeInvalidUUIDDoesNotLeakClosure(t *testing.T) {
	before := closureRegistryLen()

	subscription, err := ScopeSubscribe("not-a-uuid", func(event Event) {
		_ = event
	})
	if err == nil {
		t.Fatal("expected ScopeSubscribe to fail for invalid UUID")
	}
	if subscription != nil {
		t.Fatal("expected no subscription for invalid UUID")
	}

	after := closureRegistryLen()
	if after != before {
		t.Fatalf("closure registry length changed after invalid ScopeSubscribe: before=%d after=%d", before, after)
	}
}

func TestScopeSubscribeMissingScopeDoesNotLeakClosure(t *testing.T) {
	before := closureRegistryLen()

	subscription, err := ScopeSubscribe("00000000-0000-0000-0000-000000000000", func(event Event) {
		_ = event
	})
	if err == nil {
		t.Fatal("expected ScopeSubscribe to fail for missing scope")
	}
	if subscription != nil {
		t.Fatal("expected no subscription for missing scope")
	}

	after := closureRegistryLen()
	if after != before {
		t.Fatalf("closure registry length changed after missing-scope ScopeSubscribe: before=%d after=%d", before, after)
	}
}

func TestSubscriptionCloseConcurrentIsIdempotent(t *testing.T) {
	subscription, err := Subscribe(func(event Event) {
		_ = event
	})
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}

	var wg sync.WaitGroup
	results := make(chan struct {
		closed bool
		err    error
	}, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			closed, err := subscription.Close()
			results <- struct {
				closed bool
				err    error
			}{closed: closed, err: err}
		}()
	}
	wg.Wait()
	close(results)

	closedCount := 0
	for result := range results {
		if result.err != nil {
			t.Fatalf("Close failed: %v", result.err)
		}
		if result.closed {
			closedCount++
		}
	}
	if closedCount != 1 {
		t.Fatalf("expected exactly one concurrent Close to close the handle, got %d", closedCount)
	}
	if closed, err := subscription.Close(); err != nil {
		t.Fatalf("Close after concurrent closes failed: %v", err)
	} else if closed {
		t.Fatal("Close after concurrent closes reported a live close")
	}
}
