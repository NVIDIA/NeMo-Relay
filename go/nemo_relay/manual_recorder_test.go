// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestManualLLMOptimizationRecorderLifecycle(t *testing.T) {
	runWithTestScopeStack(t, func() {
		var mu sync.Mutex
		var callID string
		var events []Event
		if err := RegisterSubscriber("manual-recorder-events", func(event Event) {
			mu.Lock()
			defer mu.Unlock()
			if event.ParentUUID() == callID && event.Name() == "nemo_relay.llm.optimization" || event.Name() == "manual-recorder" && event.ScopeCategory() == "end" {
				events = append(events, event)
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer DeregisterSubscriber("manual-recorder-events")
		handle, err := LlmCall("manual-recorder", requestForTest(makeRequest()))
		if err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		callID = handle.UUID()
		mu.Unlock()
		first, err := handle.RetainOptimizationRecorder()
		if err != nil {
			t.Fatal(err)
		}
		defer first.Close()
		second, err := handle.RetainOptimizationRecorder()
		if err != nil {
			t.Fatal(err)
		}
		defer second.Close()
		contribution := LLMOptimizationContribution{Producer: "manual-evidence", Kind: "custom", Applied: true}
		if accepted, err := first.RecordOptimizationContribution(contribution); err != nil || !accepted {
			t.Fatalf("first record: %v %v", accepted, err)
		}
		if err := first.Close(); err != nil {
			t.Fatal(err)
		}
		if accepted, err := first.RecordOptimizationContribution(contribution); err == nil || accepted {
			t.Fatalf("closed owner must fail: %v %v", accepted, err)
		}
		if accepted, err := second.RecordOptimizationContribution(contribution); err != nil || !accepted {
			t.Fatalf("independent owner: %v %v", accepted, err)
		}
		if err := LlmCallEnd(handle, json.RawMessage(`{"response":"ok"}`)); err != nil {
			t.Fatal(err)
		}
		afterEnd, err := handle.RetainOptimizationRecorder()
		if err != nil {
			t.Fatal(err)
		}
		defer afterEnd.Close()
		if accepted, err := afterEnd.RecordOptimizationContribution(contribution); err != nil || accepted {
			t.Fatalf("retaining must not reopen accounting: %v %v", accepted, err)
		}
		handle = nil
		runtime.GC()
		if accepted, err := second.RecordOptimizationContribution(contribution); err != nil || accepted {
			t.Fatalf("late record: %v %v", accepted, err)
		}
		if err := FlushSubscribers(); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(events) != 3 {
			t.Fatalf("expected two marks followed by end, got %d events", len(events))
		}
		for i := 0; i < 2; i++ {
			if events[i].Name() != "nemo_relay.llm.optimization" || events[i].ParentUUID() != callID {
				t.Fatalf("unexpected mark: %s", events[i].JSON())
			}
			var emitted LLMOptimizationContribution
			if err := json.Unmarshal(events[i].Data(), &emitted); err != nil {
				t.Fatal(err)
			}
			if emitted.Producer != contribution.Producer || emitted.Sequence == nil || *emitted.Sequence != uint64(i) {
				t.Fatalf("unexpected contribution: %+v", emitted)
			}
		}
		if events[2].ScopeCategory() != "end" || !strings.Contains(string(events[2].AnnotatedResponse()), "manual-evidence") {
			t.Fatalf("missing finalized summary: %s", events[2].JSON())
		}
	})
}

func TestManualLLMOptimizationRecorderRejectsNilHandle(t *testing.T) {
	for _, handle := range []*LLMHandle{nil, {}} {
		recorder, err := handle.RetainOptimizationRecorder()
		if recorder != nil || !errors.Is(err, ErrNullPointer) {
			t.Fatalf("nil handle: recorder=%v error=%v", recorder, err)
		}
	}
}
