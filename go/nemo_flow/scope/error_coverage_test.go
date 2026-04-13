// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package scope_test

import (
	"encoding/json"
	"testing"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow/scope"
)

func TestWithScopeCleanupNoopsWhenPushFails(t *testing.T) {
	before, err := nemo_flow.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle before failed: %v", err)
	}

	cleanup := scope.WithScope("invalid_scope", nemo_flow.ScopeTypeAgent, nemo_flow.WithData(json.RawMessage("{")))
	cleanup()

	after, err := nemo_flow.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after WithScope failure failed: %v", err)
	}
	if after.UUID() != before.UUID() {
		t.Fatalf("expected top of stack to remain %q, got %q", before.UUID(), after.UUID())
	}

	handle, cleanupHandle := scope.WithScopeHandle("invalid_scope", nemo_flow.ScopeTypeAgent, nemo_flow.WithData(json.RawMessage("{")))
	if handle != nil {
		t.Fatalf("expected nil handle on failed push, got %#v", handle)
	}
	cleanupHandle()

	afterHandle, err := nemo_flow.GetHandle()
	if err != nil {
		t.Fatalf("GetHandle after WithScopeHandle failure failed: %v", err)
	}
	if afterHandle.UUID() != before.UUID() {
		t.Fatalf("expected top of stack to remain %q, got %q", before.UUID(), afterHandle.UUID())
	}
}
