// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "testing"

func runWithTestScopeStack(t *testing.T, fn func()) {
	t.Helper()

	stack, err := NewScopeStack()
	if err != nil {
		t.Fatalf("NewScopeStack failed: %v", err)
	}
	defer stack.Close()

	stack.Run(fn)
}

func runTestWithScopeStack(t *testing.T, fn func(*testing.T)) {
	t.Helper()
	runWithTestScopeStack(t, func() { fn(t) })
}
