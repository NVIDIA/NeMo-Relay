// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"os"
	"testing"
)

// runTestInIsolatedWorkingDirectory changes the process-wide working directory,
// so callers must not use it with t.Parallel().
func runTestInIsolatedWorkingDirectory(t *testing.T, fn func(*testing.T)) {
	t.Helper()

	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	original, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd failed: %v", err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatalf("Chdir to temporary directory failed: %v", err)
	}
	defer func() {
		if err := os.Chdir(original); err != nil {
			t.Errorf("restore working directory failed: %v", err)
		}
	}()

	fn(t)
}

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
