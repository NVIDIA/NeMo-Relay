// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package adaptive

import "testing"

func TestAdaptivePackageRuntimeHelpers(t *testing.T) {
	report, err := ValidateConfig(NewConfig())
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected clean report, got %#v", report.Diagnostics)
	}

	runtime, err := NewRuntime(NewConfig())
	if err != nil {
		t.Fatalf("NewRuntime failed: %v", err)
	}
	if err := runtime.Register(); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	if err := runtime.Shutdown(); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}
}
