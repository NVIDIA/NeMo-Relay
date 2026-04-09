// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package optimizer

import "testing"

func TestConfigBuilders(t *testing.T) {
	config := NewConfig()
	if config.Version != 1 {
		t.Fatalf("expected version 1, got %d", config.Version)
	}

	config.State = &StateConfig{Backend: NewInMemoryBackend()}
	config.Components = []ComponentSpec{
		TelemetryComponent(TelemetryComponentConfig{
			Learners: []string{"latency_sensitivity"},
		}),
		DynamoHintsComponent(NewDynamoHintsComponentConfig()),
		ToolParallelismComponent(NewToolParallelismComponentConfig()),
	}

	report, err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected no diagnostics, got %+v", report.Diagnostics)
	}
}
