// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "testing"

func TestAllBuiltinKindsAreAvailableWithoutComponentHelpers(t *testing.T) {
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatal(err)
	}
	available := make(map[string]bool)
	for _, kind := range kinds {
		available[kind] = true
	}
	for _, kind := range []string{"observability", "pricing", "resource_metrics", "adaptive", "pii_redaction"} {
		if !available[kind] {
			t.Errorf("missing built-in plugin %s", kind)
		}
	}
}
