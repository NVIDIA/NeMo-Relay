// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
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
