// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"
)

func TestRegisterConditionalMiddlewareGuardrailRejectsNilCallback(t *testing.T) {
	err := RegisterConditionalMiddlewareGuardrail(
		"go-runtime-nil-gate",
		[]RuntimeRegistrationKind{RuntimeRegistrationSubscriber},
		"target-subscriber",
		nil,
	)
	if !errors.Is(err, errConditionalMiddlewareGuardrailCallbackNil) {
		t.Fatalf("RegisterConditionalMiddlewareGuardrail() error = %v, want %v", err, errConditionalMiddlewareGuardrailCallbackNil)
	}
}

func TestConditionalMiddlewareGuardrailTogglesExistingRegistration(t *testing.T) {
	suffix := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	targetName := "go-runtime-target-" + suffix
	gateName := "go-runtime-gate-" + suffix

	if err := RegisterToolRequestIntercept(targetName, 0, false, func(_ string, args json.RawMessage) json.RawMessage {
		return json.RawMessage(`{"intercepted":true}`)
	}); err != nil {
		t.Fatalf("register target: %v", err)
	}
	defer DeregisterToolRequestIntercept(targetName)

	registrations, err := ListRuntimeRegistrations([]RuntimeRegistrationKind{RuntimeRegistrationToolRequestIntercept})
	if err != nil {
		t.Fatalf("list registrations: %v", err)
	}
	var target *RuntimeRegistrationIdentity
	for index := range registrations {
		if registrations[index].LocalName == targetName {
			target = &registrations[index]
			break
		}
	}
	if target == nil {
		t.Fatalf("target registration %q was not discovered", targetName)
	}

	reason := "timer active"
	callbackMatched := false
	if err := RegisterConditionalMiddlewareGuardrail(
		gateName,
		[]RuntimeRegistrationKind{RuntimeRegistrationToolRequestIntercept},
		target.EffectiveName,
		func(kinds []RuntimeRegistrationKind, registrationName string) *string {
			callbackMatched = len(kinds) == 1 &&
				kinds[0] == RuntimeRegistrationToolRequestIntercept &&
				registrationName == targetName
			return &reason
		},
	); err != nil {
		t.Fatalf("register gate: %v", err)
	}
	defer DeregisterConditionalMiddlewareGuardrail(gateName)

	disabled, err := ToolRequestIntercepts("tool", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("resolve disabled target: %v", err)
	}
	if string(disabled) != "{}" {
		t.Fatalf("disabled target changed arguments: %s", disabled)
	}
	if !callbackMatched {
		t.Fatal("gate callback did not receive the configured selector and target")
	}

	removed, err := DeregisterConditionalMiddlewareGuardrail(gateName)
	if err != nil || !removed {
		t.Fatalf("deregister gate: removed=%v err=%v", removed, err)
	}

	enabled, err := ToolRequestIntercepts("tool", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("resolve enabled target: %v", err)
	}
	if string(enabled) != `{"intercepted":true}` {
		t.Fatalf("enabled target result: %s", enabled)
	}
}
