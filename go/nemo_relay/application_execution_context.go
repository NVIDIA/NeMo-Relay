// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
)

// ApplicationExecutionContext is a concrete invocation-local Relay context.
// Application-specific fields belong in Data; middleware validates and interprets
// them. A nil context pointer omits context. A nil Data map marshals to {}, while
// an allocated empty map preserves {"data":{}}. Context does not grant authority.
type ApplicationExecutionContext struct {
	Data map[string]json.RawMessage
}

func contextObject(data []byte) (map[string]json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, fmt.Errorf("execution context must be a JSON object")
	}
	return fields, nil
}

// MarshalJSON preserves absent versus explicitly empty application data.
func (c ApplicationExecutionContext) MarshalJSON() ([]byte, error) {
	fields := make(map[string]any)
	if c.Data != nil {
		fields["data"] = cloneContextFields(c.Data)
	}
	return json.Marshal(fields)
}

// UnmarshalJSON accepts only Relay-defined outer fields and opaque object data.
func (c *ApplicationExecutionContext) UnmarshalJSON(data []byte) error {
	fields, err := contextObject(data)
	if err != nil {
		return err
	}
	for key := range fields {
		if key != "data" {
			return fmt.Errorf("invalid application execution context")
		}
	}
	var payload map[string]json.RawMessage
	if value, exists := fields["data"]; exists {
		payload, err = contextObject(value)
		if err != nil {
			return fmt.Errorf("execution context data must be a JSON object")
		}
	}
	*c = ApplicationExecutionContext{Data: payload}
	return nil
}

func cloneContextFields(fields map[string]json.RawMessage) map[string]json.RawMessage {
	result := make(map[string]json.RawMessage, len(fields))
	for key, value := range fields {
		result[key] = append(json.RawMessage(nil), value...)
	}
	return result
}
