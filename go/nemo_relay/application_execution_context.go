// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ApplicationExecutionContext is invocation-local application context. HostApproval
// has a checked public shape; AdditionalFields preserves application-owned JSON.
// A nil pointer means absent context, while an empty value marshals to {}.
// These are application assertions, not proof: middleware selects trusted sources.
type ApplicationExecutionContext struct {
	HostApproval     *HostApproval
	AdditionalFields map[string]json.RawMessage
}

// HostApproval describes an application-asserted approval of specific arguments.
// Version must be 1, Decision must be "approved", and identifying strings must
// be nonempty. ApprovedArguments accepts any JSON value, including explicit null.
type HostApproval struct {
	Version           int
	Source            string
	Decision          string
	ApprovalID        string
	ToolName          string
	ApprovedArguments json.RawMessage
	AdditionalFields  map[string]json.RawMessage
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

// MarshalJSON preserves additional fields and serializes the checked known field.
func (c ApplicationExecutionContext) MarshalJSON() ([]byte, error) {
	fields := cloneContextFields(c.AdditionalFields)
	if _, exists := fields["host_approval"]; exists {
		return nil, fmt.Errorf("host_approval must use the typed field")
	}
	if c.HostApproval != nil {
		value, err := json.Marshal(c.HostApproval)
		if err != nil {
			return nil, err
		}
		fields["host_approval"] = value
	}
	return json.Marshal(fields)
}

// UnmarshalJSON validates known fields while retaining unknown JSON unchanged.
func (c *ApplicationExecutionContext) UnmarshalJSON(data []byte) error {
	fields, err := contextObject(data)
	if err != nil {
		return err
	}
	var approval *HostApproval
	if value, exists := fields["host_approval"]; exists {
		var known HostApproval
		if err := json.Unmarshal(value, &known); err != nil {
			return err
		}
		approval = &known
		delete(fields, "host_approval")
	}
	*c = ApplicationExecutionContext{HostApproval: approval, AdditionalFields: fields}
	return nil
}

func cloneContextFields(fields map[string]json.RawMessage) map[string]json.RawMessage {
	result := make(map[string]json.RawMessage, len(fields))
	for key, value := range fields {
		result[key] = append(json.RawMessage(nil), value...)
	}
	return result
}

func (a HostApproval) validate() error {
	if a.Version != 1 || a.Decision != "approved" || a.Source == "" || a.ApprovalID == "" || a.ToolName == "" || !json.Valid(a.ApprovedArguments) {
		return fmt.Errorf("host_approval requires version 1, decision approved, nonempty source, approval_id and tool_name, and JSON approved_arguments")
	}
	return nil
}

// MarshalJSON validates the standard approval fields and preserves extensions.
func (a HostApproval) MarshalJSON() ([]byte, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}
	fields := cloneContextFields(a.AdditionalFields)
	known := map[string]any{"version": a.Version, "source": a.Source, "decision": a.Decision, "approval_id": a.ApprovalID, "tool_name": a.ToolName, "approved_arguments": a.ApprovedArguments}
	for key, value := range known {
		if _, exists := fields[key]; exists {
			return nil, fmt.Errorf("%s must use the typed field", key)
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		fields[key] = encoded
	}
	return json.Marshal(fields)
}

// UnmarshalJSON rejects malformed or incomplete approval claims.
func (a *HostApproval) UnmarshalJSON(data []byte) error {
	fields, err := contextObject(data)
	if err != nil {
		return err
	}
	var result HostApproval
	known := map[string]any{"version": &result.Version, "source": &result.Source, "decision": &result.Decision, "approval_id": &result.ApprovalID, "tool_name": &result.ToolName, "approved_arguments": &result.ApprovedArguments}
	for key, target := range known {
		value, exists := fields[key]
		if !exists {
			return fmt.Errorf("host_approval requires %s", key)
		}
		if key == "version" {
			decoder := json.NewDecoder(bytes.NewReader(value))
			decoder.UseNumber()
			var number any
			if err := decoder.Decode(&number); err != nil {
				return err
			}
			numeric, ok := number.(json.Number)
			if !ok {
				return fmt.Errorf("host_approval version must be numeric 1")
			}
			version, err := numeric.Float64()
			if err != nil || version != 1 {
				return fmt.Errorf("host_approval version must be 1")
			}
			result.Version = 1
		} else if err := json.Unmarshal(value, target); err != nil {
			return err
		}
		delete(fields, key)
	}
	if err := result.validate(); err != nil {
		return err
	}
	result.AdditionalFields = fields
	*a = result
	return nil
}
