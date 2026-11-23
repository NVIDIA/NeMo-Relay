// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdint.h>
#include <stdlib.h>
extern int32_t nemo_relay_inspect_llm_content_json(uint32_t source, const char *input, uint32_t selection, int64_t index, char **out);
extern void nemo_relay_string_free(char *value);
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"unsafe"
)

// ContentSource identifies the normalized JSON supplied to InspectLLMContent.
type ContentSource uint32

// Source discriminators correspond to the NEMO_RELAY_CONTENT_SOURCE_* C constants.
const (
	ContentSourceRequest  ContentSource = 1
	ContentSourceResponse ContentSource = 2
	ContentSourceMessage  ContentSource = 3
	ContentSourceContent  ContentSource = 4
)

// ContentSelection selects one view before extracting any text.
type ContentSelection uint32

// Selection discriminators correspond to the NEMO_RELAY_CONTENT_SELECT_* C constants.
const (
	ContentSelectLastMessage     ContentSelection = 1
	ContentSelectMessage         ContentSelection = 2
	ContentSelectInstructions    ContentSelection = 3
	ContentSelectSystemPrompt    ContentSelection = 4
	ContentSelectLastUserMessage ContentSelection = 5
	ContentSelectOutput          ContentSelection = 6
	ContentSelectAll             ContentSelection = 7
)

// ContentSnapshot holds Go-owned entries and eligible text in normalized order.
type ContentSnapshot struct {
	Parts     []ContentEntry `json:"parts"`
	TextParts []string       `json:"text_parts"`
}

// Text joins eligible parts with newlines. An explicit empty string returns ("", true).
func (s *ContentSnapshot) Text() (string, bool) {
	if s == nil || len(s.TextParts) == 0 {
		return "", false
	}
	return strings.Join(s.TextParts, "\n"), true
}

// ContentRoot identifies the normalized object containing an entry.
type ContentRoot string

const (
	ContentRootRequestMessage ContentRoot = "request_message"
	ContentRootMessage        ContentRoot = "message"
	ContentRootResponse       ContentRoot = "response"
	ContentRootContent        ContentRoot = "content"
)

// ContentField identifies a field within a normalized root.
type ContentField string

const (
	ContentFieldContent         ContentField = "content"
	ContentFieldMessage         ContentField = "message"
	ContentFieldToolCalls       ContentField = "tool_calls"
	ContentFieldToolCallItem    ContentField = "tool_call_item"
	ContentFieldOutput          ContentField = "output"
	ContentFieldValue           ContentField = "value"
	ContentFieldNativeContent   ContentField = "native_content"
	ContentFieldNativeRefusal   ContentField = "native_refusal"
	ContentFieldNativeToolCalls ContentField = "native_tool_calls"
)

// ContentKind identifies the populated value in a ContentEntry.
type ContentKind string

const (
	ContentKindText           ContentKind = "text"
	ContentKindRefusal        ContentKind = "refusal"
	ContentKindToolCall       ContentKind = "tool_call"
	ContentKindToolResult     ContentKind = "tool_result"
	ContentKindImageURL       ContentKind = "image_url"
	ContentKindImage          ContentKind = "image"
	ContentKindAudio          ContentKind = "audio"
	ContentKindFile           ContentKind = "file"
	ContentKindProviderNative ContentKind = "provider_native"
)

// ContentPayloadKind identifies the normalized representation of a tool payload.
type ContentPayloadKind string

const (
	ContentPayloadKindNull    ContentPayloadKind = "null"
	ContentPayloadKindString  ContentPayloadKind = "string"
	ContentPayloadKindJSON    ContentPayloadKind = "json"
	ContentPayloadKindContent ContentPayloadKind = "content"
)

// ContentLocation identifies the normalized source at the time of selection.
type ContentLocation struct {
	Root         ContentRoot  `json:"root"`
	MessageIndex *uint64      `json:"message_index"`
	Field        ContentField `json:"field"`
	Index        *uint64      `json:"index"`
}

// ContentPayload distinguishes "null", "string", "json", and "content" representations.
// Value holds the normalized payload as JSON, preserving integer precision. The "null" kind has no value.
type ContentPayload struct {
	Kind  ContentPayloadKind `json:"kind"`
	Value json.RawMessage    `json:"value"`
}

// ContentToolCall retains the item and correlation IDs and the original argument representation.
type ContentToolCall struct {
	ID        *string        `json:"id"`
	CallID    *string        `json:"call_id"`
	Name      string         `json:"name"`
	Arguments ContentPayload `json:"arguments"`
}

// ContentToolResult contains the entire tool result, including optional error status.
type ContentToolResult struct {
	ID      *string        `json:"id"`
	CallID  *string        `json:"call_id"`
	Name    *string        `json:"name"`
	IsError *bool          `json:"is_error"`
	Payload ContentPayload `json:"payload"`
}

// ContentProviderNative retains an opaque provider value and its provider-specific discriminator.
type ContentProviderNative struct {
	Provider string          `json:"provider"`
	Kind     string          `json:"kind"`
	Value    json.RawMessage `json:"value"`
}

// ContentEntry is one normalized part. Kind selects Text ("text" or "refusal"),
// ToolCall, ToolResult, ProviderNative, or Value (media and other opaque kinds).
type ContentEntry struct {
	Kind           ContentKind
	Location       ContentLocation
	Text           string
	ToolCall       *ContentToolCall
	ToolResult     *ContentToolResult
	ProviderNative *ContentProviderNative
	Value          json.RawMessage
}

// UnmarshalJSON decodes a tagged entry in the native snapshot format.
func (entry *ContentEntry) UnmarshalJSON(data []byte) error {
	var wire struct {
		Location ContentLocation `json:"location"`
		Kind     ContentKind     `json:"kind"`
		Value    json.RawMessage `json:"value"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	result := ContentEntry{Kind: wire.Kind, Location: wire.Location}
	var target any
	switch wire.Kind {
	case ContentKindText, ContentKindRefusal:
		target = &result.Text
	case ContentKindToolCall:
		target = &result.ToolCall
	case ContentKindToolResult:
		target = &result.ToolResult
	case ContentKindProviderNative:
		target = &result.ProviderNative
	default:
		result.Value = wire.Value
	}
	if target != nil {
		if err := json.Unmarshal(wire.Value, target); err != nil {
			return err
		}
	}
	*entry = result
	return nil
}

// InspectLLMContent selects from normalized JSON, including annotations returned by
// sanitizer codecs. Only ContentSelectMessage uses index (negative counts from the end).
// An absent selection returns (nil, nil). A selected item with no eligible text
// returns a snapshot. All results are independent of the input and callback lifetime.
func InspectLLMContent(source ContentSource, input json.RawMessage, selection ContentSelection, index int64) (*ContentSnapshot, error) {
	if bytes.IndexByte(input, 0) >= 0 {
		return nil, errors.New("invalid normalized JSON: unescaped NUL byte")
	}
	cInput := C.CString(string(input))
	defer C.free(unsafe.Pointer(cInput))
	var out *C.char
	err := callStatus(func() C.int32_t {
		return C.nemo_relay_inspect_llm_content_json(C.uint32_t(source), cInput, C.uint32_t(selection), C.int64_t(index), &out)
	})
	if err != nil || out == nil {
		return nil, err
	}
	defer C.nemo_relay_string_free(out)
	var snapshot ContentSnapshot
	if err := json.Unmarshal([]byte(C.GoString(out)), &snapshot); err != nil {
		return nil, err
	}
	return &snapshot, nil
}
