// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"testing"
)

func TestInspectLLMContentSelections(t *testing.T) {
	request := json.RawMessage(`{"instructions":"top","messages":[{"role":"user","content":""},{"role":"assistant","content":[{"type":"file","file":{"id":"f"}}]}]}`)
	for _, test := range []struct {
		selection ContentSelection
		index     int64
		root      ContentRoot
		text      string
		hasText   bool
	}{
		{ContentSelectInstructions, 0, "content", "top", true},
		{ContentSelectSystemPrompt, 0, "content", "top", true},
		{ContentSelectLastUserMessage, 0, "request_message", "", true},
		{ContentSelectMessage, -2, "request_message", "", true},
		{ContentSelectLastMessage, 0, "request_message", "", false},
	} {
		view, err := InspectLLMContent(ContentSourceRequest, request, test.selection, test.index)
		if err != nil || view == nil || len(view.Parts) != 1 {
			t.Fatalf("selection %d: view=%+v err=%v", test.selection, view, err)
		}
		if text, ok := view.Text(); text != test.text || ok != test.hasText || view.Parts[0].Location.Root != test.root {
			t.Fatalf("selection %d: text=(%q,%t) location=%+v", test.selection, text, ok, view.Parts[0].Location)
		}
		if test.selection == ContentSelectMessage {
			index := view.Parts[0].Location.MessageIndex
			if index == nil || *index != 0 {
				t.Fatalf("resolved index=%v", index)
			}
		}
	}
	for _, selection := range []ContentSelection{ContentSelectInstructions, ContentSelectLastMessage} {
		if view, err := InspectLLMContent(ContentSourceRequest, json.RawMessage(`{}`), selection, 0); err != nil || view != nil {
			t.Fatalf("absent: view=%+v err=%v", view, err)
		}
	}
}

func TestInspectLLMContentOwnsOrderedPartsAndLosslessPayloads(t *testing.T) {
	response := json.RawMessage(`{"message":[{"type":"text","text":"hi\u0000"},{"type":"refusal","refusal":""},{"type":"image","image":{"id":"img"}}],"tool_calls":[{"id":"c","name":"f","arguments":{"n":18446744073709551615}}]}`)
	view, err := InspectLLMContent(ContentSourceResponse, response, ContentSelectOutput, 0)
	if err != nil {
		t.Fatal(err)
	}
	clear(response)
	if text, ok := view.Text(); !ok || text != "hi\x00\n" {
		t.Fatalf("text=(%q,%v)", text, ok)
	}
	if len(view.Parts) != 4 {
		t.Fatalf("parts=%+v", view.Parts)
	}
	for i, kind := range []ContentKind{"text", "refusal", "image", "tool_call"} {
		if view.Parts[i].Kind != kind {
			t.Fatalf("part %d kind=%s", i, view.Parts[i].Kind)
		}
	}
	call := view.Parts[3].ToolCall
	if call == nil || call.ID == nil || *call.ID != "c" || call.Arguments.Kind != "json" || string(call.Arguments.Value) != `{"n":18446744073709551615}` {
		t.Fatalf("tool call=%+v", call)
	}
	if view.Parts[0].Text != "hi\x00" || string(view.Parts[2].Value) != `{"id":"img"}` || view.Parts[3].Location.Field != "tool_calls" {
		t.Fatalf("snapshot=%+v", view)
	}
}

func TestInspectLLMContentStandalonePayloadRepresentations(t *testing.T) {
	for _, test := range []struct{ message, kind, value string }{
		{`{"role":"tool_result","call_id":"c","output":null}`, "json", "null"},
		{`{"role":"tool_result","call_id":"c","output":"{invalid}"}`, "json", `"{invalid}"`},
		{`{"role":"function","name":"f","content":null}`, "null", ""},
		{`{"role":"function","name":"f","content":"{invalid}"}`, "string", `"{invalid}"`},
		{`{"role":"tool","tool_call_id":"c","content":"hidden"}`, "content", `"hidden"`},
	} {
		view, err := InspectLLMContent(ContentSourceMessage, json.RawMessage(test.message), ContentSelectAll, 0)
		if err != nil {
			t.Fatal(err)
		}
		result := view.Parts[0].ToolResult
		if result == nil || string(result.Payload.Kind) != test.kind || string(result.Payload.Value) != test.value {
			t.Fatalf("result=%+v", result)
		}
		if _, ok := view.Text(); ok {
			t.Fatal("tool result contributed display text")
		}
	}
	view, err := InspectLLMContent(ContentSourceContent, json.RawMessage(`[{"type":"provider_native","provider":"custom","kind":"opaque","value":{"n":9007199254740993}}]`), ContentSelectAll, 0)
	if err != nil {
		t.Fatal(err)
	}
	native := view.Parts[0].ProviderNative
	if native == nil || native.Provider != "custom" || native.Kind != "opaque" || string(native.Value) != `{"n":9007199254740993}` {
		t.Fatalf("native=%+v", native)
	}
}

func TestInspectLLMContentErrors(t *testing.T) {
	for _, test := range []struct {
		source    ContentSource
		input     string
		selection ContentSelection
		index     int64
	}{
		{ContentSourceRequest, "{", ContentSelectLastMessage, 0},
		{ContentSourceRequest, "{}\x00trailing", ContentSelectLastMessage, 0},
		{ContentSourceRequest, "{}", ContentSelectMessage, -1},
		{ContentSourceContent, `""`, ContentSelectLastMessage, 0},
	} {
		view, err := InspectLLMContent(test.source, json.RawMessage(test.input), test.selection, test.index)
		if view != nil || err == nil {
			t.Fatalf("%+v: view=%v err=%v", test, view, err)
		}
	}
}
