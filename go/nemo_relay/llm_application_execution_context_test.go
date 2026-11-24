// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestLLMContextJSONCanonicalRoundTrip(t *testing.T) {
	for _, test := range []struct{ input, expected string }{
		{`null`, `null`}, {`true`, `true`}, {`false`, `false`},
		{`0.0`, `0.0`}, {`-0`, `-0.0`}, {`-1.25`, `-1.25`},
		{`1e-5`, `0.00001`}, {`-1e-5`, `-0.00001`}, {`1e-6`, `1e-6`},
		{`1e15`, `1000000000000000.0`}, {`1e16`, `1e+16`},
		{`18446744073709551616`, `1.8446744073709552e+19`},
		{`{"list":[false,true,null,{},[],1.25]}`, `{"list":[false,true,null,{},[],1.25]}`},
		{`"a\/b\n\t\"\\\b\f\r\u0000\u20ac\ud83d\ude00"`, "\"a/b\\n\\t\\\"\\\\\\b\\f\\r\\u0000€😀\""},
	} {
		t.Run(test.input, func(t *testing.T) {
			encoded, err := json.Marshal(LLMApplicationExecutionContext{Data: map[string]json.RawMessage{"value": json.RawMessage(test.input)}})
			if err != nil {
				t.Fatal(err)
			}
			if string(encoded) != `{"data":{"value":`+test.expected+`}}` {
				t.Fatalf("context = %s, want value %s", encoded, test.expected)
			}
		})
	}
}

func TestLLMContextJSONBudgetIncludesTrailingContainersAndMembers(t *testing.T) {
	for _, tail := range []string{`""`, `[]`, `{}`, `{"":0}`, `{"key":""}`} {
		t.Run(tail, func(t *testing.T) {
			prefix := `{"data":{"value":["`
			suffix := `",` + tail + `]}}`
			padding := 16384 - len(prefix) - len(suffix)
			for extra := 0; extra <= 2; extra++ {
				raw := json.RawMessage(`["` + strings.Repeat("x", padding+extra) + `",` + tail + `]`)
				encoded, err := json.Marshal(LLMApplicationExecutionContext{Data: map[string]json.RawMessage{"value": raw}})
				if extra == 0 {
					if err != nil || len(encoded) != 16384 {
						t.Fatalf("exact context boundary: %d bytes, %v", len(encoded), err)
					}
				} else if err == nil {
					t.Fatalf("context with %d extra bytes accepted", extra)
				}
			}
		})
	}
}

func TestLLMContextJSONRejectionBeforeExecutionAndRecovery(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprint(streaming), func(t *testing.T) {
			var middlewareCalls, providerCalls atomic.Int32
			callback := LLMExecutionInterceptFunc(func(request json.RawMessage, context LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				middlewareCalls.Add(1)
				return next(request)
			})
			if streaming {
				if err := RegisterLlmStreamExecutionIntercept("context-rejection", 1, callback.AsStream()); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmStreamExecutionIntercept("context-rejection")
			} else {
				if err := RegisterLlmExecutionIntercept("context-rejection", 1, callback); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmExecutionIntercept("context-rejection")
			}
			provider := LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) {
				providerCalls.Add(1)
				return json.RawMessage(`{"ok":true}`), nil
			})
			invoke := func(value LLMApplicationExecutionContext) error {
				option := WithLLMApplicationExecutionContext(&value)
				if streaming {
					stream, err := LlmStreamCallExecute("context-validation", requestForTest(makeRequest()), provider.AsStream(), nil, nil, option)
					if err != nil {
						return err
					}
					chunk, err := stream.Next()
					closeErr := stream.Close()
					if err == nil {
						err = closeErr
					}
					if err == nil && string(chunk) != `{"ok":true}` {
						return fmt.Errorf("unexpected chunk %s", chunk)
					}
					return err
				}
				result, err := LlmCallExecute("context-validation", requestForTest(makeRequest()), provider, option)
				if err == nil && string(result) != `{"ok":true}` {
					return fmt.Errorf("unexpected result %s", result)
				}
				return err
			}
			cases := map[string]map[string]json.RawMessage{
				"invalid raw UTF-8":           {"value": {'"', 0xff, '"'}},
				"number overflow":             {"value": json.RawMessage(`1e400`)},
				"lone low surrogate":          {"value": json.RawMessage(`"\udfff"`)},
				"high surrogate without pair": {"value": json.RawMessage(`"\ud800x"`)},
				"high surrogate wrong pair":   {"value": json.RawMessage(`"\ud800\u0061"`)},
				"oversized key":               {strings.Repeat("x", 16384): json.RawMessage(`0`)},
				"escaped key budget":          {strings.Repeat("\x00", 3000): json.RawMessage(`0`)},
				"escaped value budget":        {"value": json.RawMessage(`"` + strings.Repeat(`\u0000`, 3000) + `"`)},
				"array number expansion":      {"value": json.RawMessage(`[` + strings.Repeat(`1e2,`, 4000) + `1e2]`)},
				"object number expansion":     {"value": json.RawMessage(`{"a":[` + strings.Repeat(`1e2,`, 4000) + `1e2]}`)},
				"nested key budget":           {"value": json.RawMessage(`{"` + strings.Repeat(`\u20ac`, 5500) + `":null}`)},
				"container too deep":          {"value": json.RawMessage(strings.Repeat("[", 63) + `null` + strings.Repeat("]", 63))},
				"compact spelling cap":        {"value": json.RawMessage(`1e` + strings.Repeat("0", 98304))},
				"empty input":                 {"value": json.RawMessage{}},
			}
			wide := map[string]json.RawMessage{}
			for i := range 16384 {
				wide[fmt.Sprint(i)] = nil
			}
			cases["too many fields"] = wide
			for name, data := range cases {
				t.Run(name, func(t *testing.T) {
					middlewareCalls.Store(0)
					providerCalls.Store(0)
					if err := invoke(LLMApplicationExecutionContext{Data: data}); err == nil {
						t.Fatal("invalid context accepted")
					}
					if middlewareCalls.Load() != 0 || providerCalls.Load() != 0 {
						t.Fatal("invalid context reached execution")
					}
					if err := invoke(LLMApplicationExecutionContext{Data: map[string]json.RawMessage{"recovery": json.RawMessage(`true`)}}); err != nil {
						t.Fatal(err)
					}
					if middlewareCalls.Load() != 1 || providerCalls.Load() != 1 {
						t.Fatal("valid recovery did not execute")
					}
				})
			}
		})
	}
}

func TestLLMContextOptionReplacementOmissionAndManualAPIs(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprint(streaming), func(t *testing.T) {
			seen := make(chan *LLMApplicationExecutionContext, 1)
			callback := LLMExecutionInterceptFunc(func(request json.RawMessage, context LLMExecutionContext, next func(json.RawMessage) (json.RawMessage, error)) (json.RawMessage, error) {
				select {
				case seen <- context.ExecutionContext:
				default:
					return nil, fmt.Errorf("duplicate context callback")
				}
				return next(request)
			})
			if streaming {
				if err := RegisterLlmStreamExecutionIntercept("context-replacement", 1, callback.AsStream()); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmStreamExecutionIntercept("context-replacement")
			} else {
				if err := RegisterLlmExecutionIntercept("context-replacement", 1, callback); err != nil {
					t.Fatal(err)
				}
				defer DeregisterLlmExecutionIntercept("context-replacement")
			}
			provider := LLMExecutionFunc(func(json.RawMessage) (json.RawMessage, error) { return json.RawMessage(`{}`), nil })
			for _, value := range []*LLMApplicationExecutionContext{nil, {}, {Data: map[string]json.RawMessage{}}, {Data: map[string]json.RawMessage{"number": json.RawMessage(`18446744073709551615`)}}} {
				options := []LLMCallOption{WithLLMApplicationExecutionContext(&LLMApplicationExecutionContext{}), WithLLMApplicationExecutionContext(value)}
				if streaming {
					stream, err := LlmStreamCallExecute("context-replacement", requestForTest(makeRequest()), provider.AsStream(), nil, nil, options...)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := stream.Next(); err != nil {
						t.Fatal(err)
					}
					if err := stream.Close(); err != nil {
						t.Fatal(err)
					}
				} else if _, err := LlmCallExecute("context-replacement", requestForTest(makeRequest()), provider, options...); err != nil {
					t.Fatal(err)
				}
				var actual *LLMApplicationExecutionContext
				select {
				case actual = <-seen:
				case <-time.After(5 * time.Second):
					t.Fatal("execution intercept did not run")
				}
				if (actual == nil) != (value == nil) {
					t.Fatal("omission changed")
				}
				if value != nil {
					got, _ := json.Marshal(actual)
					want, _ := json.Marshal(value)
					if string(got) != string(want) {
						t.Fatalf("context = %s, want %s", got, want)
					}
				}
			}
		})
	}
	option := WithLLMApplicationExecutionContext(&LLMApplicationExecutionContext{})
	if _, err := LlmCall("context-manual", requestForTest(makeRequest()), option); err == nil || !strings.Contains(err.Error(), "managed") {
		t.Fatalf("manual start accepted context: %v", err)
	}
	handle, err := LlmCall("context-manual", requestForTest(makeRequest()))
	if err != nil {
		t.Fatal(err)
	}
	defer LlmCallEnd(handle, json.RawMessage(`{}`))
	if err := LlmCallEnd(handle, json.RawMessage(`{}`), option); err == nil || !strings.Contains(err.Error(), "managed") {
		t.Fatalf("manual end accepted context: %v", err)
	}
}
