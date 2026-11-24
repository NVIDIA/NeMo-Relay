// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// LLMApplicationExecutionContext carries application data for one LLM invocation.
// A nil Data map omits data; an allocated empty map preserves {"data":{}}.
type LLMApplicationExecutionContext struct {
	Data map[string]json.RawMessage `json:"data,omitempty"`
}

// MarshalJSON validates the context and preserves absent versus empty data.
func (value LLMApplicationExecutionContext) MarshalJSON() ([]byte, error) {
	return marshalLLMApplicationContext(value)
}

const maximumLLMApplicationContextBytes = 16_384

func llmContextBudgetCharge(remaining *int, amount int) error {
	if amount > *remaining {
		return fmt.Errorf("LLM execution context exceeds 16384 serialized bytes")
	}
	*remaining -= amount
	return nil
}

// Reject large strings and containers before encoding/json materializes tokens.
// Escapes contribute a decoded lower bound; whitespace contributes no JSON bytes.
func checkLLMContextInputBudget(input []byte, parentDepth int, remaining *int) error {
	depth, quoted := parentDepth, false
	for index := 0; index < len(input); index++ {
		character, amount := input[index], 1
		if quoted {
			switch character {
			case '"':
				quoted, amount = false, 0
			case '\\':
				if index+1 < len(input) {
					index++
					switch input[index] {
					case 'u':
						index += 4
						amount = 1
					case '/':
						amount = 1
					default:
						amount = 2
					}
				}
			}
		} else {
			switch character {
			case ' ', '\n', '\r', '\t':
				amount = 0
			case '"':
				quoted, amount = true, 2
			case '{', '[':
				depth++
				if depth > 64 {
					return fmt.Errorf("LLM execution context exceeds 64 container levels")
				}
			case '}', ']':
				depth--
			default:
				if character == '-' || character >= '0' && character <= '9' {
					for index+1 < len(input) && strings.ContainsRune("0123456789.eE+-", rune(input[index+1])) {
						index++
					}
				}
			}
		}
		if failure := llmContextBudgetCharge(remaining, amount); failure != nil {
			return failure
		}
	}
	return nil
}

func chargeLLMContextString(value string, remaining *int) error {
	if !utf8.ValidString(value) {
		return fmt.Errorf("LLM execution context contains invalid UTF-8")
	}
	if len(value) > *remaining {
		return fmt.Errorf("LLM execution context exceeds 16384 serialized bytes")
	}
	if failure := llmContextBudgetCharge(remaining, 2); failure != nil {
		return failure
	}
	for _, character := range value {
		amount := utf8.RuneLen(character)
		switch character {
		case '"', '\\', '\b', '\f', '\n', '\r', '\t':
			amount = 2
		default:
			if character < 0x20 {
				amount = 6
			}
		}
		if failure := llmContextBudgetCharge(remaining, amount); failure != nil {
			return failure
		}
	}
	return nil
}

// Match serde_json's finite numeric representation, retaining integer precision.
func llmContextNumber(value json.Number) (string, error) {
	text := value.String()
	if !strings.ContainsAny(text, ".eE") && text != "-0" {
		if _, failure := strconv.ParseInt(text, 10, 64); failure == nil {
			return text, nil
		}
		if _, failure := strconv.ParseUint(text, 10, 64); failure == nil {
			return text, nil
		}
	}
	number, failure := strconv.ParseFloat(text, 64)
	if failure != nil || math.IsNaN(number) || math.IsInf(number, 0) {
		return "", fmt.Errorf("invalid application execution context number")
	}
	if number == 0 {
		return strconv.FormatFloat(number, 'f', 1, 64), nil
	}
	mantissa, exponentText, _ := strings.Cut(strconv.FormatFloat(number, 'e', -1, 64), "e")
	exponent, failure := strconv.Atoi(exponentText)
	if failure != nil {
		return "", failure
	}
	if exponent < -5 || exponent > 15 {
		sign := ""
		if exponent >= 0 {
			sign = "+"
		}
		return mantissa + "e" + sign + strconv.Itoa(exponent), nil
	}
	sign := ""
	if strings.HasPrefix(mantissa, "-") {
		sign, mantissa = "-", mantissa[1:]
	}
	digits := strings.ReplaceAll(mantissa, ".", "")
	point := exponent + 1
	if point <= 0 {
		return sign + "0." + strings.Repeat("0", -point) + digits, nil
	}
	if point >= len(digits) {
		return sign + digits + strings.Repeat("0", point-len(digits)) + ".0", nil
	}
	return sign + digits[:point] + "." + digits[point:], nil
}

// compactLLMContextInput discards padding before retaining input. The lexical
// budget bounds decoded content; the separate spelling cap bounds parser work.
func compactLLMContextInput(raw []byte, remainingWork *int) ([]byte, error) {
	if err := checkLLMContextInputBudget(raw, 2, remainingWork); err != nil {
		return nil, err
	}
	if !utf8.Valid(raw) {
		return nil, fmt.Errorf("LLM execution context contains invalid UTF-8")
	}
	// Validate the original token boundaries before discarding whitespace.
	// Valid scans syntax without materializing values or retaining padding.
	if !json.Valid(raw) {
		return nil, fmt.Errorf("invalid LLM execution context JSON")
	}
	compact := make([]byte, 0, min(len(raw), maximumLLMApplicationContextBytes))
	quoted, escaped := false, false
	for _, b := range raw {
		if !quoted && (b == ' ' || b == '\n' || b == '\r' || b == '\t') {
			continue
		}
		if len(compact) >= 6*maximumLLMApplicationContextBytes {
			return nil, fmt.Errorf("LLM execution context input exceeds parsing budget")
		}
		compact = append(compact, b)
		if quoted {
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				quoted = false
			}
		} else if b == '"' {
			quoted = true
		}
	}
	if err := validateLLMContextSurrogates(compact); err != nil {
		return nil, err
	}
	return compact, nil
}

func validateLLMContextSurrogates(raw []byte) error {
	invalid := func() error { return fmt.Errorf("LLM execution context contains an unpaired UTF-16 surrogate") }
	for i := 0; i < len(raw); i++ {
		if raw[i] != '\\' {
			continue
		}
		i++
		if i >= len(raw) || raw[i] != 'u' {
			continue
		}
		if i+4 >= len(raw) {
			return invalid()
		}
		unit, err := strconv.ParseUint(string(raw[i+1:i+5]), 16, 16)
		if err != nil {
			return invalid()
		}
		i += 4
		if unit >= 0xdc00 && unit <= 0xdfff {
			return invalid()
		}
		if unit < 0xd800 || unit > 0xdbff {
			continue
		}
		if i+6 >= len(raw) || raw[i+1] != '\\' || raw[i+2] != 'u' {
			return invalid()
		}
		low, err := strconv.ParseUint(string(raw[i+3:i+7]), 16, 16)
		if err != nil || low < 0xdc00 || low > 0xdfff {
			return invalid()
		}
		i += 6
	}
	return nil
}

// Normalize numbers without narrowing supported integer precision, then charge
// only retained object members (encoding/json and serde_json keep the last key).
func chargeLLMContextValue(value any, depth int, remaining *int) (any, error) {
	switch value := value.(type) {
	case nil:
		return nil, llmContextBudgetCharge(remaining, 4)
	case bool:
		return value, llmContextBudgetCharge(remaining, len(strconv.FormatBool(value)))
	case string:
		return value, chargeLLMContextString(value, remaining)
	case json.Number:
		number, err := llmContextNumber(value)
		if err != nil {
			return nil, err
		}
		return json.Number(number), llmContextBudgetCharge(remaining, len(number))
	case []any:
		if depth > 64 {
			return nil, fmt.Errorf("LLM execution context exceeds 64 container levels")
		}
		if err := llmContextBudgetCharge(remaining, 2+max(0, len(value)-1)); err != nil {
			return nil, err
		}
		for i, child := range value {
			normalized, err := chargeLLMContextValue(child, depth+1, remaining)
			if err != nil {
				return nil, err
			}
			value[i] = normalized
		}
		return value, nil
	case map[string]any:
		if depth > 64 {
			return nil, fmt.Errorf("LLM execution context exceeds 64 container levels")
		}
		if err := llmContextBudgetCharge(remaining, 2+max(0, len(value)-1)); err != nil {
			return nil, err
		}
		for key, child := range value {
			if err := chargeLLMContextString(key, remaining); err != nil {
				return nil, err
			}
			if err := llmContextBudgetCharge(remaining, 1); err != nil {
				return nil, err
			}
			normalized, err := chargeLLMContextValue(child, depth+1, remaining)
			if err != nil {
				return nil, err
			}
			value[key] = normalized
		}
		return value, nil
	}
	return nil, fmt.Errorf("invalid LLM application execution context")
}

func marshalLLMApplicationContext(value LLMApplicationExecutionContext) ([]byte, error) {
	if value.Data == nil {
		return []byte("{}"), nil
	}
	remaining := maximumLLMApplicationContextBytes - len(`{"data":{}}`) - max(0, len(value.Data)-1)
	if remaining < 0 {
		return nil, fmt.Errorf("LLM execution context exceeds 16384 serialized bytes")
	}
	fields := make(map[string]any)
	remainingWork := maximumLLMApplicationContextBytes
	for key, raw := range value.Data {
		if err := chargeLLMContextString(key, &remaining); err != nil {
			return nil, err
		}
		if err := llmContextBudgetCharge(&remaining, 1); err != nil {
			return nil, err
		}
		// nil RawMessage means JSON null; a non-nil empty slice remains invalid.
		if raw == nil {
			raw = json.RawMessage("null")
		}
		compact, err := compactLLMContextInput(raw, &remainingWork)
		if err != nil {
			return nil, err
		}
		decoder := json.NewDecoder(bytes.NewReader(compact))
		decoder.UseNumber()
		var decoded any
		if err := decoder.Decode(&decoded); err != nil {
			return nil, err
		}
		if _, err := decoder.Token(); err != io.EOF {
			return nil, fmt.Errorf("execution context must contain one JSON value")
		}
		normalized, err := chargeLLMContextValue(decoded, 3, &remaining)
		if err != nil {
			return nil, err
		}
		fields[key] = normalized
	}
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(map[string]any{"data": fields}); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(output.Bytes(), []byte("\n")), nil
}
