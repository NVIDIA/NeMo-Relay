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
	return marshalApplicationContext(c)
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
		result[key] = bytes.Clone(value)
	}
	return result
}

const maximumApplicationContextBytes = 16_384

func contextBudgetCharge(remaining *int, amount int) error {
	if amount > *remaining {
		return fmt.Errorf("tool execution context exceeds 16384 serialized bytes")
	}
	*remaining -= amount
	return nil
}

// Reject large strings and containers before encoding/json materializes tokens.
// Escapes contribute a decoded lower bound; whitespace contributes no JSON bytes.
func checkContextInputBudget(input []byte, parentDepth int) error {
	remaining, depth, quoted := maximumApplicationContextBytes, parentDepth, false
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
					return fmt.Errorf("tool execution context exceeds 64 container levels")
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
		if failure := contextBudgetCharge(&remaining, amount); failure != nil {
			return failure
		}
	}
	return nil
}

func chargeContextString(value string, remaining *int) error {
	if len(value) > *remaining {
		return fmt.Errorf("tool execution context exceeds 16384 serialized bytes")
	}
	if failure := contextBudgetCharge(remaining, 2); failure != nil {
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
		if failure := contextBudgetCharge(remaining, amount); failure != nil {
			return failure
		}
	}
	return nil
}

// Match serde_json's finite numeric representation, retaining integer precision.
func contextNumber(value json.Number) (string, error) {
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

func chargeContextValue(decoder *json.Decoder, parentDepth int, remaining *int) error {
	token, failure := decoder.Token()
	if failure != nil {
		return failure
	}
	switch value := token.(type) {
	case string:
		return chargeContextString(value, remaining)
	case json.Number:
		encoded, failure := contextNumber(value)
		if failure != nil {
			return failure
		}
		return contextBudgetCharge(remaining, len(encoded))
	case nil:
		return contextBudgetCharge(remaining, 4)
	case bool:
		return contextBudgetCharge(remaining, len(strconv.FormatBool(value)))
	case json.Delim:
		depth := parentDepth + 1
		if depth > 64 {
			return fmt.Errorf("tool execution context exceeds 64 container levels")
		}
		if failure := contextBudgetCharge(remaining, 2); failure != nil {
			return failure
		}
		closingDelimiter := json.Delim(']')
		var fieldSizes map[string]int
		if value == '{' {
			closingDelimiter = '}'
			fieldSizes = make(map[string]int)
		} else if value != '[' {
			return fmt.Errorf("invalid application execution context")
		}
		for index := 0; decoder.More(); index++ {
			fieldStart := *remaining
			key := ""
			if fieldSizes != nil {
				token, failure := decoder.Token()
				if failure != nil {
					return failure
				}
				var valid bool
				key, valid = token.(string)
				if !valid {
					return fmt.Errorf("invalid application execution context key")
				}
				if previousSize, duplicate := fieldSizes[key]; duplicate {
					// Both JSON decoders retain the last duplicate key.
					*remaining += previousSize
				} else if len(fieldSizes) > 0 {
					if failure := contextBudgetCharge(remaining, 1); failure != nil {
						return failure
					}
				}
				fieldStart = *remaining
				if failure := chargeContextString(key, remaining); failure != nil {
					return failure
				}
				if failure := contextBudgetCharge(remaining, 1); failure != nil {
					return failure
				}
			} else if index > 0 {
				if failure := contextBudgetCharge(remaining, 1); failure != nil {
					return failure
				}
			}
			if failure := chargeContextValue(decoder, depth, remaining); failure != nil {
				return failure
			}
			if fieldSizes != nil {
				fieldSizes[key] = fieldStart - *remaining
			}
		}
		closing, failure := decoder.Token()
		if failure != nil || closing != closingDelimiter {
			return fmt.Errorf("invalid application execution context container")
		}
		return nil
	}
	return fmt.Errorf("invalid application execution context")
}

func validateContextBudget(input []byte, parentDepth int, remaining *int) error {
	if failure := checkContextInputBudget(input, parentDepth); failure != nil {
		return failure
	}
	decoder := json.NewDecoder(bytes.NewReader(input))
	decoder.UseNumber()
	measurementRemaining := math.MaxInt
	if failure := chargeContextValue(decoder, parentDepth, &measurementRemaining); failure != nil {
		return failure
	}
	if _, failure := decoder.Token(); failure != io.EOF {
		return fmt.Errorf("execution context must contain one JSON value")
	}
	return contextBudgetCharge(remaining, math.MaxInt-measurementRemaining)
}

func marshalApplicationContext(context ApplicationExecutionContext) ([]byte, error) {
	if context.Data == nil {
		return []byte("{}"), nil
	}
	remaining := maximumApplicationContextBytes - len(`{"data":{}}`)
	if len(context.Data) > remaining {
		return nil, fmt.Errorf("tool execution context exceeds 16384 serialized bytes")
	}
	processedFields := 0
	for key, value := range context.Data {
		if !utf8.ValidString(key) {
			return nil, fmt.Errorf("application execution context key must be valid UTF-8")
		}
		if processedFields > 0 {
			if failure := contextBudgetCharge(&remaining, 1); failure != nil {
				return nil, failure
			}
		}
		if failure := chargeContextString(key, &remaining); failure != nil {
			return nil, failure
		}
		if failure := contextBudgetCharge(&remaining, 1); failure != nil {
			return nil, failure
		}
		if value == nil {
			value = json.RawMessage("null")
		}
		failure := validateContextBudget(value, 2, &remaining)
		if failure != nil {
			return nil, failure
		}
		processedFields++
	}
	// Copy only after validation, retaining raw numeric representations and nilness.
	return json.Marshal(map[string]any{"data": cloneContextFields(context.Data)})
}
