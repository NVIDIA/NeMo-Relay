// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "context"

// StatusCode preserves a status returned by the native FFI. Unknown is used
// when a pointer-returning API supplies a message without a status code.
type StatusCode int32

const (
	StatusUnknown           StatusCode = -1
	StatusOK                StatusCode = 0
	StatusAlreadyExists     StatusCode = 1
	StatusNotFound          StatusCode = 2
	StatusScopeStackEmpty   StatusCode = 3
	StatusGuardrailRejected StatusCode = 4
	StatusInternal          StatusCode = 5
	StatusNullPointer       StatusCode = 6
	StatusInvalidJSON       StatusCode = 7
	StatusInvalidUTF8       StatusCode = 8
	StatusInvalidArgument   StatusCode = 9
	StatusCancelled         StatusCode = 10
)

func (code StatusCode) String() string {
	switch code {
	case StatusOK:
		return "ok"
	case StatusAlreadyExists:
		return "already exists"
	case StatusNotFound:
		return "not found"
	case StatusScopeStackEmpty:
		return "scope stack empty"
	case StatusGuardrailRejected:
		return "guardrail rejected"
	case StatusInternal:
		return "internal error"
	case StatusNullPointer:
		return "null pointer"
	case StatusInvalidJSON:
		return "invalid JSON"
	case StatusInvalidUTF8:
		return "invalid UTF-8"
	case StatusInvalidArgument:
		return "invalid argument"
	case StatusCancelled:
		return "cancelled"
	default:
		return "unknown native status"
	}
}

// RelayError holds the native FFI status, if available, and the original
// error message. Use errors.As to inspect Code and errors.Is
// with the status errors below. Go validation can return other errors.
type RelayError struct {
	Code    StatusCode
	Message string
}

func (err *RelayError) Error() string {
	if err.Message != "" {
		return err.Message
	}
	return err.Code.String()
}
func (err *RelayError) Is(target error) bool {
	switch target := target.(type) {
	case statusSentinel:
		return err.Code == StatusCode(target)
	case *RelayError:
		return target != nil && err.Code == target.Code
	default:
		return false
	}
}

// Unwrap makes native cancellation compatible with context.Canceled.
func (err *RelayError) Unwrap() error {
	if err.Code == StatusCancelled {
		return context.Canceled
	}
	return nil
}

type statusSentinel StatusCode

func (code statusSentinel) Error() string { return StatusCode(code).String() }

var (
	ErrUnknownStatus     = statusSentinel(StatusUnknown)
	ErrAlreadyExists     = statusSentinel(StatusAlreadyExists)
	ErrNotFound          = statusSentinel(StatusNotFound)
	ErrScopeStackEmpty   = statusSentinel(StatusScopeStackEmpty)
	ErrGuardrailRejected = statusSentinel(StatusGuardrailRejected)
	ErrInternal          = statusSentinel(StatusInternal)
	ErrNullPointer       = statusSentinel(StatusNullPointer)
	ErrInvalidJSON       = statusSentinel(StatusInvalidJSON)
	ErrInvalidUTF8       = statusSentinel(StatusInvalidUTF8)
	ErrInvalidArgument   = statusSentinel(StatusInvalidArgument)
	ErrCancelled         = statusSentinel(StatusCancelled)
)
