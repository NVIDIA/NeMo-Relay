// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

/*
#include <stdlib.h>
#include <stdint.h>
typedef struct FfiDiagnosticsReader FfiDiagnosticsReader;
extern int32_t nemo_relay_diagnostics_reader(FfiDiagnosticsReader** out);
extern void nemo_relay_diagnostics_reader_free(FfiDiagnosticsReader** reader);
extern int32_t nemo_relay_diagnostics_snapshot_json(const FfiDiagnosticsReader* reader, const char* query, char** out);
extern int32_t nemo_relay_diagnostics_info_json(const FfiDiagnosticsReader* reader, char** out);
extern uint32_t nemo_relay_diagnostics_query_max_bytes(void);
extern const char* nemo_relay_last_error(void);
extern void nemo_relay_string_free(char* value);
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"sync"
	"unsafe"
)

func maxDiagnosticsQueryBytes() int {
	return int(C.nemo_relay_diagnostics_query_max_bytes())
}

// DiagnosticsSettings is a startup-only host service profile. A supplied profile
// overrides file settings; nil inherits the resolved file profile.
type DiagnosticsSettings struct {
	Enabled bool              `json:"enabled"`
	Limits  map[string]uint64 `json:"limits,omitempty"`
}

// DiagnosticsError provides a stable reader reason without retained evidence.
type DiagnosticsError struct{ Reason string }

func (e *DiagnosticsError) Error() string { return e.Reason }
func diagnosticStatus(status C.int32_t) error {
	if status == 0 {
		return nil
	}
	return &DiagnosticsError{Reason: C.GoString(C.nemo_relay_last_error())}
}

// DiagnosticsReader is a read capability with an explicit, idempotent Close.
// Copies share one lifetime. Inspection remains possible after plugin close.
type DiagnosticsReader struct{ state *diagnosticsReaderState }
type diagnosticsReaderState struct {
	mu  sync.Mutex
	ptr *C.FfiDiagnosticsReader
}

// ReadDiagnostics obtains a capability without initializing diagnostics.
func ReadDiagnostics() (*DiagnosticsReader, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var ptr *C.FfiDiagnosticsReader
	if err := diagnosticStatus(C.nemo_relay_diagnostics_reader(&ptr)); err != nil {
		return nil, err
	}
	state := &diagnosticsReaderState{ptr: ptr}
	runtime.SetFinalizer(state, func(state *diagnosticsReaderState) { state.close() })
	return &DiagnosticsReader{state: state}, nil
}
func (state *diagnosticsReaderState) close() {
	state.mu.Lock()
	defer state.mu.Unlock()
	C.nemo_relay_diagnostics_reader_free(&state.ptr)
}

// Close releases inspection authority; it does not stop the runtime service.
func (reader *DiagnosticsReader) Close() {
	if reader != nil && reader.state != nil {
		reader.state.close()
	}
}

func (reader *DiagnosticsReader) read(query string, info bool) (string, error) {
	if reader == nil || reader.state == nil {
		return "", &DiagnosticsError{Reason: "closed"}
	}
	if len(query) > maxDiagnosticsQueryBytes() {
		return "", &DiagnosticsError{Reason: "invalid_query"}
	}
	state := reader.state
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.ptr == nil {
		return "", &DiagnosticsError{Reason: "closed"}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var out *C.char
	var status C.int32_t
	if info {
		status = C.nemo_relay_diagnostics_info_json(state.ptr, &out)
	} else {
		input := C.CString(query)
		defer C.free(unsafe.Pointer(input))
		status = C.nemo_relay_diagnostics_snapshot_json(state.ptr, input, &out)
	}
	if err := diagnosticStatus(status); err != nil {
		return "", err
	}
	defer C.nemo_relay_string_free(out)
	value := C.GoString(out)
	runtime.KeepAlive(state)
	return value, nil
}

// SnapshotJSON returns one canonical page; the Go string is caller-owned.
func (reader *DiagnosticsReader) SnapshotJSON(query DiagnosticQuery) (string, error) {
	for _, text := range []*string{
		query.ComponentKey, query.ComponentInstanceID, query.Code, query.Domain,
		query.PluginFamily, query.FailureClass, query.Ownership.ConfigurationID,
		query.Ownership.Responsibility, query.Ownership.PluginInstanceID,
		query.Ownership.SubscriberInstanceID,
	} {
		if text != nil && len(*text) > maxDiagnosticsQueryBytes() {
			return "", &DiagnosticsError{Reason: "invalid_query"}
		}
	}
	if query.Continuation != nil && len(query.Continuation.SourceInstanceID) > maxDiagnosticsQueryBytes() {
		return "", &DiagnosticsError{Reason: "invalid_query"}
	}
	payload, err := json.Marshal(query)
	if err != nil {
		return "", err
	}
	return reader.read(string(payload), false)
}

// Snapshot decodes lossless counters and preserves opaque code/context names.
func (reader *DiagnosticsReader) Snapshot(query DiagnosticQuery) (DiagnosticSnapshot, error) {
	var snapshot DiagnosticSnapshot
	raw, err := reader.SnapshotJSON(query)
	if err != nil {
		return snapshot, err
	}
	err = json.Unmarshal([]byte(raw), &snapshot)
	return snapshot, err
}

// InfoJSON returns separate core/copy budgets, occupancy, and logging coverage.
func (reader *DiagnosticsReader) InfoJSON() (string, error) { return reader.read("", true) }
