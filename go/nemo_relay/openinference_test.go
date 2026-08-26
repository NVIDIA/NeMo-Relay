// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type otelRequest struct {
	Path        string
	ContentType string
	RelayRoute  string
	Body        []byte
}

func TestOpenInferenceUsesTypedOpenTelemetrySubscriber(t *testing.T) {
	requests := make(chan otelRequest, 1)
	server := NewOtelTestServer(t, requests)
	defer server.Close()

	config := NewOpenTelemetryConfig(OpenTelemetryTypeOpenInference, server.URL+"/v1/traces")
	config.ServiceName = "go-agent"

	subscriber, err := NewOpenTelemetrySubscriber(config)
	if err != nil {
		t.Fatalf("NewOpenTelemetrySubscriber failed: %v", err)
	}
	defer subscriber.Close()

	name := "go_openinference_typed"
	if err := subscriber.Register(name); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	defer func() { _ = subscriber.Deregister(name) }()

	runWithTestScopeStack(t, func() {
		handle, err := PushScope("openinference_scope", ScopeTypeAgent)
		requireNoError(t, err, "PushScope failed")
		requireNoError(t, PopScope(handle), "PopScope failed")
	})
	requireNoError(t, subscriber.ForceFlush(), "ForceFlush failed")

	select {
	case request := <-requests:
		if request.Path != "/v1/traces" {
			t.Fatalf("expected /v1/traces path, got %q", request.Path)
		}
		if request.ContentType != "application/x-protobuf" {
			t.Fatalf("expected protobuf content type, got %q", request.ContentType)
		}
		for _, needle := range [][]byte{[]byte("openinference.span.kind"), []byte("AGENT")} {
			if !bytes.Contains(request.Body, needle) {
				t.Fatalf("expected OTLP request body to contain %q", needle)
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OTLP request")
	}
}

func NewOtelTestServer(t *testing.T, requests chan<- otelRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		requests <- otelRequest{
			Path:        r.URL.Path,
			ContentType: r.Header.Get("Content-Type"),
			RelayRoute:  r.Header.Get("x-relay-route"),
			Body:        body,
		}
		w.WriteHeader(http.StatusOK)
	}))
}
