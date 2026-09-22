// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const loggingHelperEnvironment = "NEMO_RELAY_TEST_LOGGING_HELPER"

var loggingEnvironmentNames = map[string]struct{}{
	"NEMO_RELAY_LOG":               {},
	"NEMO_RELAY_LOG_STDERR":        {},
	"NEMO_RELAY_LOG_STDERR_FORMAT": {},
	"NEMO_RELAY_LOG_CONFIG_PATH":   {},
}

const loggingEnvironmentTestRunArg = "-test.run=TestBindingLoggingEnvironment"

func loggingTestEnvironment(values ...string) []string {
	environment := make([]string, 0, len(os.Environ())+len(values))
	for _, value := range os.Environ() {
		name, _, _ := strings.Cut(value, "=")
		if _, isLoggingEnvironment := loggingEnvironmentNames[name]; !isLoggingEnvironment {
			environment = append(environment, value)
		}
	}
	return append(environment, values...)
}

func TestBindingLoggingEnvironment(t *testing.T) {
	if helper := os.Getenv(loggingHelperEnvironment); helper != "" {
		if helper == "shutdown" {
			if err := ShutdownLogging(); err != nil {
				t.Fatalf("logging shutdown failed: %v", err)
			}
		} else if helper == "emit" {
			for ordinal, emit := range []func(string, string, map[string]any) error{
				Trace, Debug, Info, Warn, Error,
			} {
				if err := emit("binding", []string{
					"trace message", "debug message", "info message", "warn message", "error message",
				}[ordinal], map[string]any{"ordinal": ordinal}); err != nil {
					t.Fatalf("emit log record: %v", err)
				}
			}
			if err := Info("", "default target", nil); err != nil {
				t.Fatalf("emit default-target log record: %v", err)
			}
			if err := ShutdownLogging(); err != nil {
				t.Fatalf("logging shutdown failed: %v", err)
			}
		}
		return
	}

	t.Run("initializes from environment", testLoggingInitialization)
	t.Run("disables stderr from environment", testLoggingDisabledStderr)
	t.Run("rejects invalid environment", testLoggingInvalidEnvironment)
	t.Run("flushes file sink during shutdown", testLoggingFileSinkShutdown)
	t.Run("emits structured records through every level helper", testBindingLoggingLevels)
}

func testBindingLoggingLevels(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "logging.toml")
	logPath := filepath.Join(directory, "operational.jsonl")
	config := `[logging]
level = "trace"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = ` + strconv.Quote(logPath) + `
level = "trace"
format = "jsonl"
queue_capacity = 16
`
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatalf("write logging config: %v", err)
	}

	command := exec.Command(os.Args[0], loggingEnvironmentTestRunArg)
	command.Env = loggingTestEnvironment(
		loggingHelperEnvironment+"=emit",
		"NEMO_RELAY_LOG_CONFIG_PATH="+configPath,
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("binding log emission failed: %v\n%s", err, output)
	}
	contents, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read operational log: %v", err)
	}
	records := map[string]map[string]any{}
	for _, line := range strings.Split(strings.TrimSpace(string(contents)), "\n") {
		var record map[string]any
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("decode JSONL record: %v", err)
		}
		if message, ok := record["message"].(string); ok {
			records[message] = record
		}
	}
	for ordinal, level := range []string{"trace", "debug", "info", "warn", "error"} {
		record := records[level+" message"]
		if record["level"] != level {
			t.Errorf("%s level = %v, want %q", level, record["level"], level)
		}
		if record["target"] != "nemo_relay.go.binding" {
			t.Errorf("%s target = %v", level, record["target"])
		}
		fields, ok := record["fields"].(map[string]any)
		if !ok || fields["ordinal"] != float64(ordinal) {
			t.Errorf("%s fields = %#v", level, record["fields"])
		}
	}
	if records["default target"]["target"] != "nemo_relay.go" {
		t.Errorf("default target = %v", records["default target"]["target"])
	}
}

func testLoggingDisabledStderr(t *testing.T) {
	command := exec.Command(os.Args[0], loggingEnvironmentTestRunArg)
	command.Env = loggingTestEnvironment(
		loggingHelperEnvironment+"=shutdown",
		"NEMO_RELAY_LOG=info",
		"NEMO_RELAY_LOG_STDERR=false",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("binding initialization failed: %v\n%s", err, output)
	}
	if strings.Contains(string(output), `"event":"logging_initialized"`) {
		t.Fatalf("stderr logging was not disabled:\n%s", output)
	}
}

func testLoggingInitialization(t *testing.T) {
	command := exec.Command(os.Args[0], loggingEnvironmentTestRunArg)
	command.Env = loggingTestEnvironment(
		loggingHelperEnvironment+"=shutdown",
		"NEMO_RELAY_LOG=info",
		"NEMO_RELAY_LOG_STDERR_FORMAT=jsonl",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("binding import failed: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), `"event":"logging_initialized"`) {
		t.Fatalf("logging initialization event missing from output:\n%s", output)
	}
}

func testLoggingInvalidEnvironment(t *testing.T) {
	command := exec.Command(os.Args[0], loggingEnvironmentTestRunArg)
	command.Env = loggingTestEnvironment(
		loggingHelperEnvironment+"=1",
		"NEMO_RELAY_LOG=",
	)
	output, err := command.CombinedOutput()
	if err == nil {
		t.Fatalf("binding initialization unexpectedly succeeded:\n%s", output)
	}
	if !strings.Contains(string(output), "NEMO_RELAY_LOG must not be empty") {
		t.Fatalf("logging initialization error missing from output:\n%s", output)
	}
}

func testLoggingFileSinkShutdown(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "logging.toml")
	logPath := filepath.Join(directory, "operational.jsonl")
	config := `[logging]
level = "info"
stderr_format = "human"
flush_interval_millis = 0

[[logging.sinks]]
path = ` + strconv.Quote(logPath) + `
level = "info"
format = "jsonl"
queue_capacity = 16
`
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatalf("write logging config: %v", err)
	}

	command := exec.Command(os.Args[0], loggingEnvironmentTestRunArg)
	command.Env = loggingTestEnvironment(
		loggingHelperEnvironment+"=shutdown",
		"NEMO_RELAY_LOG_CONFIG_PATH="+configPath,
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("binding logging shutdown failed: %v\n%s", err, output)
	}
	contents, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read operational log: %v", err)
	}
	if !strings.Contains(string(contents), `"event":"logging_shutdown_started"`) {
		t.Fatalf("logging shutdown event missing from file:\n%s", contents)
	}
}
