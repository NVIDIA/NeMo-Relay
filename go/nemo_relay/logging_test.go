// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

const loggingHelperEnvironment = "NEMO_RELAY_TEST_LOGGING_HELPER"

var loggingEnvironmentNames = map[string]struct{}{
	"NEMO_RELAY_LOG":               {},
	"NEMO_RELAY_LOG_STDERR_FORMAT": {},
	"NEMO_RELAY_LOG_CONFIG_PATH":   {},
}

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
	if os.Getenv(loggingHelperEnvironment) == "1" {
		return
	}

	t.Run("initializes from environment", func(t *testing.T) {
		command := exec.Command(os.Args[0], "-test.run=TestBindingLoggingEnvironment")
		command.Env = loggingTestEnvironment(
			loggingHelperEnvironment+"=1",
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
	})

	t.Run("rejects invalid environment", func(t *testing.T) {
		command := exec.Command(os.Args[0], "-test.run=TestBindingLoggingEnvironment")
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
	})
}
