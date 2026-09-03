// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"
)

const (
	cargoManifestName = "Cargo.toml"
	goNativeToolName  = "go-native-tool"
)

var (
	workspacePackagePattern = regexp.MustCompile(`(?ms)^[\t ]*\[workspace\.package\][\t ]*(?:#[^\r\n]*)?\r?\n(.*?)(?:^[\t ]*\[|\z)`)
	workspaceVersionPattern = regexp.MustCompile(`(?m)^[\t ]*version[\t ]*=[\t ]*(?:"([^"\r\n]+)"|'([^'\r\n]+)')[\t ]*(?:#[^\r\n]*)?\r?$`)
)

func withPluginHostStubs(t *testing.T) {
	t.Helper()
	originalInitialize := initializePluginHostJSON
	originalReport := pluginHostActivationReportJSON
	originalIsActive := pluginHostActivationIsActive
	originalClear := clearPluginHostActivation
	originalFree := freePluginHostActivation
	originalReporter := reportPluginHostActivationCleanupError
	t.Cleanup(func() {
		initializePluginHostJSON = originalInitialize
		pluginHostActivationReportJSON = originalReport
		pluginHostActivationIsActive = originalIsActive
		clearPluginHostActivation = originalClear
		freePluginHostActivation = originalFree
		reportPluginHostActivationCleanupError = originalReporter
	})
}

func TestInitializePluginHostSerializesConfigAndOwnsCleanup(t *testing.T) {
	withPluginHostStubs(t)
	token := new(byte)
	ptr := unsafe.Pointer(token)
	pluginsTOML := filepath.Join(t.TempDir(), "plugins.toml")
	var gotConfig map[string]any
	var gotPath *string
	var cleanup []string

	initializePluginHostJSON = func(configJSON string, additional *string) (unsafe.Pointer, string, error) {
		if err := json.Unmarshal([]byte(configJSON), &gotConfig); err != nil {
			t.Fatalf("invalid config JSON: %v", err)
		}
		gotPath = additional
		return ptr, `{"config":{"diagnostics":[]},"dynamic_plugins":[]}`, nil
	}
	pluginHostActivationReportJSON = func(got unsafe.Pointer) (string, error) {
		if got != ptr {
			t.Fatalf("report pointer = %p, want %p", got, ptr)
		}
		return `{"config":{"diagnostics":[]},"dynamic_plugins":[]}`, nil
	}
	pluginHostActivationIsActive = func(got unsafe.Pointer) (bool, error) {
		if got != ptr {
			t.Fatalf("is-active pointer = %p, want %p", got, ptr)
		}
		return true, nil
	}
	clearPluginHostActivation = func(got unsafe.Pointer) error {
		if got != ptr {
			t.Fatalf("clear pointer = %p, want %p", got, ptr)
		}
		cleanup = append(cleanup, "clear")
		return nil
	}
	freePluginHostActivation = func(got unsafe.Pointer) {
		if got != ptr {
			t.Fatalf("free pointer = %p, want %p", got, ptr)
		}
		cleanup = append(cleanup, "free")
	}

	activation, report, err := Initialize(NewPluginConfig(), &pluginsTOML)
	if err != nil {
		t.Fatalf("Initialize() error = %v", err)
	}
	if gotConfig["version"] != float64(1) || gotPath == nil || *gotPath != pluginsTOML {
		t.Fatalf("serialized input = (%#v, %#v)", gotConfig, gotPath)
	}
	if len(report.Config.Diagnostics) != 0 || len(report.DynamicPlugins) != 0 {
		t.Fatalf("report = %#v", report)
	}
	if _, err := activation.Report(); err != nil {
		t.Fatalf("Report() error = %v", err)
	}
	if !activation.IsActive() {
		t.Fatal("IsActive() = false before Close()")
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if activation.IsActive() {
		t.Fatal("IsActive() = true after Close()")
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("repeated Close() error = %v", err)
	}
	if strings.Join(cleanup, ",") != "clear,free" {
		t.Fatalf("cleanup calls = %v", cleanup)
	}
	runtime.KeepAlive(token)
}

func TestInitializePluginHostRejectsInvalidReportAndCleansUp(t *testing.T) {
	withPluginHostStubs(t)
	token := new(byte)
	ptr := unsafe.Pointer(token)
	var cleared, freed bool
	initializePluginHostJSON = func(string, *string) (unsafe.Pointer, string, error) {
		return ptr, "{", nil
	}
	clearPluginHostActivation = func(unsafe.Pointer) error { cleared = true; return nil }
	freePluginHostActivation = func(unsafe.Pointer) { freed = true }
	if activation, _, err := Initialize(NewPluginConfig(), nil); err == nil || activation != nil {
		t.Fatalf("Initialize() = (%#v, %v), want report error", activation, err)
	}
	if !cleared || !freed {
		t.Fatalf("invalid report cleanup = clear:%t free:%t", cleared, freed)
	}
	runtime.KeepAlive(token)
}

func TestPluginHostActivationCloseFreesAfterClearFailure(t *testing.T) {
	withPluginHostStubs(t)
	token := new(byte)
	ptr := unsafe.Pointer(token)
	closeErr := errors.New("clear failed")
	var cleanup []string
	clearPluginHostActivation = func(got unsafe.Pointer) error {
		if got != ptr {
			t.Fatalf("clear pointer = %p, want %p", got, ptr)
		}
		cleanup = append(cleanup, "clear")
		return closeErr
	}
	freePluginHostActivation = func(got unsafe.Pointer) {
		if got != ptr {
			t.Fatalf("free pointer = %p, want %p", got, ptr)
		}
		cleanup = append(cleanup, "free")
	}

	activation := newPluginHostActivation(ptr)
	firstErr := activation.Close()
	if !errors.Is(firstErr, closeErr) {
		t.Fatalf("Close() error = %v, want %v", firstErr, closeErr)
	}
	if strings.Join(cleanup, ",") != "clear,free" {
		t.Fatalf("cleanup calls = %v", cleanup)
	}
	if repeatedErr := activation.Close(); repeatedErr != firstErr {
		t.Fatalf("repeated Close() error = %v, want cached %v", repeatedErr, firstErr)
	}
	if strings.Join(cleanup, ",") != "clear,free" {
		t.Fatalf("repeated cleanup calls = %v", cleanup)
	}
	runtime.KeepAlive(token)
}

func TestNilPluginHostActivationCloseIsSafe(t *testing.T) {
	var activation *PluginHostActivation
	if err := activation.Close(); err != nil {
		t.Fatalf("nil Close() error = %v", err)
	}
}

func TestInitializePluginHostLoadsNativePluginThroughCgo(t *testing.T) {
	t.Setenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1")
	library := preparedPluginFixture(t, "NEMO_RELAY_TEST_NATIVE_PLUGIN")
	manifest := writeGoNativePluginManifest(t, library)
	pluginsTOML := writeGoPluginHostConfig(t, manifest)
	activation, report, err := Initialize(NewPluginConfig(), &pluginsTOML)
	if err != nil {
		t.Fatalf("Initialize() error = %v", err)
	}
	defer func() { _ = activation.Close() }()
	if len(report.DynamicPlugins) != 1 || report.DynamicPlugins[0].PluginID != "fixture_native" {
		t.Fatalf("dynamic reports = %#v", report.DynamicPlugins)
	}

	transformed, err := ToolRequestIntercepts(goNativeToolName, json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(transformed, &payload); err != nil || payload["native_plugin"] != true {
		t.Fatalf("transformed tool args = %s, error = %v", transformed, err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	after, err := ToolRequestIntercepts(goNativeToolName, json.RawMessage(`{"input":true}`))
	if err != nil || string(after) != `{"input":true}` {
		t.Fatalf("tool args after close = %s, error = %v", after, err)
	}
}

func TestInitializePluginHostLoadsWorkerPluginThroughCgo(t *testing.T) {
	t.Setenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1")
	executable := preparedPluginFixture(t, "NEMO_RELAY_TEST_WORKER_PLUGIN")
	manifest := writeGoWorkerPluginManifest(t, executable)
	pluginsTOML := writeGoPluginHostConfig(t, manifest)
	activation, report, err := Initialize(NewPluginConfig(), &pluginsTOML)
	if err != nil {
		t.Fatalf("Initialize() error = %v", err)
	}
	defer func() { _ = activation.Close() }()
	if len(report.DynamicPlugins) != 1 || report.DynamicPlugins[0].PluginID != "fixture_worker" {
		t.Fatalf("dynamic reports = %#v", report.DynamicPlugins)
	}
	transformed, err := ToolRequestIntercepts("go-worker-tool", json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(transformed, &payload); err != nil || payload["worker_plugin"] != true {
		t.Fatalf("transformed tool args = %s, error = %v", transformed, err)
	}
}

func TestPluginHostActivationFinalizerReleasesOwnership(t *testing.T) {
	t.Setenv("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "1")
	library := preparedPluginFixture(t, "NEMO_RELAY_TEST_NATIVE_PLUGIN")
	manifest := writeGoNativePluginManifest(t, library)
	pluginsTOML := writeGoPluginHostConfig(t, manifest)
	createUnclosedPluginHostActivation(t, pluginsTOML)

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.Gosched()
		activation, _, err := Initialize(NewPluginConfig(), &pluginsTOML)
		if err == nil {
			if closeErr := activation.Close(); closeErr != nil {
				t.Fatalf("replacement Close() error = %v", closeErr)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("plugin host finalizer did not release ownership")
}

//go:noinline
func createUnclosedPluginHostActivation(t *testing.T, pluginsTOML string) {
	t.Helper()
	activation, _, err := Initialize(NewPluginConfig(), &pluginsTOML)
	if err != nil {
		t.Fatalf("Initialize() error = %v", err)
	}
	runtime.KeepAlive(activation)
}

func writeGoPluginHostConfig(t *testing.T, manifests ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "plugins.toml")
	var declarations strings.Builder
	for _, manifest := range manifests {
		fmt.Fprintf(&declarations, "[[plugins.dynamic]]\nmanifest = %q\n\n", manifest)
	}
	contents := fmt.Sprintf(
		"version = 1\n\n[plugins.policy.defaults]\nstartup = \"required\"\nattestation = \"integrity_only\"\n\n%s",
		declarations.String(),
	)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write plugins.toml: %v", err)
	}
	return path
}

func preparedPluginFixture(t *testing.T, environment string) string {
	t.Helper()
	path := os.Getenv(environment)
	if path == "" {
		repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			t.Fatal(err)
		}
		filename := "nemo-relay-worker-plugin-fixture"
		if environment == "NEMO_RELAY_TEST_NATIVE_PLUGIN" {
			switch runtime.GOOS {
			case "windows":
				filename = "nemo_relay_plugin_fixture.dll"
			case "darwin":
				filename = "libnemo_relay_plugin_fixture.dylib"
			default:
				filename = "libnemo_relay_plugin_fixture.so"
			}
		} else if runtime.GOOS == "windows" {
			filename += ".exe"
		}
		path = filepath.Join(repoRoot, "target", "test-plugin-fixtures", "debug", filename)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("plugin test fixture %q is missing; run `just build-test-plugin-fixtures`: %v", path, err)
	}
	return path
}

func artifactDigest(t *testing.T, path string) string {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read plugin artifact: %v", err)
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func writeGoNativePluginManifest(t *testing.T, library string) string {
	t.Helper()
	manifest := filepath.Join(t.TempDir(), "relay-plugin.toml")
	contents := fmt.Sprintf(`manifest_version = 1

[plugin]
id = "fixture_native"
kind = "rust_dynamic"

[compat]
relay = "=%s"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[source]
artifact = %q

[integrity]
sha256 = "sha256:%s"

[load]
library = %q
symbol = "nemo_relay_fixture_native_plugin"
`, relayWorkspaceVersion(t), library, artifactDigest(t, library), library)
	if err := os.WriteFile(manifest, []byte(contents), 0o600); err != nil {
		t.Fatalf("write native plugin manifest: %v", err)
	}
	return manifest
}

func writeGoWorkerPluginManifest(t *testing.T, executable string) string {
	t.Helper()
	manifest := filepath.Join(t.TempDir(), "relay-plugin.toml")
	contents := fmt.Sprintf(`manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "worker"

[compat]
relay = "=%s"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[source]
artifact = %q

[integrity]
sha256 = "sha256:%s"

[load]
runtime = "rust"
entrypoint = %q
`, relayWorkspaceVersion(t), executable, artifactDigest(t, executable), executable)
	if err := os.WriteFile(manifest, []byte(contents), 0o600); err != nil {
		t.Fatalf("write worker plugin manifest: %v", err)
	}
	return manifest
}

func relayWorkspaceVersion(t *testing.T) string {
	t.Helper()
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	payload, err := os.ReadFile(filepath.Join(repoRoot, cargoManifestName))
	if err != nil {
		t.Fatalf("read workspace Cargo.toml: %v", err)
	}
	version, err := workspaceVersionFromCargoTOML(payload)
	if err != nil {
		t.Fatal(err)
	}
	return version
}

func workspaceVersionFromCargoTOML(payload []byte) (string, error) {
	section := workspacePackagePattern.FindSubmatch(payload)
	if section == nil {
		return "", errors.New("workspace Cargo.toml has no [workspace.package] section")
	}
	version := workspaceVersionPattern.FindSubmatch(section[1])
	if version == nil {
		return "", errors.New("workspace package version not found")
	}
	if len(version[1]) != 0 {
		return string(version[1]), nil
	}
	return string(version[2]), nil
}

func TestWorkspaceVersionFromCargoTOML(t *testing.T) {
	for _, test := range []struct {
		name, payload, want, wantErr string
	}{
		{"standard", "[workspace.package]\nversion = \"0.8.0\"\n", "0.8.0", ""},
		{"literal", "[workspace.package]\nversion = '0.8.1'\n", "0.8.1", ""},
		{"missing section", "[package]\nversion = \"0.8.0\"\n", "", "no [workspace.package] section"},
		{"missing version", "[workspace.package]\nedition = \"2024\"\n", "", "version not found"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := workspaceVersionFromCargoTOML([]byte(test.payload))
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("error = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("version = %q, error = %v", got, err)
			}
		})
	}
}
