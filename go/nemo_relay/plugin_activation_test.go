// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

var (
	goNativePluginFixtureOnce sync.Once
	goNativePluginFixturePath string
	goNativePluginFixtureErr  error
	goWorkerPluginFixtureOnce sync.Once
	goWorkerPluginFixturePath string
	goWorkerPluginFixtureErr  error
)

func withPluginActivationStubs(t *testing.T) {
	t.Helper()
	originalActivate := activateDynamicPluginsJSON
	originalClear := clearPluginActivation
	originalFree := freePluginActivation
	originalReporter := reportPluginActivationCleanupError
	t.Cleanup(func() {
		activateDynamicPluginsJSON = originalActivate
		clearPluginActivation = originalClear
		freePluginActivation = originalFree
		reportPluginActivationCleanupError = originalReporter
	})
}

func fixtureDynamicPluginSpecs() []DynamicPluginActivationSpec {
	return []DynamicPluginActivationSpec{{
		PluginID:    "fixture.native",
		Kind:        DynamicPluginKindRustDynamic,
		ManifestRef: "/tmp/relay-plugin.toml",
	}}
}

func TestActivateDynamicPluginsSerializesSpecsAndOwnsCleanup(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	ptr := unsafe.Pointer(token)
	var gotConfig PluginConfig
	var gotSpecs []DynamicPluginActivationSpec
	activateDynamicPluginsJSON = func(configJSON, specsJSON string) (unsafe.Pointer, string, error) {
		if err := json.Unmarshal([]byte(configJSON), &gotConfig); err != nil {
			t.Fatalf("invalid config JSON: %v", err)
		}
		if err := json.Unmarshal([]byte(specsJSON), &gotSpecs); err != nil {
			t.Fatalf("invalid specs JSON: %v", err)
		}
		return ptr, `{"diagnostics":[{"level":"warning","code":"fixture.warning","message":"fixture"}]}`, nil
	}
	var calls []string
	clearPluginActivation = func(got unsafe.Pointer) error {
		if got != ptr {
			t.Fatalf("clear pointer = %p, want %p", got, ptr)
		}
		calls = append(calls, "clear")
		return nil
	}
	freePluginActivation = func(got unsafe.Pointer) {
		if got != ptr {
			t.Fatalf("free pointer = %p, want %p", got, ptr)
		}
		calls = append(calls, "free")
	}

	environment := "/tmp/fixture-environment"
	activation, report, err := ActivateDynamicPlugins(NewPluginConfig(), []DynamicPluginActivationSpec{
		{
			PluginID:       "fixture.worker",
			Kind:           DynamicPluginKindWorker,
			ManifestRef:    "/tmp/relay-plugin.toml",
			EnvironmentRef: &environment,
			Config:         map[string]any{"tag": "go"},
		},
	})
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	if gotConfig.Version != 1 {
		t.Fatalf("config version = %d, want 1", gotConfig.Version)
	}
	if len(gotSpecs) != 1 || gotSpecs[0].PluginID != "fixture.worker" {
		t.Fatalf("specs = %#v", gotSpecs)
	}
	if gotSpecs[0].EnvironmentRef == nil || *gotSpecs[0].EnvironmentRef != environment {
		t.Fatalf("environment ref = %#v", gotSpecs[0].EnvironmentRef)
	}
	if len(report.Diagnostics) != 1 || report.Diagnostics[0].Code != "fixture.warning" {
		t.Fatalf("report = %#v", report)
	}

	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("repeated Close() error = %v", err)
	}
	if strings.Join(calls, ",") != "clear,free" {
		t.Fatalf("cleanup calls = %v", calls)
	}
	runtime.KeepAlive(token)
}

func TestPluginConfigPublicJSONShapeRemainsCompatible(t *testing.T) {
	type envelope struct {
		PluginConfig
		Name string `json:"name"`
	}

	payload, err := json.Marshal(envelope{
		PluginConfig: PluginConfig{Version: 1},
		Name:         "fixture",
	})
	if err != nil {
		t.Fatalf("marshal embedded plugin config: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		t.Fatalf("unmarshal embedded plugin config: %v", err)
	}
	if string(fields["name"]) != `"fixture"` || string(fields["version"]) != "1" {
		t.Fatalf("embedded plugin config lost envelope fields: %s", payload)
	}

	emptyPayload, err := json.Marshal(PluginConfig{})
	if err != nil {
		t.Fatalf("marshal empty plugin config: %v", err)
	}
	if string(emptyPayload) != "{}" {
		t.Fatalf("empty plugin config JSON = %s, want {}", emptyPayload)
	}
}

func TestActivateDynamicPluginsUsesPrivateConfigWireShape(t *testing.T) {
	withPluginActivationStubs(t)

	config := PluginConfig{Components: []PluginComponentSpec{{Kind: "fixture.disabled"}}}
	publicPayload, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal public plugin config: %v", err)
	}
	const wantPublic = `{"components":[{"kind":"fixture.disabled"}]}`
	if string(publicPayload) != wantPublic {
		t.Fatalf("public plugin config JSON = %s, want %s", publicPayload, wantPublic)
	}

	token := new(byte)
	activateDynamicPluginsJSON = func(configJSON, specsJSON string) (unsafe.Pointer, string, error) {
		const wantConfig = `{"components":[{"kind":"fixture.disabled","enabled":false}]}`
		if configJSON != wantConfig {
			t.Fatalf("config JSON = %s, want %s", configJSON, wantConfig)
		}
		const wantSpecs = `[{"plugin_id":"fixture.native","kind":"rust_dynamic","manifest_ref":"/tmp/relay-plugin.toml"}]`
		if specsJSON != wantSpecs {
			t.Fatalf("dynamic plugin specs JSON = %s, want %s", specsJSON, wantSpecs)
		}
		return unsafe.Pointer(token), `{"diagnostics":[]}`, nil
	}
	clearPluginActivation = func(unsafe.Pointer) error { return nil }
	freePluginActivation = func(unsafe.Pointer) {}

	activation, _, err := ActivateDynamicPlugins(config, fixtureDynamicPluginSpecs())
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	runtime.KeepAlive(token)
}

func TestComponentWrappersPreserveDisabledValuesDuringConversion(t *testing.T) {
	disabled := []PluginComponentSpec{
		(AdaptiveComponentSpec{Config: NewAdaptiveConfig()}).PluginComponent(),
		(ObservabilityComponentSpec{Config: NewObservabilityConfig()}).PluginComponent(),
		(PricingComponentSpec{Config: NewPricingConfig()}).PluginComponent(),
		(PiiRedactionComponentSpec{Config: NewPiiRedactionConfig()}).PluginComponent(),
	}
	for _, component := range disabled {
		payload, err := marshalPluginActivationConfig(PluginConfig{
			Components: []PluginComponentSpec{component},
		})
		if err != nil {
			t.Fatalf("marshal %s component: %v", component.Kind, err)
		}
		var wire struct {
			Components []struct {
				Enabled *bool `json:"enabled"`
			} `json:"components"`
		}
		if err := json.Unmarshal(payload, &wire); err != nil {
			t.Fatalf("unmarshal %s component wire payload: %v", component.Kind, err)
		}
		if len(wire.Components) != 1 || wire.Components[0].Enabled == nil || *wire.Components[0].Enabled {
			t.Fatalf("%s component wire payload = %s, want enabled false", component.Kind, payload)
		}
	}
}

func TestActivateDynamicPluginsRejectsEmptySpecsWithoutCallingCgo(t *testing.T) {
	withPluginActivationStubs(t)

	activationCalls := 0
	activateDynamicPluginsJSON = func(string, string) (unsafe.Pointer, string, error) {
		activationCalls++
		return nil, "", errors.New("unexpected CGo call")
	}

	for _, specs := range [][]DynamicPluginActivationSpec{nil, {}} {
		activation, report, err := ActivateDynamicPlugins(NewPluginConfig(), specs)
		if err == nil || !strings.Contains(err.Error(), "at least one dynamic plugin") {
			t.Fatalf("ActivateDynamicPlugins(%#v) error = %v, want empty-spec diagnostic", specs, err)
		}
		if activation != nil || len(report.Diagnostics) != 0 {
			t.Fatalf("ActivateDynamicPlugins(%#v) = (%#v, %#v), want empty outputs", specs, activation, report)
		}
	}
	if activationCalls != 0 {
		t.Fatalf("activation calls = %d, want 0", activationCalls)
	}
}

func TestActivateDynamicPluginsCleansUpInvalidReport(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	ptr := unsafe.Pointer(token)
	activateDynamicPluginsJSON = func(string, string) (unsafe.Pointer, string, error) {
		return ptr, "not-json", nil
	}
	var calls []string
	clearPluginActivation = func(unsafe.Pointer) error {
		calls = append(calls, "clear")
		return nil
	}
	freePluginActivation = func(unsafe.Pointer) { calls = append(calls, "free") }

	activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), fixtureDynamicPluginSpecs())
	if err == nil {
		t.Fatal("ActivateDynamicPlugins() error = nil, want invalid report error")
	}
	if activation != nil {
		t.Fatalf("activation = %#v, want nil", activation)
	}
	if strings.Join(calls, ",") != "clear,free" {
		t.Fatalf("cleanup calls = %v", calls)
	}
	runtime.KeepAlive(token)
}

func TestPluginActivationCloseFreesAfterClearFailure(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	ptr := unsafe.Pointer(token)
	wantErr := errors.New("teardown failed")
	var calls []string
	clearPluginActivation = func(unsafe.Pointer) error {
		calls = append(calls, "clear")
		return wantErr
	}
	freePluginActivation = func(unsafe.Pointer) { calls = append(calls, "free") }
	activation := newPluginActivation(ptr)

	if err := activation.Close(); !errors.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
	if err := activation.Close(); !errors.Is(err, wantErr) {
		t.Fatalf("repeated Close() error = %v, want %v", err, wantErr)
	}
	if strings.Join(calls, ",") != "clear,free" {
		t.Fatalf("cleanup calls = %v", calls)
	}
	runtime.KeepAlive(token)
}

func TestPluginActivationFinalizerReportsClearFailure(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	wantErr := errors.New("teardown failed")
	clearPluginActivation = func(unsafe.Pointer) error { return wantErr }
	freePluginActivation = func(unsafe.Pointer) {}
	var reported error
	reportPluginActivationCleanupError = func(err error) { reported = err }

	finalizePluginActivation(&pluginActivationState{ptr: unsafe.Pointer(token)})
	if !errors.Is(reported, wantErr) {
		t.Fatalf("reported finalizer error = %v, want %v", reported, wantErr)
	}
	runtime.KeepAlive(token)
}

func TestPluginActivationCopiesShareCloseStateAndError(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	ptr := unsafe.Pointer(token)
	wantErr := errors.New("teardown failed")
	var callsMu sync.Mutex
	var calls []string
	clearPluginActivation = func(got unsafe.Pointer) error {
		if got != ptr {
			return fmt.Errorf("clear pointer = %p, want %p", got, ptr)
		}
		callsMu.Lock()
		calls = append(calls, "clear")
		callsMu.Unlock()
		return wantErr
	}
	freePluginActivation = func(got unsafe.Pointer) {
		callsMu.Lock()
		defer callsMu.Unlock()
		if got != ptr {
			calls = append(calls, "free-wrong-pointer")
			return
		}
		calls = append(calls, "free")
	}

	activation := newPluginActivation(ptr)
	copyValue := *activation
	closeErrors := make(chan error, 2)
	var closeCalls sync.WaitGroup
	for _, handle := range []*PluginActivation{activation, &copyValue} {
		closeCalls.Add(1)
		go func(handle *PluginActivation) {
			defer closeCalls.Done()
			closeErrors <- handle.Close()
		}(handle)
	}
	closeCalls.Wait()
	close(closeErrors)
	for err := range closeErrors {
		if !errors.Is(err, wantErr) {
			t.Fatalf("Close() error = %v, want %v", err, wantErr)
		}
	}
	if err := activation.Close(); !errors.Is(err, wantErr) {
		t.Fatalf("repeated Close() error = %v, want %v", err, wantErr)
	}

	callsMu.Lock()
	gotCalls := strings.Join(calls, ",")
	callsMu.Unlock()
	if gotCalls != "clear,free" {
		t.Fatalf("cleanup calls = %s, want clear,free", gotCalls)
	}
	runtime.KeepAlive(token)
}

func TestPluginActivationCopyPreventsEarlyFinalization(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	ptr := unsafe.Pointer(token)
	var callsMu sync.Mutex
	var calls []string
	clearPluginActivation = func(unsafe.Pointer) error {
		callsMu.Lock()
		calls = append(calls, "clear")
		callsMu.Unlock()
		return nil
	}
	freePluginActivation = func(unsafe.Pointer) {
		callsMu.Lock()
		calls = append(calls, "free")
		callsMu.Unlock()
	}

	wrapperCollected := make(chan struct{})
	copyValue := copiedPluginActivationWithGCSentinel(ptr, wrapperCollected)
	deadline := time.Now().Add(5 * time.Second)
	for {
		runtime.GC()
		runtime.Gosched()
		select {
		case <-wrapperCollected:
			goto wrapperWasCollected
		default:
			if time.Now().After(deadline) {
				t.Fatal("unreachable activation wrapper was not collected")
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

wrapperWasCollected:
	for i := 0; i < 3; i++ {
		runtime.GC()
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
	}
	callsMu.Lock()
	gotCalls := strings.Join(calls, ",")
	callsMu.Unlock()
	if gotCalls != "" {
		t.Fatalf("cleanup ran while a copied activation was reachable: %s", gotCalls)
	}

	if err := copyValue.Close(); err != nil {
		t.Fatalf("copied activation Close() error = %v", err)
	}
	callsMu.Lock()
	gotCalls = strings.Join(calls, ",")
	callsMu.Unlock()
	if gotCalls != "clear,free" {
		t.Fatalf("cleanup calls = %s, want clear,free", gotCalls)
	}
	runtime.KeepAlive(copyValue)
	runtime.KeepAlive(token)
}

type pluginActivationGCSentinel struct {
	activation *PluginActivation
	padding    [64]byte
}

//go:noinline
func copiedPluginActivationWithGCSentinel(
	ptr unsafe.Pointer,
	wrapperCollected chan<- struct{},
) PluginActivation {
	activation := newPluginActivation(ptr)
	copyValue := *activation
	sentinel := &pluginActivationGCSentinel{activation: activation}
	runtime.SetFinalizer(sentinel, func(sentinel *pluginActivationGCSentinel) {
		runtime.KeepAlive(sentinel.activation)
		close(wrapperCollected)
	})
	runtime.KeepAlive(activation)
	runtime.KeepAlive(sentinel)
	return copyValue
}

func TestActivateDynamicPluginsSurfacesSerializationAndActivationErrors(t *testing.T) {
	withPluginActivationStubs(t)

	activationCalls := 0
	activateDynamicPluginsJSON = func(string, string) (unsafe.Pointer, string, error) {
		activationCalls++
		return nil, "", errors.New("load failed")
	}

	invalidConfig := NewPluginConfig()
	invalidConfig.Components = append(invalidConfig.Components, PluginComponentSpec{
		Kind:    "fixture",
		Enabled: true,
		Config:  map[string]any{"invalid": make(chan int)},
	})
	if _, _, err := ActivateDynamicPlugins(invalidConfig, fixtureDynamicPluginSpecs()); err == nil {
		t.Fatal("invalid config serialization error = nil")
	}
	if activationCalls != 0 {
		t.Fatalf("activation calls after config serialization failure = %d", activationCalls)
	}

	invalidSpecs := []DynamicPluginActivationSpec{{
		PluginID:    "fixture",
		Kind:        DynamicPluginKindRustDynamic,
		ManifestRef: "/tmp/relay-plugin.toml",
		Config:      map[string]any{"invalid": make(chan int)},
	}}
	if _, _, err := ActivateDynamicPlugins(NewPluginConfig(), invalidSpecs); err == nil {
		t.Fatal("invalid specs serialization error = nil")
	}
	if activationCalls != 0 {
		t.Fatalf("activation calls after specs serialization failure = %d", activationCalls)
	}

	if _, _, err := ActivateDynamicPlugins(NewPluginConfig(), fixtureDynamicPluginSpecs()); err == nil || err.Error() != "load failed" {
		t.Fatalf("activation error = %v, want load failed", err)
	}
	if activationCalls != 1 {
		t.Fatalf("activation calls = %d, want 1", activationCalls)
	}
}

func TestNilPluginActivationCloseIsSafe(t *testing.T) {
	var activation *PluginActivation
	if err := activation.Close(); err != nil {
		t.Fatalf("nil Close() error = %v", err)
	}
}

func TestActivateDynamicPluginsLoadsNativePluginThroughCgo(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration() error = %v", err)
	}
	library := goNativePluginFixture(t)
	manifest := writeGoNativePluginManifest(t, library)
	projectDir := t.TempDir()
	projectConfigDir := filepath.Join(projectDir, ".nemo-relay")
	if err := os.MkdirAll(projectConfigDir, 0o700); err != nil {
		t.Fatalf("MkdirAll(project config) error = %v", err)
	}
	pluginsTOML := filepath.Join(projectConfigDir, "plugins.toml")
	const staticKind = "go.fixture.static_base"
	fileConfig := fmt.Sprintf(`version = 999

[[components]]
kind = %q
enabled = true

[components.config]
source = "project-file"
`, staticKind)
	if err := os.WriteFile(pluginsTOML, []byte(fileConfig), 0o600); err != nil {
		t.Fatalf("WriteFile(plugins.toml) error = %v", err)
	}
	previousCWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd() error = %v", err)
	}
	if err := os.Chdir(projectDir); err != nil {
		t.Fatalf("Chdir(project) error = %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previousCWD); err != nil {
			t.Errorf("restore working directory error = %v", err)
		}
	})
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(projectDir, "xdg"))

	var staticRegistrations atomic.Int32
	var staticCallbacks atomic.Int32
	if err := RegisterPlugin(staticKind, PluginFuncs{
		RegisterFunc: func(config map[string]any, ctx *PluginContext) error {
			if config["source"] != "project-file" {
				return fmt.Errorf("static plugin config = %#v, want project-file source", config)
			}
			staticRegistrations.Add(1)
			return ctx.RegisterToolRequestIntercept(
				"go_static_base",
				-1,
				false,
				func(_ string, args json.RawMessage) json.RawMessage {
					staticCallbacks.Add(1)
					var payload map[string]any
					if err := json.Unmarshal(args, &payload); err != nil {
						return args
					}
					payload["static_saw_dynamic"] = payload["native_plugin"] == true
					payload["go_static_base"] = true
					out, _ := json.Marshal(payload)
					return out
				},
			)
		},
	}); err != nil {
		t.Fatalf("RegisterPlugin() error = %v", err)
	}
	defer func() {
		if err := DeregisterPlugin(staticKind); err != nil {
			t.Errorf("DeregisterPlugin() error = %v", err)
		}
	}()

	activation, report, err := ActivateDynamicPlugins(NewPluginConfig(), []DynamicPluginActivationSpec{{
		PluginID:    "fixture_native",
		Kind:        DynamicPluginKindRustDynamic,
		ManifestRef: manifest,
		Config:      map[string]any{},
	}})
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	defer func() {
		if err := activation.Close(); err != nil {
			t.Errorf("deferred Close() error = %v", err)
		}
	}()
	if len(report.Diagnostics) != 0 {
		t.Fatalf("activation diagnostics = %#v, want none", report.Diagnostics)
	}
	if got := staticRegistrations.Load(); got != 1 {
		t.Fatalf("static registrations = %d, want 1", got)
	}
	// The activation has already resolved its configuration; subsequent file
	// mutations are neither watched nor reloaded.
	if err := os.WriteFile(pluginsTOML, []byte("invalid = ["), 0o600); err != nil {
		t.Fatalf("mutate plugins.toml error = %v", err)
	}

	transformed, err := ToolRequestIntercepts("go-native-tool", json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() error = %v", err)
	}
	var transformedObject map[string]any
	if err := json.Unmarshal(transformed, &transformedObject); err != nil {
		t.Fatalf("transformed tool args are invalid JSON: %v", err)
	}
	if transformedObject["native_plugin"] != true {
		t.Fatalf("transformed tool args = %s, want native_plugin marker", transformed)
	}
	if transformedObject["go_static_base"] != true {
		t.Fatalf("transformed tool args = %s, want static base marker", transformed)
	}
	if transformedObject["static_saw_dynamic"] != false {
		t.Fatalf("transformed tool args = %s, want static callback before dynamic callback", transformed)
	}
	if got := staticCallbacks.Load(); got != 1 {
		t.Fatalf("static callbacks = %d, want 1", got)
	}

	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	afterClose, err := ToolRequestIntercepts("go-native-tool", json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() after Close error = %v", err)
	}
	if string(afterClose) != `{"input":true}` {
		t.Fatalf("tool args after Close = %s, want unchanged args", afterClose)
	}
	if got := staticCallbacks.Load(); got != 1 {
		t.Fatalf("static callbacks after Close = %d, want 1", got)
	}
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds() error = %v", err)
	}
	for _, kind := range kinds {
		if kind == "fixture_native" {
			t.Fatal("fixture_native remains registered after Close")
		}
	}
	if err := os.Remove(pluginsTOML); err != nil {
		t.Fatalf("Remove(plugins.toml) error = %v", err)
	}

	_, _, err = ActivateDynamicPlugins(NewPluginConfig(), []DynamicPluginActivationSpec{{
		PluginID:    "fixture_missing",
		Kind:        DynamicPluginKindRustDynamic,
		ManifestRef: filepath.Join(t.TempDir(), "missing-relay-plugin.toml"),
	}})
	if err == nil || !strings.Contains(err.Error(), "native plugin load failed") {
		t.Fatalf("missing-manifest error = %v, want native plugin load diagnostic", err)
	}
}

func TestActivateDynamicPluginsLoadsWorkerPluginThroughCgo(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration() error = %v", err)
	}

	executable := goWorkerPluginFixture(t)
	manifest := writeGoWorkerPluginManifest(t, executable)
	activation, report, err := ActivateDynamicPlugins(NewPluginConfig(), []DynamicPluginActivationSpec{{
		PluginID:    "fixture_worker",
		Kind:        DynamicPluginKindWorker,
		ManifestRef: manifest,
		Config:      map[string]any{},
	}})
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	defer func() {
		if err := activation.Close(); err != nil {
			t.Errorf("deferred Close() error = %v", err)
		}
	}()
	if len(report.Diagnostics) != 0 {
		t.Fatalf("activation diagnostics = %#v, want none", report.Diagnostics)
	}

	transformed, err := ToolRequestIntercepts("go-worker-tool", json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() error = %v", err)
	}
	var transformedObject map[string]any
	if err := json.Unmarshal(transformed, &transformedObject); err != nil {
		t.Fatalf("transformed tool args are invalid JSON: %v", err)
	}
	if transformedObject["worker_plugin"] != true {
		t.Fatalf("transformed tool args = %s, want worker_plugin marker", transformed)
	}

	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	afterClose, err := ToolRequestIntercepts("go-worker-tool", json.RawMessage(`{"input":true}`))
	if err != nil {
		t.Fatalf("ToolRequestIntercepts() after Close error = %v", err)
	}
	if string(afterClose) != `{"input":true}` {
		t.Fatalf("tool args after Close = %s, want unchanged args", afterClose)
	}
}

func TestPluginActivationFinalizerReleasesHostOwnership(t *testing.T) {
	if err := ClearPluginConfiguration(); err != nil {
		t.Fatalf("ClearPluginConfiguration() error = %v", err)
	}
	library := goNativePluginFixture(t)
	manifest := writeGoNativePluginManifest(t, library)
	specs := []DynamicPluginActivationSpec{{
		PluginID:    "fixture_native",
		Kind:        DynamicPluginKindRustDynamic,
		ManifestRef: manifest,
		Config:      map[string]any{},
	}}
	createUnclosedPluginActivation(t, specs)

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.Gosched()
		activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), specs)
		if err == nil {
			if closeErr := activation.Close(); closeErr != nil {
				t.Fatalf("replacement activation Close() error = %v", closeErr)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("plugin activation finalizer did not release host ownership")
}

// Keep creation in a separate frame so the activation is unreachable when the
// caller starts forcing collection.
//
//go:noinline
func createUnclosedPluginActivation(t *testing.T, specs []DynamicPluginActivationSpec) {
	t.Helper()
	activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), specs)
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	runtime.KeepAlive(activation)
}

func goNativePluginFixture(t *testing.T) string {
	t.Helper()
	goNativePluginFixtureOnce.Do(func() {
		repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			goNativePluginFixtureErr = err
			return
		}
		sourceRoot, err := os.MkdirTemp("", "nemo-relay-go-native-plugin-")
		if err != nil {
			goNativePluginFixtureErr = err
			return
		}
		defer os.RemoveAll(sourceRoot)
		fixtureRoot := filepath.Join(sourceRoot, "native_plugin")
		if err := os.MkdirAll(filepath.Join(fixtureRoot, "src"), 0o700); err != nil {
			goNativePluginFixtureErr = err
			return
		}
		fixtureSource := filepath.Join(repoRoot, "crates", "core", "tests", "fixtures", "native_plugin")
		manifestBytes, err := os.ReadFile(filepath.Join(fixtureSource, "Cargo.toml"))
		if err != nil {
			goNativePluginFixtureErr = err
			return
		}
		pluginPath := filepath.Join(repoRoot, "crates", "plugin")
		manifestContents := strings.Replace(
			string(manifestBytes),
			`nemo-relay-plugin = { path = "../../../../plugin" }`,
			fmt.Sprintf("nemo-relay-plugin = { path = %q }", pluginPath),
			1,
		)
		manifest := filepath.Join(fixtureRoot, "Cargo.toml")
		if err := os.WriteFile(manifest, []byte(manifestContents), 0o600); err != nil {
			goNativePluginFixtureErr = err
			return
		}
		librarySource, err := os.ReadFile(filepath.Join(fixtureSource, "src", "lib.rs"))
		if err != nil {
			goNativePluginFixtureErr = err
			return
		}
		if err := os.WriteFile(filepath.Join(fixtureRoot, "src", "lib.rs"), librarySource, 0o600); err != nil {
			goNativePluginFixtureErr = err
			return
		}
		target := filepath.Join(repoRoot, "target")
		cargo := os.Getenv("CARGO")
		if cargo == "" {
			cargo = "cargo"
		}
		command := exec.Command(cargo, "build", "--quiet", "--manifest-path", manifest, "--target-dir", target)
		if output, err := command.CombinedOutput(); err != nil {
			goNativePluginFixtureErr = fmt.Errorf("build native plugin fixture: %w\n%s", err, output)
			return
		}
		goNativePluginFixturePath = filepath.Join(target, "debug", goNativeLibraryName())
		if _, err := os.Stat(goNativePluginFixturePath); err != nil {
			goNativePluginFixtureErr = fmt.Errorf("native plugin fixture output: %w", err)
		}
	})
	if goNativePluginFixtureErr != nil {
		t.Fatal(goNativePluginFixtureErr)
	}
	return goNativePluginFixturePath
}

func goWorkerPluginFixture(t *testing.T) string {
	t.Helper()
	goWorkerPluginFixtureOnce.Do(func() {
		repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			goWorkerPluginFixtureErr = err
			return
		}
		manifest := filepath.Join(repoRoot, "crates", "core", "tests", "fixtures", "worker_plugin", "Cargo.toml")
		target := filepath.Join(repoRoot, "target")
		cargo := os.Getenv("CARGO")
		if cargo == "" {
			cargo = "cargo"
		}
		command := exec.Command(cargo, "build", "--quiet", "--locked", "--manifest-path", manifest, "--target-dir", target)
		if output, err := command.CombinedOutput(); err != nil {
			goWorkerPluginFixtureErr = fmt.Errorf("build worker plugin fixture: %w\n%s", err, output)
			return
		}
		executable := "nemo-relay-worker-plugin-fixture"
		if runtime.GOOS == "windows" {
			executable += ".exe"
		}
		goWorkerPluginFixturePath = filepath.Join(target, "debug", executable)
		if _, err := os.Stat(goWorkerPluginFixturePath); err != nil {
			goWorkerPluginFixtureErr = fmt.Errorf("worker plugin fixture output: %w", err)
		}
	})
	if goWorkerPluginFixtureErr != nil {
		t.Fatal(goWorkerPluginFixtureErr)
	}
	return goWorkerPluginFixturePath
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

[load]
library = %q
symbol = "nemo_relay_fixture_native_plugin"
`, relayWorkspaceVersion(t), library)
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

[load]
runtime = "rust"
entrypoint = %q
`, relayWorkspaceVersion(t), executable)
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
	payload, err := os.ReadFile(filepath.Join(repoRoot, "Cargo.toml"))
	if err != nil {
		t.Fatalf("read workspace Cargo.toml: %v", err)
	}
	section := string(payload)
	const header = "[workspace.package]"
	start := strings.Index(section, header)
	if start < 0 {
		t.Fatal("workspace Cargo.toml has no [workspace.package] section")
	}
	section = section[start+len(header):]
	if end := strings.Index(section, "\n["); end >= 0 {
		section = section[:end]
	}
	for _, line := range strings.Split(section, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "version =") {
			continue
		}
		version, err := strconv.Unquote(strings.TrimSpace(strings.TrimPrefix(line, "version =")))
		if err != nil {
			t.Fatalf("parse workspace package version: %v", err)
		}
		return version
	}
	t.Fatal("workspace package version not found")
	return ""
}

func goNativeLibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "nemo_relay_plugin_fixture.dll"
	case "darwin":
		return "libnemo_relay_plugin_fixture.dylib"
	default:
		return "libnemo_relay_plugin_fixture.so"
	}
}
