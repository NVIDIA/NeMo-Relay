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
	"strings"
	"sync"
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
	t.Cleanup(func() {
		activateDynamicPluginsJSON = originalActivate
		clearPluginActivation = originalClear
		freePluginActivation = originalFree
	})
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

func TestActivateDynamicPluginsPreservesLegacyOmittedDefaults(t *testing.T) {
	withPluginActivationStubs(t)

	emptyPayload, err := json.Marshal(PluginConfig{})
	if err != nil {
		t.Fatalf("marshal empty plugin config: %v", err)
	}
	if string(emptyPayload) != "{}" {
		t.Fatalf("empty plugin config JSON = %s, want {}", emptyPayload)
	}

	token := new(byte)
	activateDynamicPluginsJSON = func(configJSON, specsJSON string) (unsafe.Pointer, string, error) {
		const wantConfig = `{"components":[{"kind":"fixture.default"}]}`
		if configJSON != wantConfig {
			t.Fatalf("config JSON = %s, want %s", configJSON, wantConfig)
		}
		if specsJSON != "[]" {
			t.Fatalf("dynamic plugin specs JSON = %s, want []", specsJSON)
		}
		return unsafe.Pointer(token), `{"diagnostics":[]}`, nil
	}
	clearPluginActivation = func(unsafe.Pointer) error { return nil }
	freePluginActivation = func(unsafe.Pointer) {}

	activation, _, err := ActivateDynamicPlugins(PluginConfig{
		Components: []PluginComponentSpec{
			{Kind: "fixture.default"},
		},
	}, nil)
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	runtime.KeepAlive(token)
}

func TestActivateDynamicPluginsPreservesExplicitZeroAndFalse(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	activateDynamicPluginsJSON = func(configJSON, specsJSON string) (unsafe.Pointer, string, error) {
		const wantConfig = `{"version":0,"components":[{"kind":"fixture.disabled","enabled":false}]}`
		if configJSON != wantConfig {
			t.Fatalf("config JSON = %s, want %s", configJSON, wantConfig)
		}
		if specsJSON != "[]" {
			t.Fatalf("dynamic plugin specs JSON = %s, want []", specsJSON)
		}
		var roundTrip PluginConfig
		if err := json.Unmarshal([]byte(configJSON), &roundTrip); err != nil {
			t.Fatalf("unmarshal explicit plugin config: %v", err)
		}
		roundTripPayload, err := json.Marshal(roundTrip)
		if err != nil {
			t.Fatalf("remarshal explicit plugin config: %v", err)
		}
		if string(roundTripPayload) != wantConfig {
			t.Fatalf("round-trip config JSON = %s, want %s", roundTripPayload, wantConfig)
		}
		return unsafe.Pointer(token), `{"diagnostics":[]}`, nil
	}
	clearPluginActivation = func(unsafe.Pointer) error { return nil }
	freePluginActivation = func(unsafe.Pointer) {}

	component := (PluginComponentSpec{Kind: "fixture.disabled"}).WithEnabled(false)
	config := PluginConfig{Components: []PluginComponentSpec{component}}
	config.SetVersion(0)
	activation, _, err := ActivateDynamicPlugins(config, nil)
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	runtime.KeepAlive(token)
}

func TestPluginConfigUnmarshalOmissionsResetPresenceAndValues(t *testing.T) {
	staleComponent := NewPluginComponent("fixture.stale")
	staleComponent.Config = map[string]any{"stale": true}
	config := NewPluginConfig()
	config.Version = 7
	config.Components = []PluginComponentSpec{staleComponent}
	config.Policy = &ConfigPolicy{UnknownField: UnsupportedBehaviorError}

	const payload = `{"components":[{"kind":"fixture.fresh"}]}`
	if err := json.Unmarshal([]byte(payload), &config); err != nil {
		t.Fatalf("unmarshal config with omitted defaults: %v", err)
	}
	if config.Version != 0 || config.versionSet {
		t.Fatalf("omitted version retained stale state: %#v", config)
	}
	if config.Policy != nil {
		t.Fatalf("omitted policy retained stale state: %#v", config.Policy)
	}
	if len(config.Components) != 1 {
		t.Fatalf("components = %#v, want one", config.Components)
	}
	component := config.Components[0]
	if component.Kind != "fixture.fresh" || component.Enabled || component.enabledSet || component.Config != nil {
		t.Fatalf("omitted enabled/config retained stale state: %#v", component)
	}

	roundTrip, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("remarshal config with omitted defaults: %v", err)
	}
	if string(roundTrip) != payload {
		t.Fatalf("round-trip config JSON = %s, want %s", roundTrip, payload)
	}
}

func TestComponentWrapperEnabledPresenceSurvivesConversion(t *testing.T) {
	adaptive := NewAdaptiveComponentSpec(NewAdaptiveConfig())
	adaptive.Enabled = false
	explicitDisabled := []PluginComponentSpec{
		adaptive.PluginComponent(),
		NewObservabilityComponentSpec(NewObservabilityConfig()).WithEnabled(false).PluginComponent(),
		NewPricingComponentSpec(NewPricingConfig()).WithEnabled(false).PluginComponent(),
		NewPiiRedactionComponentSpec(NewPiiRedactionConfig()).WithEnabled(false).PluginComponent(),
	}
	legacyDefaults := []PluginComponentSpec{
		(AdaptiveComponentSpec{Config: NewAdaptiveConfig()}).PluginComponent(),
		(ObservabilityComponentSpec{Config: NewObservabilityConfig()}).PluginComponent(),
		(PricingComponentSpec{Config: NewPricingConfig()}).PluginComponent(),
		(PiiRedactionComponentSpec{Config: NewPiiRedactionConfig()}).PluginComponent(),
	}

	assertEnabledPresence := func(component PluginComponentSpec, wantPresent bool) {
		t.Helper()
		payload, err := json.Marshal(component)
		if err != nil {
			t.Fatalf("marshal %s component: %v", component.Kind, err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(payload, &fields); err != nil {
			t.Fatalf("unmarshal %s component JSON: %v", component.Kind, err)
		}
		enabled, present := fields["enabled"]
		if present != wantPresent {
			t.Fatalf("%s enabled presence = %t, want %t: %s", component.Kind, present, wantPresent, payload)
		}
		if wantPresent && string(enabled) != "false" {
			t.Fatalf("%s enabled JSON = %s, want false", component.Kind, enabled)
		}
	}

	for _, component := range explicitDisabled {
		assertEnabledPresence(component, true)
	}
	for _, component := range legacyDefaults {
		assertEnabledPresence(component, false)
	}

	wrapperPayload, err := json.Marshal(
		NewAdaptiveComponentSpec(NewAdaptiveConfig()).WithEnabled(false),
	)
	if err != nil {
		t.Fatalf("marshal explicit adaptive wrapper: %v", err)
	}
	decodedWrapper := NewAdaptiveComponentSpec(NewAdaptiveConfig())
	if err := json.Unmarshal(wrapperPayload, &decodedWrapper); err != nil {
		t.Fatalf("unmarshal explicit adaptive wrapper: %v", err)
	}
	if decodedWrapper.Enabled || !decodedWrapper.enabledSet {
		t.Fatalf("explicit wrapper enabled state was not preserved: %#v", decodedWrapper)
	}
	assertEnabledPresence(decodedWrapper.PluginComponent(), true)

	if err := json.Unmarshal([]byte(`{"config":{}}`), &decodedWrapper); err != nil {
		t.Fatalf("unmarshal adaptive wrapper with omitted enabled: %v", err)
	}
	if decodedWrapper.Enabled || decodedWrapper.enabledSet {
		t.Fatalf("omitted wrapper enabled retained stale state: %#v", decodedWrapper)
	}
	assertEnabledPresence(decodedWrapper.PluginComponent(), false)
}

func TestActivateDynamicPluginsNormalizesNilSpecs(t *testing.T) {
	withPluginActivationStubs(t)

	token := new(byte)
	activateDynamicPluginsJSON = func(_ string, specsJSON string) (unsafe.Pointer, string, error) {
		if specsJSON != "[]" {
			t.Fatalf("nil specs encoded as %q, want []", specsJSON)
		}
		return unsafe.Pointer(token), `{"diagnostics":[]}`, nil
	}
	clearPluginActivation = func(unsafe.Pointer) error { return nil }
	freePluginActivation = func(unsafe.Pointer) {}

	activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), nil)
	if err != nil {
		t.Fatalf("ActivateDynamicPlugins() error = %v", err)
	}
	if err := activation.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	runtime.KeepAlive(token)
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

	activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), nil)
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
	if _, _, err := ActivateDynamicPlugins(invalidConfig, nil); err == nil {
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

	if _, _, err := ActivateDynamicPlugins(NewPluginConfig(), nil); err == nil || err.Error() != "load failed" {
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
	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds() error = %v", err)
	}
	for _, kind := range kinds {
		if kind == "fixture_native" {
			t.Fatal("fixture_native remains registered after Close")
		}
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
	createUnclosedPluginActivation(t)

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.Gosched()
		activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), nil)
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
func createUnclosedPluginActivation(t *testing.T) {
	t.Helper()
	activation, _, err := ActivateDynamicPlugins(NewPluginConfig(), nil)
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
relay = "=0.6.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = %q
symbol = "nemo_relay_fixture_native_plugin"
`, library)
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
relay = "=0.6.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "rust"
entrypoint = %q
`, executable)
	if err := os.WriteFile(manifest, []byte(contents), 0o600); err != nil {
		t.Fatalf("write worker plugin manifest: %v", err)
	}
	return manifest
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
