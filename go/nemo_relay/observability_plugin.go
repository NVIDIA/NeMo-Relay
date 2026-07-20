// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// ObservabilityPluginKind is the top-level plugin kind used by the core observability component.
const ObservabilityPluginKind = "observability"

// ObservabilityMarkProjection controls how point-in-time marks are exported.
type ObservabilityMarkProjection string

const (
	// ObservabilityMarkProjectionInherit preserves exporter-native mark handling.
	ObservabilityMarkProjectionInherit ObservabilityMarkProjection = "inherit"
	// ObservabilityMarkProjectionEvent preserves marks as events.
	ObservabilityMarkProjectionEvent ObservabilityMarkProjection = "event"
	// ObservabilityMarkProjectionTool emits visible tool projections.
	ObservabilityMarkProjectionTool ObservabilityMarkProjection = "tool"
)

// ObservabilityConfig is the canonical Go shape for the observability plugin config document.
type ObservabilityConfig struct {
	Version       uint32                   `json:"version,omitempty"`
	Atof          *ObservabilityAtofConfig `json:"atof,omitempty"`
	Atif          *ObservabilityAtifConfig `json:"atif,omitempty"`
	OpenTelemetry *ObservabilityOtlpConfig `json:"opentelemetry,omitempty"`
	OpenInference *ObservabilityOtlpConfig `json:"openinference,omitempty"`
	Policy        *ConfigPolicy            `json:"policy,omitempty"`
}

// ObservabilityAtofConfig configures filesystem-backed raw ATOF JSONL export.
type ObservabilityAtofConfig struct {
	Enabled bool                              `json:"enabled,omitempty"`
	Sinks   []ObservabilityAtofSinkConfigurer `json:"sinks,omitempty"`
}

// ObservabilityAtofSinkConfigurer is one ATOF destination.
type ObservabilityAtofSinkConfigurer interface {
	atofSinkConfig()
}

// ObservabilityAtofFileSinkConfig configures one filesystem ATOF JSONL destination.
type ObservabilityAtofFileSinkConfig struct {
	OutputDirectory string `json:"output_directory,omitempty"`
	Filename        string `json:"filename,omitempty"`
	Mode            string `json:"mode,omitempty"`
}

func (ObservabilityAtofFileSinkConfig) atofSinkConfig() {
	// This marker method intentionally has no runtime behavior.
}

// MarshalJSON serializes the fixed file sink discriminator.
func (config ObservabilityAtofFileSinkConfig) MarshalJSON() ([]byte, error) {
	type alias ObservabilityAtofFileSinkConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{Type: "file", alias: alias(config)})
}

// ObservabilityAtofStreamSinkConfig configures one remote ATOF destination.
type ObservabilityAtofStreamSinkConfig struct {
	URL             string            `json:"url"`
	Transport       string            `json:"transport,omitempty"`
	Headers         map[string]string `json:"headers,omitempty"`
	HeaderEnv       map[string]string `json:"header_env,omitempty"`
	TimeoutMillis   uint64            `json:"timeout_millis,omitempty"`
	FieldNamePolicy string            `json:"field_name_policy,omitempty"`
	Name            string            `json:"name,omitempty"`
}

func (ObservabilityAtofStreamSinkConfig) atofSinkConfig() {
	// This marker method intentionally has no runtime behavior.
}

// MarshalJSON serializes the fixed stream sink discriminator.
func (config ObservabilityAtofStreamSinkConfig) MarshalJSON() ([]byte, error) {
	type alias ObservabilityAtofStreamSinkConfig
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
	}{Type: "stream", alias: alias(config)})
}

// ObservabilityAtofEndpoint is the deprecated name for an ATOF stream sink.
// Deprecated: Use ObservabilityAtofStreamSinkConfig.
type ObservabilityAtofEndpoint = ObservabilityAtofStreamSinkConfig

// ObservabilityAtifConfig configures per-top-level-agent ATIF file export.
type ObservabilityAtifConfig struct {
	Enabled          bool                                 `json:"enabled,omitempty"`
	AgentName        string                               `json:"agent_name,omitempty"`
	AgentVersion     string                               `json:"agent_version,omitempty"`
	ModelName        string                               `json:"model_name,omitempty"`
	ToolDefinitions  []map[string]any                     `json:"tool_definitions,omitempty"`
	Extra            map[string]any                       `json:"extra,omitempty"`
	OutputDirectory  string                               `json:"output_directory,omitempty"`
	FilenameTemplate string                               `json:"filename_template,omitempty"`
	Storage          []ObservabilityAtifStorageConfigurer `json:"storage,omitempty"`
}

// ObservabilityAtifStorageConfigurer is one remote ATIF trajectory storage destination.
type ObservabilityAtifStorageConfigurer interface {
	atifStorageConfig()
}

// ObservabilityS3StorageConfig configures S3-compatible ATIF trajectory upload.
type ObservabilityS3StorageConfig struct {
	Bucket             string `json:"bucket"`
	KeyPrefix          string `json:"key_prefix,omitempty"`
	AccessKeyID        string `json:"access_key_id,omitempty"`
	SecretAccessKeyVar string `json:"secret_access_key_var,omitempty"`
	SessionTokenVar    string `json:"session_token_var,omitempty"`
	Region             string `json:"region,omitempty"`
	EndpointURL        string `json:"endpoint_url,omitempty"`
	AllowHTTP          *bool  `json:"allow_http,omitempty"`
}

func (ObservabilityS3StorageConfig) atifStorageConfig() {
	// Marker method: S3 storage is serialized by MarshalJSON.
}

// MarshalJSON serializes the S3 config with the core plugin's fixed type discriminator.
func (config ObservabilityS3StorageConfig) MarshalJSON() ([]byte, error) {
	type s3StorageJSON struct {
		Type               string `json:"type"`
		Bucket             string `json:"bucket"`
		KeyPrefix          string `json:"key_prefix,omitempty"`
		AccessKeyID        string `json:"access_key_id,omitempty"`
		SecretAccessKeyVar string `json:"secret_access_key_var,omitempty"`
		SessionTokenVar    string `json:"session_token_var,omitempty"`
		Region             string `json:"region,omitempty"`
		EndpointURL        string `json:"endpoint_url,omitempty"`
		AllowHTTP          *bool  `json:"allow_http,omitempty"`
	}
	return json.Marshal(s3StorageJSON{
		Type:               "s3",
		Bucket:             config.Bucket,
		KeyPrefix:          config.KeyPrefix,
		AccessKeyID:        config.AccessKeyID,
		SecretAccessKeyVar: config.SecretAccessKeyVar,
		SessionTokenVar:    config.SessionTokenVar,
		Region:             config.Region,
		EndpointURL:        config.EndpointURL,
		AllowHTTP:          config.AllowHTTP,
	})
}

// ObservabilityHttpStorageConfig configures HTTP ATIF trajectory upload.
type ObservabilityHttpStorageConfig struct {
	Endpoint      string            `json:"endpoint"`
	Headers       map[string]string `json:"headers,omitempty"`
	HeaderEnv     map[string]string `json:"header_env,omitempty"`
	TimeoutMillis uint64            `json:"timeout_millis,omitempty"`
}

func (ObservabilityHttpStorageConfig) atifStorageConfig() {
	// Marker method: HTTP storage is serialized by MarshalJSON.
}

// MarshalJSON serializes the HTTP config with the core plugin's fixed type discriminator.
func (config ObservabilityHttpStorageConfig) MarshalJSON() ([]byte, error) {
	type httpStorageJSON struct {
		Type          string            `json:"type"`
		Endpoint      string            `json:"endpoint"`
		Headers       map[string]string `json:"headers,omitempty"`
		HeaderEnv     map[string]string `json:"header_env,omitempty"`
		TimeoutMillis uint64            `json:"timeout_millis,omitempty"`
	}
	return json.Marshal(httpStorageJSON{
		Type:          "http",
		Endpoint:      config.Endpoint,
		Headers:       config.Headers,
		HeaderEnv:     config.HeaderEnv,
		TimeoutMillis: config.TimeoutMillis,
	})
}

// ObservabilityOtlpConfig configures OpenTelemetry or OpenInference OTLP export.
type ObservabilityOtlpConfig struct {
	Enabled              bool                            `json:"enabled,omitempty"`
	MarkProjection       ObservabilityMarkProjection     `json:"mark_projection,omitempty"`
	MarkExcludeNames     []string                        `json:"mark_exclude_names,omitempty"`
	AttributeMappings    []OtlpAttributeMapping          `json:"attribute_mappings,omitempty"`
	SemanticConvention   OpenTelemetrySemanticConvention `json:"semantic_convention,omitempty"`
	CaptureContent       bool                            `json:"capture_content,omitempty"`
	Transport            string                          `json:"transport,omitempty"`
	Endpoint             string                          `json:"endpoint,omitempty"`
	Headers              map[string]string               `json:"headers,omitempty"`
	ResourceAttributes   map[string]string               `json:"resource_attributes,omitempty"`
	ServiceName          string                          `json:"service_name,omitempty"`
	ServiceNamespace     string                          `json:"service_namespace,omitempty"`
	ServiceVersion       string                          `json:"service_version,omitempty"`
	InstrumentationScope string                          `json:"instrumentation_scope,omitempty"`
	TimeoutMillis        uint64                          `json:"timeout_millis,omitempty"`
}

// MarshalJSON preserves the distinction between a nil exclusion list, which
// inherits the core default, and an explicitly empty list, which disables all
// default exclusions.
func (config ObservabilityOtlpConfig) MarshalJSON() ([]byte, error) {
	type alias ObservabilityOtlpConfig
	payload, err := json.Marshal(alias(config))
	if err != nil {
		return nil, err
	}

	var object map[string]json.RawMessage
	if err := json.Unmarshal(payload, &object); err != nil {
		return nil, err
	}
	if config.MarkExcludeNames != nil {
		exclusions, err := json.Marshal(config.MarkExcludeNames)
		if err != nil {
			return nil, err
		}
		object["mark_exclude_names"] = exclusions
	}
	return json.Marshal(object)
}

// ObservabilityComponentSpec wraps one observability config as a top-level plugin component.
type ObservabilityComponentSpec struct {
	Enabled bool                `json:"enabled,omitempty"`
	Config  ObservabilityConfig `json:"config"`
}

// NewObservabilityConfig returns a default observability config with version 2.
func NewObservabilityConfig() ObservabilityConfig {
	return ObservabilityConfig{Version: 2}
}

// NewObservabilityAtofConfig returns disabled ATOF JSONL settings with native defaults.
func NewObservabilityAtofConfig() ObservabilityAtofConfig {
	return ObservabilityAtofConfig{}
}

// NewObservabilityAtofFileSinkConfig returns one file ATOF sink with native defaults.
func NewObservabilityAtofFileSinkConfig() ObservabilityAtofFileSinkConfig {
	return ObservabilityAtofFileSinkConfig{Mode: "append"}
}

// NewObservabilityAtofStreamSinkConfig returns one stream ATOF sink.
func NewObservabilityAtofStreamSinkConfig(url string) ObservabilityAtofStreamSinkConfig {
	return ObservabilityAtofStreamSinkConfig{URL: url, Transport: "http_post", TimeoutMillis: 3000, FieldNamePolicy: "preserve"}
}

// NewObservabilityAtifConfig returns disabled ATIF settings with core defaults.
func NewObservabilityAtifConfig() ObservabilityAtifConfig {
	return ObservabilityAtifConfig{
		AgentName:        "NeMo Relay",
		ModelName:        "unknown",
		FilenameTemplate: "nemo-relay-atif-{session_id}.json",
	}
}

// NewObservabilityS3StorageConfig returns an S3-compatible ATIF storage destination.
func NewObservabilityS3StorageConfig(bucket string) ObservabilityS3StorageConfig {
	return ObservabilityS3StorageConfig{Bucket: bucket}
}

// NewObservabilityHttpStorageConfig returns an HTTP ATIF storage destination.
func NewObservabilityHttpStorageConfig(endpoint string) ObservabilityHttpStorageConfig {
	return ObservabilityHttpStorageConfig{Endpoint: endpoint}
}

// NewObservabilityOtlpConfig returns disabled OTLP settings with core defaults.
func NewObservabilityOtlpConfig() ObservabilityOtlpConfig {
	return ObservabilityOtlpConfig{
		Transport:          "http_binary",
		MarkProjection:     ObservabilityMarkProjectionInherit,
		MarkExcludeNames:   []string{"llm.chunk"},
		SemanticConvention: OpenTelemetrySemanticConventionGeneric,
		Headers:            map[string]string{},
		ResourceAttributes: map[string]string{},
		ServiceName:        "nemo-relay",
		TimeoutMillis:      3000,
	}
}

// NewObservabilityComponentSpec wraps observability config as an enabled top-level component.
func NewObservabilityComponentSpec(config ObservabilityConfig) ObservabilityComponentSpec {
	return ObservabilityComponentSpec{
		Enabled: true,
		Config:  config,
	}
}

// PluginComponent converts the observability component wrapper into the shared plugin shape.
func (spec ObservabilityComponentSpec) PluginComponent() PluginComponentSpec {
	return PluginComponentSpec{
		Kind:    ObservabilityPluginKind,
		Enabled: spec.Enabled,
		Config:  mustConfigMap(spec.Config),
	}
}

// ObservabilityComponent converts observability config directly into a shared plugin component.
func ObservabilityComponent(config ObservabilityConfig) PluginComponentSpec {
	return NewObservabilityComponentSpec(config).PluginComponent()
}
