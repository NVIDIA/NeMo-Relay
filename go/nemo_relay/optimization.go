// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"bytes"
	"encoding/json"
	"errors"
)

// LLMOptimizationKind is an open optimization classification. The constants
// cover Relay's standard kinds; custom string values remain wire-compatible.
type LLMOptimizationKind string

const (
	// LLMOptimizationKindInputCompression identifies request token reduction.
	LLMOptimizationKindInputCompression LLMOptimizationKind = "input_compression"
	// LLMOptimizationKindModelRouting identifies a model-routing decision.
	LLMOptimizationKindModelRouting LLMOptimizationKind = "model_routing"
)

// LLMOptimizationDataSchema identifies an opaque contribution payload schema.
type LLMOptimizationDataSchema struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// LLMOptimizationModel identifies one model for accounting and repricing.
type LLMOptimizationModel struct {
	Model    string  `json:"model"`
	Provider *string `json:"provider,omitempty"`
}

// LLMOptimizationModelTransition describes baseline and effective models.
type LLMOptimizationModelTransition struct {
	Baseline  *LLMOptimizationModel `json:"baseline,omitempty"`
	Effective *LLMOptimizationModel `json:"effective,omitempty"`
}

// LLMOptimizationTokens retains token evidence independently from pricing.
type LLMOptimizationTokens struct {
	PromptTokens     *uint64 `json:"prompt_tokens,omitempty"`
	CompletionTokens *uint64 `json:"completion_tokens,omitempty"`
	CacheReadTokens  *uint64 `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens *uint64 `json:"cache_write_tokens,omitempty"`
	TotalTokens      *uint64 `json:"total_tokens,omitempty"`
}

// LLMOptimizationEvidenceQuality classifies observed versus estimated counts.
type LLMOptimizationEvidenceQuality string

const (
	// LLMOptimizationEvidenceObserved means a provider or runtime observed the count.
	LLMOptimizationEvidenceObserved LLMOptimizationEvidenceQuality = "observed"
	// LLMOptimizationEvidenceEstimated means a tokenizer or estimator produced the count.
	LLMOptimizationEvidenceEstimated LLMOptimizationEvidenceQuality = "estimated"
)

// LLMOptimizationTokenImpact describes baseline, effective, and saved tokens.
type LLMOptimizationTokenImpact struct {
	Baseline         *LLMOptimizationTokens          `json:"baseline,omitempty"`
	Effective        *LLMOptimizationTokens          `json:"effective,omitempty"`
	Saved            *LLMOptimizationTokens          `json:"saved,omitempty"`
	Quality          *LLMOptimizationEvidenceQuality `json:"quality,omitempty"`
	EstimationMethod *string                         `json:"estimation_method,omitempty"`
}

// LLMOptimizationContribution is one plugin's optimization evidence. Extra
// captures unknown top-level fields and MarshalJSON flattens them again, so a
// newer producer can round-trip through this SDK without losing information.
type LLMOptimizationContribution struct {
	ID              *string                         `json:"id,omitempty"`
	Sequence        *uint64                         `json:"sequence,omitempty"`
	Producer        string                          `json:"producer"`
	Kind            LLMOptimizationKind             `json:"kind"`
	Applied         bool                            `json:"applied"`
	ModelTransition *LLMOptimizationModelTransition `json:"model_transition,omitempty"`
	TokenImpact     *LLMOptimizationTokenImpact     `json:"token_impact,omitempty"`
	PayloadSchema   *LLMOptimizationDataSchema      `json:"payload_schema,omitempty"`
	Payload         json.RawMessage                 `json:"payload,omitempty"`
	Extra           map[string]json.RawMessage      `json:"-"`
}

type llmOptimizationContributionWire struct {
	ID              *string                         `json:"id,omitempty"`
	Sequence        *uint64                         `json:"sequence,omitempty"`
	Producer        string                          `json:"producer"`
	Kind            LLMOptimizationKind             `json:"kind"`
	Applied         bool                            `json:"applied"`
	ModelTransition *LLMOptimizationModelTransition `json:"model_transition,omitempty"`
	TokenImpact     *LLMOptimizationTokenImpact     `json:"token_impact,omitempty"`
	PayloadSchema   *LLMOptimizationDataSchema      `json:"payload_schema,omitempty"`
	Payload         json.RawMessage                 `json:"payload,omitempty"`
}

var llmOptimizationContributionKnownFields = [...]string{
	"id",
	"sequence",
	"producer",
	"kind",
	"applied",
	"model_transition",
	"token_impact",
	"payload_schema",
	"payload",
}

func hasLLMOptimizationPayload(payload json.RawMessage) bool {
	trimmed := bytes.TrimSpace(payload)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

// MarshalJSON preserves and flattens forward-compatible top-level fields.
func (c LLMOptimizationContribution) MarshalJSON() ([]byte, error) {
	if hasLLMOptimizationPayload(c.Payload) && c.PayloadSchema == nil {
		return nil, errors.New("LLM optimization contribution payload requires payload_schema")
	}
	known, err := json.Marshal(llmOptimizationContributionWire{
		ID:              c.ID,
		Sequence:        c.Sequence,
		Producer:        c.Producer,
		Kind:            c.Kind,
		Applied:         c.Applied,
		ModelTransition: c.ModelTransition,
		TokenImpact:     c.TokenImpact,
		PayloadSchema:   c.PayloadSchema,
		Payload:         c.Payload,
	})
	if err != nil {
		return nil, err
	}
	fields := make(map[string]json.RawMessage, len(c.Extra)+len(llmOptimizationContributionKnownFields))
	for name, value := range c.Extra {
		fields[name] = value
	}
	for _, name := range llmOptimizationContributionKnownFields {
		delete(fields, name)
	}
	if err := json.Unmarshal(known, &fields); err != nil {
		return nil, err
	}
	return json.Marshal(fields)
}

// UnmarshalJSON decodes known fields and retains unknown top-level fields.
func (c *LLMOptimizationContribution) UnmarshalJSON(data []byte) error {
	var known llmOptimizationContributionWire
	if err := json.Unmarshal(data, &known); err != nil {
		return err
	}
	var extra map[string]json.RawMessage
	if err := json.Unmarshal(data, &extra); err != nil {
		return err
	}
	if extra == nil {
		return errors.New("LLM optimization contribution must be a JSON object")
	}
	if _, ok := extra["producer"]; !ok {
		return errors.New("LLM optimization contribution requires producer")
	}
	if _, ok := extra["kind"]; !ok {
		return errors.New("LLM optimization contribution requires kind")
	}
	if hasLLMOptimizationPayload(known.Payload) && known.PayloadSchema == nil {
		return errors.New("LLM optimization contribution payload requires payload_schema")
	}
	for _, name := range llmOptimizationContributionKnownFields {
		delete(extra, name)
	}
	*c = LLMOptimizationContribution{
		ID:              known.ID,
		Sequence:        known.Sequence,
		Producer:        known.Producer,
		Kind:            known.Kind,
		Applied:         known.Applied,
		ModelTransition: known.ModelTransition,
		TokenImpact:     known.TokenImpact,
		PayloadSchema:   known.PayloadSchema,
		Payload:         known.Payload,
		Extra:           extra,
	}
	return nil
}
