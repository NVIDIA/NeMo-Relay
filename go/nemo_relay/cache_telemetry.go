// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
)

// CacheMissReasonKind identifies why a cache lookup missed.
type CacheMissReasonKind string

const (
	CacheMissReasonPrefixMismatch        CacheMissReasonKind = "prefix_mismatch"
	CacheMissReasonBelowMinimumThreshold CacheMissReasonKind = "below_minimum_threshold"
	CacheMissReasonRetentionExpired      CacheMissReasonKind = "retention_expired"
	CacheMissReasonRoutingMismatch       CacheMissReasonKind = "routing_mismatch"
	CacheMissReasonEvicted               CacheMissReasonKind = "evicted"
	CacheMissReasonUnsupportedFeature    CacheMissReasonKind = "unsupported_feature"
	CacheMissReasonColdStart             CacheMissReasonKind = "cold_start"
	CacheMissReasonUnknown               CacheMissReasonKind = "unknown"
	CacheMissReasonOther                 CacheMissReasonKind = "other"
)

// CacheMissReason describes a cache miss. Set Description for CacheMissReasonOther.
type CacheMissReason struct {
	Reason      CacheMissReasonKind `json:"reason"`
	Description string              `json:"-"`
}

// MarshalJSON writes the native reason tag and its fields.
func (r CacheMissReason) MarshalJSON() ([]byte, error) {
	switch r.Reason {
	case CacheMissReasonPrefixMismatch, CacheMissReasonBelowMinimumThreshold, CacheMissReasonRetentionExpired, CacheMissReasonRoutingMismatch, CacheMissReasonEvicted, CacheMissReasonUnsupportedFeature, CacheMissReasonColdStart, CacheMissReasonUnknown:
		return json.Marshal(struct {
			Reason CacheMissReasonKind `json:"reason"`
		}{r.Reason})
	case CacheMissReasonOther:
		return json.Marshal(struct {
			Reason      CacheMissReasonKind `json:"reason"`
			Description string              `json:"description"`
		}{r.Reason, r.Description})
	default:
		return nil, fmt.Errorf("unknown cache miss reason %q", r.Reason)
	}
}

// UnmarshalJSON reads and validates the native reason tag.
func (r *CacheMissReason) UnmarshalJSON(data []byte) error {
	var wire struct {
		Reason      CacheMissReasonKind `json:"reason"`
		Description *string             `json:"description"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	result := CacheMissReason{Reason: wire.Reason}
	if wire.Reason == CacheMissReasonOther {
		if wire.Description == nil {
			return fmt.Errorf("other cache miss reason requires description")
		}
		result.Description = *wire.Description
	}
	if _, err := result.MarshalJSON(); err != nil {
		return err
	}
	*r = result
	return nil
}

// CacheMissEvidence holds one of the four evidence types below.
// Use a type switch to inspect decoded evidence.
type CacheMissEvidence interface{ cacheMissEvidence() }

// CacheMissPrefixMismatchEvidence carries prefix mismatch evidence.
type CacheMissPrefixMismatchEvidence struct {
	FirstMismatchSpanID string `json:"first_mismatch_span_id"`
	SequenceIndex       uint32 `json:"sequence_index"`
	ExpectedHashPrefix  string `json:"expected_hash_prefix"`
	ActualHashPrefix    string `json:"actual_hash_prefix"`
}

func (CacheMissPrefixMismatchEvidence) cacheMissEvidence() {}

// MarshalJSON adds the native evidence kind tag.
func (e CacheMissPrefixMismatchEvidence) MarshalJSON() ([]byte, error) {
	type fields CacheMissPrefixMismatchEvidence
	return json.Marshal(struct {
		Kind string `json:"kind"`
		fields
	}{"prefix_mismatch", fields(e)})
}

// CacheMissBelowMinimumThresholdEvidence carries below minimum threshold evidence.
type CacheMissBelowMinimumThresholdEvidence struct {
	ObservedPrefixTokens uint32 `json:"observed_prefix_tokens"`
	RequiredMinTokens    uint32 `json:"required_min_tokens"`
	EstimationSource     string `json:"estimation_source"`
}

func (CacheMissBelowMinimumThresholdEvidence) cacheMissEvidence() {}

// MarshalJSON adds the native evidence kind tag.
func (e CacheMissBelowMinimumThresholdEvidence) MarshalJSON() ([]byte, error) {
	type fields CacheMissBelowMinimumThresholdEvidence
	return json.Marshal(struct {
		Kind string `json:"kind"`
		fields
	}{"below_minimum_threshold", fields(e)})
}

// CacheMissRetentionExpiredEvidence carries retention expired evidence.
type CacheMissRetentionExpiredEvidence struct {
	ObservedGapSecs     float64 `json:"observed_gap_secs"`
	RetentionWindowSecs float64 `json:"retention_window_secs"`
	ProviderSemantics   string  `json:"provider_semantics"`
}

func (CacheMissRetentionExpiredEvidence) cacheMissEvidence() {}

// MarshalJSON adds the native evidence kind tag.
func (e CacheMissRetentionExpiredEvidence) MarshalJSON() ([]byte, error) {
	type fields CacheMissRetentionExpiredEvidence
	return json.Marshal(struct {
		Kind string `json:"kind"`
		fields
	}{"retention_expired", fields(e)})
}

// CacheMissUnknownEvidence carries unknown evidence.
type CacheMissUnknownEvidence struct {
	MissingFacts []string `json:"missing_facts"`
}

func (CacheMissUnknownEvidence) cacheMissEvidence() {}

// MarshalJSON adds the native evidence kind tag.
func (e CacheMissUnknownEvidence) MarshalJSON() ([]byte, error) {
	type fields CacheMissUnknownEvidence
	if e.MissingFacts == nil {
		e.MissingFacts = []string{}
	}
	return json.Marshal(struct {
		Kind string `json:"kind"`
		fields
	}{"unknown", fields(e)})
}

// CacheMissDiagnosis explains a cache miss and suggests an action based on evidence.
type CacheMissDiagnosis struct {
	Summary        string            `json:"summary"`
	Recommendation string            `json:"recommendation"`
	Evidence       CacheMissEvidence `json:"evidence"`
}

// UnmarshalJSON reads evidence into its matching Go type.
func (d *CacheMissDiagnosis) UnmarshalJSON(data []byte) error {
	var wire struct {
		Summary        string          `json:"summary"`
		Recommendation string          `json:"recommendation"`
		Evidence       json.RawMessage `json:"evidence"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var tag struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(wire.Evidence, &tag); err != nil {
		return err
	}
	var evidence CacheMissEvidence
	switch tag.Kind {
	case "prefix_mismatch":
		evidence = &CacheMissPrefixMismatchEvidence{}
	case "below_minimum_threshold":
		evidence = &CacheMissBelowMinimumThresholdEvidence{}
	case "retention_expired":
		evidence = &CacheMissRetentionExpiredEvidence{}
	case "unknown":
		evidence = &CacheMissUnknownEvidence{}
	default:
		return fmt.Errorf("unknown cache miss evidence kind %q", tag.Kind)
	}
	if err := json.Unmarshal(wire.Evidence, evidence); err != nil {
		return err
	}
	*d = CacheMissDiagnosis{Summary: wire.Summary, Recommendation: wire.Recommendation, Evidence: evidence}
	return nil
}
