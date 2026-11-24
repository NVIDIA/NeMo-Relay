// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import "encoding/json"

// EventCategory groups events by meaning. It accepts standard ATOF categories
// and custom strings.
type EventCategory string

const (
	EventCategoryAgent     EventCategory = "agent"
	EventCategoryFunction  EventCategory = "function"
	EventCategoryLLM       EventCategory = "llm"
	EventCategoryTool      EventCategory = "tool"
	EventCategoryRetriever EventCategory = "retriever"
	EventCategoryEmbedder  EventCategory = "embedder"
	EventCategoryReranker  EventCategory = "reranker"
	EventCategoryGuardrail EventCategory = "guardrail"
	EventCategoryEvaluator EventCategory = "evaluator"
	EventCategoryCustom    EventCategory = "custom"
	EventCategoryUnknown   EventCategory = "unknown"
)

type atifExporterOptions struct {
	trajectoryID    *string
	toolDefinitions []json.RawMessage
	extra           json.RawMessage
}

// AtifExporterOption configures the existing ATIF exporter constructor.
type AtifExporterOption func(*atifExporterOptions)

// WithAtifTrajectoryID sets a trajectory ID. The default is the session ID.
func WithAtifTrajectoryID(id string) AtifExporterOption {
	return func(o *atifExporterOptions) { o.trajectoryID = &id }
}

// WithAtifToolDefinitions sets agent tool definitions. Nil omits the field.
// a non-nil empty slice writes an explicit empty array.
func WithAtifToolDefinitions(definitions []json.RawMessage) AtifExporterOption {
	return func(o *atifExporterOptions) { o.toolDefinitions = definitions }
}

// WithAtifAgentExtra sets arbitrary JSON agent metadata. Nil omits the field.
func WithAtifAgentExtra(extra json.RawMessage) AtifExporterOption {
	return func(o *atifExporterOptions) { o.extra = extra }
}
