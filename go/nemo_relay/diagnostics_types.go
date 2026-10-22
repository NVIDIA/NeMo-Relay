// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// DiagnosticNumber is a lossless u64 encoded as a canonical decimal JSON string.
type DiagnosticNumber uint64

// MarshalJSON preserves all 64 bits in the portable diagnostic contract.
func (n DiagnosticNumber) MarshalJSON() ([]byte, error) {
	return json.Marshal(strconv.FormatUint(uint64(n), 10))
}

// UnmarshalJSON rejects floats, leading zeroes, negative values, and overflow.
func (n *DiagnosticNumber) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	value, err := strconv.ParseUint(text, 10, 64)
	if err != nil || strconv.FormatUint(value, 10) != text {
		return fmt.Errorf("invalid diagnostic integer")
	}
	*n = DiagnosticNumber(value)
	return nil
}

// SourceIdentity is the canonical diagnostic source identity view.
type SourceIdentity struct {
	// Unique incarnation assigned before logging initialization.
	InstanceID string `json:"instance_id"`
	// Descriptive operating system process ID, never an identity key.
	ProcessID *uint32 `json:"process_id"`
	// Producer-defined role, such as embedded runtime, gateway, broker, or worker.
	ProcessRole string `json:"process_role"`
	// Relay package version producing the evidence.
	RelayVersion string `json:"relay_version"`
}

// ComponentIdentity is the canonical diagnostic component identity view.
type ComponentIdentity struct {
	// Stable logical name, excluding request and session identifiers.
	Key string `json:"key"`
	// Unique registration instance, distinguishing replacement components.
	InstanceID string `json:"instance_id"`
	// Source-local registration generation, increasing when a logical key is replaced.
	Generation DiagnosticNumber `json:"generation"`
}

// ComponentOwnership is the canonical diagnostic component ownership view.
type ComponentOwnership struct {
	// Subscriber registration associated with this attribution, when applicable.
	SubscriberInstanceID *string `json:"subscriber_instance_id"`
	// Plugin registration associated with this attribution, when applicable.
	PluginInstanceID *string `json:"plugin_instance_id"`
	// Bounded identity of the committed configuration.
	ConfigurationID *string `json:"configuration_id"`
	// Bounded runtime responsibility for an embedded or process owner.
	Responsibility *string `json:"responsibility"`
}

// DiagnosticOrigin is the canonical diagnostic diagnostic origin view.
type DiagnosticOrigin struct {
	// Refresh generation associated with the originating work or observation.
	RefreshGeneration DiagnosticNumber `json:"refresh_generation"`
	// Original bounded owner and configuration attribution.
	Ownership ComponentOwnership `json:"ownership"`
}

// DiagnosticCorrelation is the canonical diagnostic diagnostic correlation view.
type DiagnosticCorrelation struct {
	// Trajectory session associated with the observation, when known.
	SessionID *string `json:"session_id"`
	// Request associated with the observation, when known.
	RequestID *string `json:"request_id"`
}

// Classification is the canonical diagnostic classification view.
type Classification struct {
	// Stable built-in name or registered namespaced extension name.
	CanonicalCode string `json:"canonical_code"`
	// Cause name; absent for successful observations and notices.
	FailureClass *string `json:"failure_class"`
}

// DiagnosticDetail is the canonical diagnostic diagnostic detail view.
type DiagnosticDetail struct {
	// Text copied within admitted byte capacity; bounding does not sanitize it.
	Message string `json:"message"`
	// Whether the original text exceeded its retained byte allowance.
	Truncated bool `json:"truncated"`
}

// DiagnosticEvent is the canonical diagnostic diagnostic event view.
type DiagnosticEvent struct {
	// Unique source-local update sequence; it is not an occurrence counter.
	Sequence DiagnosticNumber `json:"sequence"`
	// Source observation time, preserved through projections.
	ObservedAt string `json:"observed_at"`
	// Registration that owned the original observation.
	Component ComponentIdentity `json:"component"`
	// Original configuration attribution, preserved through component refresh.
	Origin DiagnosticOrigin `json:"origin"`
	// Authenticated remote submission provenance, absent for local observations.
	Remote *RemoteDiagnosticOrigin `json:"remote"`
	// Canonical event classification; never reconstructed from a latest message.
	Classification Classification `json:"classification"`
	// Bounded plugin-operation wire facts, excluded from aggregation keys.
	Context PluginObservationContext `json:"context"`
	// Evidence severity, independent of its observed operation outcome.
	Severity string `json:"severity"`
	// Explicit observed operation outcome.
	Outcome string `json:"outcome"`
	// Bounded optional evidence; absence does not imply a missing counter.
	Detail *DiagnosticDetail `json:"detail"`
	// Input occurrence delta represented by this recording call.
	Count DiagnosticNumber `json:"count"`
	// Restricted caller correlation, excluded from aggregation identity.
	Correlation DiagnosticCorrelation `json:"correlation"`
}

// RemoteDiagnosticOrigin is the canonical diagnostic remote diagnostic origin view.
type RemoteDiagnosticOrigin struct {
	// Host-issued diagnostic session/source incarnation.
	SourceInstanceID string `json:"source_instance_id"`
	// Host-issued plugin activation incarnation, distinct from the session.
	ActivationInstanceID string `json:"activation_instance_id"`
	// Ordered terminal submission sequence; not the host event sequence.
	SubmissionSequence DiagnosticNumber `json:"submission_sequence"`
	// Optional producer-reported wall time, with unknown clock uncertainty.
	ObservedAt *string `json:"observed_at"`
	// Host wall time at acceptance, also used as the event observation time.
	ReceivedAt string `json:"received_at"`
	// Host monotonic elapsed time at acceptance, relative to context startup.
	AcceptedElapsedMillis DiagnosticNumber `json:"accepted_elapsed_millis"`
}

// DiagnosticCount is the canonical diagnostic diagnostic count view.
type DiagnosticCount struct {
	// Exact count unless saturation has occurred.
	Value DiagnosticNumber `json:"value"`
	// Whether an overflowing increment made the value a lower bound.
	Saturated bool `json:"saturated"`
}

// DiagnosticTotals is the canonical diagnostic diagnostic totals view.
type DiagnosticTotals struct {
	// Accepted recording calls, one per call regardless of occurrence delta.
	RecordedEventsTotal DiagnosticCount `json:"recorded_events_total"`
	// Accepted occurrence deltas across all outcomes and codes.
	OccurrencesTotal DiagnosticCount `json:"occurrences_total"`
	// Previously retained events owned by this component or context and evicted.
	HistoryEvictionsTotal DiagnosticCount `json:"history_evictions_total"`
	// Occurrences whose owning operation explicitly succeeded.
	SuccessesTotal DiagnosticCount `json:"successes_total"`
	// Occurrences whose owning operation explicitly failed.
	FailuresTotal DiagnosticCount `json:"failures_total"`
	// Occurrences without an established operation outcome.
	UnspecifiedTotal DiagnosticCount `json:"unspecified_total"`
}

// DiagnosticLifetimeSummary is the canonical diagnostic diagnostic lifetime summary view.
type DiagnosticLifetimeSummary struct {
	// Whole-context or whole-component-instance accounting, as named by the view.
	Totals DiagnosticTotals `json:"totals"`
	// Monotonic elapsed lifetime measured by the source at capture.
	ElapsedMillis DiagnosticNumber `json:"elapsed_millis"`
	// Wall-clock correlation time of the latest observed lifetime failure.
	LastFailureAt *string `json:"last_failure_at"`
	// Monotonic elapsed time since that failure; absence never means recovery.
	SinceLastFailureMillis *DiagnosticNumber `json:"since_last_failure_millis"`
}

// DiagnosticRefreshSummary is the canonical diagnostic diagnostic refresh summary view.
type DiagnosticRefreshSummary struct {
	// Component-local generation increased by a successful refresh.
	Generation DiagnosticNumber `json:"generation"`
	// Wall-clock correlation anchor for the successful refresh.
	StartedAt string `json:"started_at"`
	// Monotonic elapsed duration since the refresh boundary.
	ElapsedMillis DiagnosticNumber `json:"elapsed_millis"`
	// Current configuration and owner attribution for this counting period.
	Ownership ComponentOwnership `json:"ownership"`
	// Counts for this current period, independently reserved from lifetime totals.
	Totals DiagnosticTotals `json:"totals"`
}

// DiagnosticCounter is the canonical diagnostic diagnostic counter view.
type DiagnosticCounter struct {
	// Canonical code; the component instance comes from the containing row.
	Code string `json:"code"`
	// Accepted occurrences, independent of optional recent detail.
	Occurrences DiagnosticCount `json:"occurrences"`
	// Source time when this retained counter was last updated.
	LastObservedAt string `json:"last_observed_at"`
}

// DiagnosticWindowBucket is the canonical diagnostic diagnostic window bucket view.
type DiagnosticWindowBucket struct {
	// Inclusive elapsed offset from the component's monotonic start.
	StartElapsedMillis DiagnosticNumber `json:"start_elapsed_millis"`
	// Exclusive nominal bucket end; coverage describes the observed portion.
	EndElapsedMillis DiagnosticNumber `json:"end_elapsed_millis"`
	// Explicit successful-operation occurrences observed in this interval.
	Successes DiagnosticCount `json:"successes"`
	// Explicit failed-operation occurrences observed in this interval.
	Failures DiagnosticCount `json:"failures"`
	// Occurrences with no established operation outcome in this interval.
	Unspecified DiagnosticCount `json:"unspecified"`
}

// DiagnosticWindowCoverage is the canonical diagnostic diagnostic window coverage view.
type DiagnosticWindowCoverage struct {
	// Availability of the admitted window series.
	Availability string `json:"availability"`
	// Earliest covered monotonic elapsed offset within this component lifetime.
	StartElapsedMillis DiagnosticNumber `json:"start_elapsed_millis"`
	// Latest covered monotonic elapsed offset at capture time.
	EndElapsedMillis DiagnosticNumber `json:"end_elapsed_millis"`
	// The first returned bucket is only partly inside the covered interval.
	PartialStart bool `json:"partial_start"`
	// The last returned bucket is still being observed or partly selected.
	PartialEnd bool `json:"partial_end"`
	// Known loss prevents treating the aggregate as a complete interval.
	Incomplete bool `json:"incomplete"`
}

// DiagnosticWindow is the canonical diagnostic diagnostic window view.
type DiagnosticWindow struct {
	// Duration of each fixed bucket.
	ResolutionMillis DiagnosticNumber `json:"resolution_millis"`
	// Configured bucket slots; memory is independent of observation rate.
	BucketCapacity uint32 `json:"bucket_capacity"`
	// Bounded selected aggregates; idle covered intervals may have zero values.
	Buckets []DiagnosticWindowBucket `json:"buckets"`
	// Actual interval and edge limitations.
	Coverage DiagnosticWindowCoverage `json:"coverage"`
}

// DiagnosticCursor is the canonical diagnostic diagnostic cursor view.
type DiagnosticCursor struct {
	// Source incarnation whose retained state was selected.
	SourceInstanceID string `json:"source_instance_id"`
	// Source revision for detecting intervening mutations or reclamation.
	Sequence DiagnosticNumber `json:"sequence"`
	// Next component-instance position in the bounded source ordering.
	ComponentOffset DiagnosticNumber `json:"component_offset"`
	// Next recent-event position in the bounded source ordering.
	EventOffset DiagnosticNumber `json:"event_offset"`
	// Source-validated fingerprint of the intersected query scope.
	Scope DiagnosticNumber `json:"scope"`
}

// DiagnosticQuery is the canonical diagnostic diagnostic query view.
type DiagnosticQuery struct {
	// Select one logical component key across retained registrations.
	ComponentKey *string `json:"component_key,omitempty"`
	// Select one registration incarnation.
	ComponentInstanceID *string `json:"component_instance_id,omitempty"`
	// Select recent events by original refresh generation only. Lifetime and current-refresh summaries keep their declared scope.
	OriginGeneration *DiagnosticNumber `json:"origin_generation,omitempty"`
	// Select exact canonical code counters and events.
	Code *string `json:"code,omitempty"`
	// Select the authoritative catalogue parent domain.
	Domain *string `json:"domain,omitempty"`
	// Select the plugin catalogue family derived from code ownership.
	PluginFamily *string `json:"plugin_family,omitempty"`
	// Select event causes only; code counters do not retain cause breakdowns.
	FailureClass *string `json:"failure_class,omitempty"`
	// Select component rows by current-refresh ownership and events by original ownership. Selected lifetime counters still span the entire component instance.
	Ownership ComponentOwnership `json:"ownership,omitempty"`
	// Select later recent events; summary and code counters remain cumulative.
	AfterSequence *DiagnosticNumber `json:"after_sequence,omitempty"`
	// Requested component-instance row cap, further constrained by core limits.
	MaxComponents *uint32 `json:"max_components,omitempty"`
	// Requested recent-event cap, further constrained by core limits.
	MaxEvents *uint32 `json:"max_events,omitempty"`
	// Requested encoded response cap, further constrained by core limits.
	MaxResponseBytes *uint32 `json:"max_response_bytes,omitempty"`
	// Resume a compatible bounded selection without granting additional scope.
	Continuation *DiagnosticCursor `json:"continuation,omitempty"`
}

// DiagnosticRetention is the canonical diagnostic diagnostic retention view.
type DiagnosticRetention struct {
	// Oldest sequence in the selected retained event history, if present.
	OldestEventSequence *DiagnosticNumber `json:"oldest_event_sequence"`
	// Latest sequence in the selected retained event history, if present.
	LatestEventSequence *DiagnosticNumber `json:"latest_event_sequence"`
	// Number of returned recent events.
	EventsRetained DiagnosticNumber `json:"events_retained"`
	// Number of returned component-instance rows.
	ComponentsRetained DiagnosticNumber `json:"components_retained"`
	// Query limits omitted otherwise matching retained evidence.
	QueryTruncated bool `json:"query_truncated"`
}

// DiagnosticLoss is the canonical diagnostic diagnostic loss view.
type DiagnosticLoss struct {
	// Previously retained events reclaimed from the circular history.
	EventsEvicted DiagnosticCount `json:"events_evicted"`
	// Accepted observations whose incoming event evidence was not retained.
	EventsOmitted DiagnosticCount `json:"events_omitted"`
	// Closed component metadata reclaimed with dependent evidence.
	ComponentsEvicted DiagnosticCount `json:"components_evicted"`
	// Component registrations rejected by admission limits.
	ComponentsRejected DiagnosticCount `json:"components_rejected"`
	// Code reservations or registrations rejected by admission limits.
	KeysRejected DiagnosticCount `json:"keys_rejected"`
	// Observations rejected before canonical counting.
	ObservationsRejected DiagnosticCount `json:"observations_rejected"`
	// Detail shortened within its configured byte allowance.
	MessagesTruncated DiagnosticCount `json:"messages_truncated"`
	// Incoming optional detail omitted without evicting a retained event.
	DetailsOmitted DiagnosticCount `json:"details_omitted"`
	// Correlation fields shortened within their own byte allowance.
	CorrelationsTruncated DiagnosticCount `json:"correlations_truncated"`
	// Safe correlation fields omitted because detail storage could not fit them.
	CorrelationsOmitted DiagnosticCount `json:"correlations_omitted"`
	// Metadata registrations rejected by their admission budget.
	MetadataRejected DiagnosticCount `json:"metadata_rejected"`
	// Accepted summary-only deltas whose classification could not be admitted.
	ClassificationsOverflowed DiagnosticCount `json:"classifications_overflowed"`
	// Window series rejected by their independent admission budget.
	WindowsRejected DiagnosticCount `json:"windows_rejected"`
	// Query materializations rejected by bounded workspace admission.
	QueriesRejected DiagnosticCount `json:"queries_rejected"`
	// Recording attempts for closed or replaced component registrations.
	StaleObservations DiagnosticCount `json:"stale_observations"`
	// Optional projection deliveries omitted for lack of bounded workspace.
	ProjectionDeliveriesOmitted DiagnosticCount `json:"projection_deliveries_omitted"`
	// Projection callbacks whose panic was isolated after acceptance.
	ProjectionsPanicked DiagnosticCount `json:"projections_panicked"`
	// Cyclic or excessively deep projection chains suppressed.
	ProjectionsSuppressed DiagnosticCount `json:"projections_suppressed"`
	// Unique sequence allocation has exhausted its representation.
	SequenceSaturated bool `json:"sequence_saturated"`
}

// DiagnosticCapability is the canonical diagnostic diagnostic capability view.
type DiagnosticCapability struct {
	// Bounded capability name, such as `otel`.
	Name string `json:"name"`
	// Availability of evidence from this capability.
	Availability string `json:"availability"`
	// Bounded explanatory reason, such as `not_integrated`.
	Reason *string `json:"reason"`
}

// DiagnosticCoverage is the canonical diagnostic diagnostic coverage view.
type DiagnosticCoverage struct {
	// Availability of this diagnostic source context.
	Availability string `json:"availability"`
	// Scope of component summaries and component outcome windows.
	SummaryScope string `json:"summary_scope"`
	// Bounded selected history boundaries and query omission.
	Retention DiagnosticRetention `json:"retention"`
	// Reserved cumulative loss and capacity accounting.
	Loss DiagnosticLoss `json:"loss"`
	// Producer integrations whose support or absence is explicitly advertised.
	Capabilities []DiagnosticCapability `json:"capabilities"`
}

// ComponentSnapshot is the canonical diagnostic component snapshot view.
type ComponentSnapshot struct {
	// Logical responsibility and registration incarnation.
	Identity ComponentIdentity `json:"identity"`
	// Source correlation time when the registration was admitted.
	RegisteredAt string `json:"registered_at"`
	// Registration closure; retained summaries remain subject to bounded retention.
	ClosedAt *string `json:"closed_at"`
	// Whole-component-instance lifetime accounting and source elapsed time.
	Lifetime DiagnosticLifetimeSummary `json:"lifetime"`
	// The sole current refresh period, with explicitly scoped current ownership.
	SinceRefresh DiagnosticRefreshSummary `json:"since_refresh"`
	// Selected lifetime code counters; no latest-message set is retained per code.
	Counters []DiagnosticCounter `json:"counters"`
	// Admitted bounded component outcome windows, independent of refresh periods.
	Windows []DiagnosticWindow `json:"windows"`
}

// DiagnosticSnapshot is the canonical diagnostic diagnostic snapshot view.
type DiagnosticSnapshot struct {
	// Supported envelope version, independent of the catalogue version.
	SchemaVersion uint32 `json:"schema_version"`
	// Stable catalogue revision; unknown names stay opaque in this envelope.
	CatalogueVersion uint32 `json:"catalogue_version"`
	// Source identity preserved through collection and projection.
	Source SourceIdentity `json:"source"`
	// Source capture time; it is not a collector receipt timestamp.
	CapturedAt string `json:"captured_at"`
	// Latest source update revision; reads do not increment it.
	Sequence DiagnosticNumber `json:"sequence"`
	// Whole-context lifetime accounting, independent of component/event selectors. Hosts must authorize access to this context-wide view before exposing it.
	Lifetime DiagnosticLifetimeSummary `json:"lifetime"`
	// Bounded rows, one per selected retained component instance.
	Components []ComponentSnapshot `json:"components"`
	// Bounded selected recent evidence.
	Events []DiagnosticEvent `json:"events"`
	// Availability, scope, integration, and loss limitations.
	Coverage DiagnosticCoverage `json:"coverage"`
	// Next bounded page, subject to mutation and retention validation.
	Continuation *DiagnosticCursor `json:"continuation"`
}

// PluginObservationContext is the canonical diagnostic plugin observation context view.
type PluginObservationContext struct {
	// Observed execution name, when known; never inferred from a code.
	Execution *string `json:"execution"`
	// Observed lifecycle phase or policy-gate context, when known.
	Phase *string `json:"phase"`
}

// UnmarshalJSON rejects unsupported envelope versions while preserving unknown names.
func (snapshot *DiagnosticSnapshot) UnmarshalJSON(data []byte) error {
	type plain DiagnosticSnapshot
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value.SchemaVersion != 1 {
		return fmt.Errorf("unsupported diagnostic snapshot schema")
	}
	*snapshot = DiagnosticSnapshot(value)
	return nil
}
