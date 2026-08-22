<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualify The Integration

Qualification must prove the claims made for an exact host, Relay, adapter, and
package revision. Start with deterministic fixtures. Use live providers only
when the user authorizes spend and provides valid credentials through the
normal environment.

## Establish A Baseline

Run the same deterministic scenario with Relay disabled and enabled. Compare:

- host return values and exceptions;
- provider requests and responses;
- tool arguments and results;
- stream chunks, ordering, and finalization;
- retries, cancellation, timing, and shutdown;
- emitted scopes, events, IDs, and parent relationships.

Absent an explicitly configured blocking or mutation policy, Relay must not
change provider or tool payloads, scheduling, or host-visible results.

## Minimum Deterministic Matrix

Exercise the rows the host actually supports:

| Scenario | Prove |
|---|---|
| One successful unary model call | One balanced call, correct provider/model, request/response projection. |
| One successful tool call | Stable tool call ID, arguments/result, correct parent. |
| Structured tool error | Error result remains a result when the host protocol defines it that way. |
| Rejected pre-tool call | Host callback does not run and rejection reaches the host in its supported form. |
| Mutated tool or model request | The exact value reaching the callback contains the approved mutation. |
| Streaming model call | No premature close, duplicate finalization, or lost terminal state. |
| Provider error and cancellation | Every accepted boundary closes once with truthful status. |
| Provider retry | Attempts are either individually represented or explicitly documented as hidden. |
| Concurrent tools and requests | No cross-request scope leakage or accidental serialization. |
| Out-of-order sibling completion | Isolated branches close independently, or a shared callback stack defers and later drains only owned scopes. |
| Callback thread or event-loop handoff | The intended Relay context is active and the worker's prior binding is restored. |
| Nested managed call | No duplicate instrumentation, event-loop deadlock, or future bound to the wrong owner. |
| Subagent or background task | Parent context is preserved only across boundaries that explicitly propagate it. |
| Compaction or summarization | Hidden model paths are covered or named as gaps. |
| Startup and shutdown | Activation is visible; teardown stops admission, waits for accepted operations, and drains according to the documented contract. |

Add provider-native fixtures for every claimed request codec. Compare the exact
body received by the fixture provider with and without Relay. Do not reduce the
assertion to a normalized message list if the integration promises preserved
provider-only fields.

## Verify Middleware Claims

Test each capability directly:

- **Observable**: assert the expected event and fields exist.
- **Correlated**: assert stable IDs and parent relationships under concurrency.
- **Evaluable**: assert a policy callback ran and returned the expected verdict.
- **Enforceable**: place a canary in the host callback and prove it never ran
  after rejection.
- **Mutable**: assert the callback or provider received the rewritten value.
- **Managed**: assert the complete managed middleware order runs exactly once.
- **Complete**: enumerate every host dispatch path and prove each one, rather
  than extrapolating from the main loop.

Test the disabled and failure paths too. A successful application result does
not prove that Relay loaded, and a Relay event does not prove that all traffic
passed through the same boundary.

For a callback-based shared stack, assert the real completion timestamp and a
snapshot of the output survive deferred close, transient pop failures remain
retryable, invalid output does not strand the scope, and one handler never
closes another handler's scope. For streams, test exhaustion, provider failure,
cancellation, explicit close, no iteration, and a complete response returned
from a nominally streaming path.

## Qualification Levels

Keep these results separate:

1. **Unit and contract fixtures** prove projections, hook returns, ordering, and
   edge-case behavior in process.
2. **Deterministic end-to-end host runs** prove real host scheduling, tools,
   streams, subagents, and Relay integration mechanics against a scripted local
   provider.
3. **Packaged artifact smoke tests** prove installation, native loading,
   activation, cleanup, and compatibility for a distribution and platform.
4. **Live provider tests** prove service authentication and protocol
   compatibility. They do not by themselves prove coverage or policy accuracy.

Report the highest completed level for each distribution. Do not call a
scripted provider a live model qualification.

## False-Result Review

Before concluding:

- verify canaries are semantically valid for the policy being tested;
- distinguish repeated history or duplicate projections from distinct calls;
- inspect raw provider-side fixtures as well as Relay output;
- force the alternate provider, retry, error, cancellation, and subagent paths;
- check that a fallback did not silently bypass Relay;
- compare exact-head results with current CI, not a historical green commit;
- list untested paths and credentials or external services that blocked tests.

## Report Template

Report:

- exact host, Relay, adapter, and package revisions;
- environment and distribution;
- scenarios and evidence;
- baseline-versus-enabled differences;
- capability results by surface;
- false-positive and false-negative checks;
- known gaps and escape paths;
- readiness statement scoped to what the tests actually prove;
- required upstream, Relay, packaging, or documentation follow-ups.
