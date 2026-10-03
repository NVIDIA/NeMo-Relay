<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# GLM 5.2 staged-router campaign status

This document records the reproducible experiment contract for the
signal-only versus classifier-enabled staged-router study. It deliberately
does not contain credentials.

## Experimental arms

Both arms evaluate the Terminal-Bench 2.1 registry export (89 tasks) with the
same Hermes commit, Nemo Relay 0.7.1 wheel, Switchyard commit
`5c84c16e84fa781452b1ab9a96a0f12303619824`, provider concurrency 24, setup
concurrency 24, and the same task/runtime settings.

| Arm | Picker | Signal confidence threshold | Ambiguous-request behavior | Efficient completion model | Capable completion model | Judge model |
|---|---|---:|---|---|---|---|
| Signal-only | `efficient_first` | 0.5 | Select the efficient tier | `nvidia/zai-org/glm-5.2` | `openai/openai/gpt-5.6-sol` | None |
| Classifier-enabled | `efficient_first` | 0.5 | Consult GLM 5.2 classifier, then select a tier | `nvidia/zai-org/glm-5.2` | `openai/openai/gpt-5.6-sol` | `nvidia/zai-org/glm-5.2` |

Sol uses `reasoning.effort = "medium"`. GLM uses
`reasoning.enabled = false` for serving and judge calls. Both targets use the
InferenceHub OpenAI Chat endpoint (`/v1/chat/completions`). The current pinned
Switchyard native plugin rejects a routed `openai_responses` default target;
the admission validator rejects that unsupported staged-router transport before
any paid run can begin.

## Pricing catalog bound into Relay

All rates are USD per million tokens and are taken from the OpenRouter model
catalog snapshot dated 2026-08-14. The catalog is an a-priori accounting model,
not an invoice reconciliation.

| Model | Input | Output | Cache read | Cache write |
|---|---:|---:|---:|---:|
| GPT-5.6 Sol | 5.00 | 30.00 | 0.50 | 6.25 |
| GLM 5.2 | 0.50 | 3.15 | 0.10 | Not separately advertised |

The final report must separately account for routing-only calls, serving calls,
token/cost coverage, decision source, confidence, selected tier, latency, and
cache metrics. It must not describe catalog-derived totals as provider invoice
spend.

## Admission record

Signal-only admission evidence is under
`/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-sol56-glm52-stage-ef05-signal-c24`.
Its all-89 no-token admission and offline Hermes→Relay→Switchyard smoke passed.

Classifier-enabled admission evidence is under
`/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-sol56-glm52-stage-ef05-classifier-c24`.
Its all-89 no-token admission and classifier smoke passed; the latter makes two
judge calls, selects GLM for one completion, and selects Sol for the forced
strong completion.

The first signal-only run starts with a fresh 89-task setup admission because
the historical setup evidence was bound to a different hermetic runtime digest.
Once its full setup evidence passes, the classifier arm may reuse that evidence
only if its dataset, Hermes runtime, Relay wheel, Switchyard library, Harbor
version, and setup parameters all match exactly.

