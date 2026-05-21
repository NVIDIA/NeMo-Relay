---
name: nemo-flow-use-optimizer-hints
description: Deprecated compatibility alias for nemo-flow-tune-adaptive-hints; retained through v0.3 for users who still ask for optimizer hints or predictions
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Deprecated Compatibility Alias For `nemo-flow-tune-adaptive-hints`

This legacy skill name is retained through v0.3 for compatibility. NeMo Flow
now describes optimizer hints as adaptive hints, adaptive guidance, or tuning
outputs.

If this alias is selected, consume current adaptive hints and guidance rather
than looking for a separate optimizer-hints API.

## Compatibility Workflow

1. Confirm the adaptive plugin is already configured and validated.
2. Identify where hints are injected or surfaced in the chosen binding.
3. Treat adaptive hints as advisory unless the consuming API defines stronger
   semantics.
4. Test behavior when no prediction or hint is available.
5. Compare application output against the baseline after enabling hints.
6. Escalate from hints to scheduling only after idempotency and race behavior
   are understood.

## Current Semantics

- Adaptive hints are request-intercept behavior.
- The default request-body path is `nvext.agent_hints`.
- `set_latency_sensitivity(...)` is request-local guidance, not persistent
  adaptive configuration.
- `NEMO_FLOW_ACG_DEBUG` is diagnostics-only.

If the user says "optimizer hints," translate that to adaptive hints and
continue with `nemo-flow-tune-adaptive-hints`.
