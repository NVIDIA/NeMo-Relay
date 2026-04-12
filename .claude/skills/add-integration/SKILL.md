<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
description: Add a new third-party framework integration maintained as a NeMo Flow patch set
---

# Add a Framework Integration

NeMo Flow integrations with upstream projects are maintained as git submodules
under `third_party/` with corresponding patch files in `patches/`.

Use this skill for a new framework integration. If the upstream checkout already
exists and you are refreshing an existing patch set, use
`maintain-integration-patches` instead.

## Required Patterns

- `nemo_flow` stays an optional dependency
- framework behavior must fall back cleanly when NeMo Flow is unavailable
- tool calls and LLM calls should use NeMo Flow managed execution where possible
- scope creation should mirror the framework's natural agent, graph, or function
  boundaries
- scope stack propagation must be explicit across worker threads or async
  boundaries

See `docs/integration-best-practices.md` for the full guide and reference
patterns.

## Workflow

1. Bootstrap or refresh the local upstream checkout:

```bash
./scripts/bootstrap-third-party.sh
```

2. Implement the integration inside `third_party/<name>/`.

3. Validate patch applicability against clean upstream HEAD:

```bash
./scripts/apply-patches.sh --check
```

4. Regenerate the patch artifact:

```bash
./scripts/generate-patches.sh
```

5. Re-run patch validation and the relevant integration tests.

## Expected Outputs

```
third_party/<name>/     # local upstream checkout pinned by third_party/sources.lock
patches/<name>/         # tracked NeMo Flow integration patch set
  0001-add-nemo-flow-integration.patch
```

## Checklist

- [ ] Upstream checkout exists under `third_party/`
- [ ] Optional import / activation guard is in place
- [ ] Tool calls are wrapped through NeMo Flow where appropriate
- [ ] LLM calls are wrapped through NeMo Flow where appropriate
- [ ] Scope boundaries match the framework's execution model
- [ ] Context propagation is correct across async or thread boundaries
- [ ] Integration patch regenerates cleanly into `patches/<name>/`
- [ ] `./scripts/apply-patches.sh --check` passes
- [ ] Relevant tests or smoke coverage exist for the integration path
- [ ] Integration docs or notes are updated when user behavior changed

## Key References

- Integration guide: `docs/integration-best-practices.md`
- Patch apply helper: `scripts/apply-patches.sh`
- Patch generation helper: `scripts/generate-patches.sh`
- Third-party bootstrap helper: `scripts/bootstrap-third-party.sh`
