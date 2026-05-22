---
name: add-integration
description: Add a new third-party framework integration maintained as a NeMo Relay patch set
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Add a Framework Integration

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

NeMo Relay integrations with upstream projects are maintained as manifest-pinned
local upstream checkouts under `third_party/`, bootstrapped from
`third_party/sources.lock`, with corresponding patch files in `patches/`.

Use this skill for a new framework integration. If the upstream checkout already
exists and you are refreshing an existing patch set, use
`maintain-integration-patches` instead.

## Required Patterns

- `nemo_relay` stays an optional dependency
- Framework behavior must fall back cleanly when NeMo Relay is unavailable
- Tool calls and LLM calls should use NeMo Relay managed execution where possible
- Scope creation should mirror the framework's natural agent, graph, or function
  boundaries
- Scope stack propagation must be explicit across worker threads or async
  boundaries

See `docs/integrate-frameworks/about.md`,
`docs/integrate-frameworks/adding-scopes.md`, and
`docs/integrate-frameworks/non-serializable-data.md` for the current guide and
reference patterns.

## Workflow

1. Bootstrap or refresh the local upstream checkout:

```bash
./scripts/bootstrap-third-party.sh
```

This root command is the stable public wrapper. The implementation lives under
`scripts/third-party/bootstrap.sh`.

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
patches/<name>/         # tracked NeMo Relay integration patch set
  0001-add-nemo-relay-integration.patch
```

## Checklist

- [ ] Upstream checkout exists under `third_party/`
- [ ] Optional import / activation guard is in place
- [ ] Tool calls are wrapped through NeMo Relay where appropriate
- [ ] LLM calls are wrapped through NeMo Relay where appropriate
- [ ] Scope boundaries match the framework's execution model
- [ ] Context propagation is correct across async or thread boundaries
- [ ] Integration patch regenerates cleanly into `patches/<name>/`
- [ ] `./scripts/apply-patches.sh --check` passes
- [ ] Relevant tests or smoke coverage exist for the integration path
- [ ] Integration docs or notes are updated when user behavior changed

## Key References

- Integration guide: `docs/integrate-frameworks/about.md`
- Injection points: `docs/integrate-frameworks/adding-scopes.md`
- JSON boundary guidance:
  `docs/integrate-frameworks/non-serializable-data.md`
- Patch apply helper: `scripts/apply-patches.sh`
- Patch generation helper: `scripts/generate-patches.sh`
- Third-party bootstrap helper: `scripts/bootstrap-third-party.sh`
- Internal implementations: `scripts/third-party/`
