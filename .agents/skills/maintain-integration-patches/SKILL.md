---
name: maintain-integration-patches
description: Refresh, rebase, regenerate, and validate existing NeMo Relay third-party integration patches
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain Integration Patches

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when an existing `patches/<name>/0001-add-nemo-relay-integration.patch`
has drifted against the pinned upstream checkout or needs regeneration after
local changes.

## Workflow

1. Bootstrap the upstream checkout from `third_party/sources.lock`:

```bash
./scripts/bootstrap-third-party.sh
```

This root command is the stable public wrapper. The implementation lives under
`scripts/third-party/bootstrap.sh`.

2. Inspect whether the local checkout is dirty before doing anything else.

3. Validate current applicability against clean upstream HEAD:

```bash
./scripts/apply-patches.sh --check
```

4. Make or reconcile the local integration changes under `third_party/<name>/`.

5. Regenerate the patch:

```bash
./scripts/generate-patches.sh
```

6. Re-run `./scripts/apply-patches.sh --check`.

7. Run the integration-specific tests or smoke path.

## Rules

- Do not apply patches on top of a dirty upstream checkout unless you explicitly
  understand and intend the merge state.
- Prefer the repo patch scripts over ad hoc `git diff > patch` workflows.
- Keep the patch minimal and focused on the NeMo Relay integration surface.
- If upstream drift changes behavior, update docs or test expectations in the
  same branch.

## Checklist

- [ ] Checkout bootstrapped from `third_party/sources.lock`
- [ ] Dirty-state situation understood before regeneration
- [ ] `./scripts/apply-patches.sh --check` passes after regeneration
- [ ] Patch artifact updated in `patches/<name>/`
- [ ] Relevant integration tests or smoke coverage pass
- [ ] No stale repo names, package names, or symbol prefixes remain in the patch

## References

- Patch apply helper: `scripts/apply-patches.sh`
- Patch generation helper: `scripts/generate-patches.sh`
- Third-party bootstrap helper: `scripts/bootstrap-third-party.sh`
- Internal implementations: `scripts/third-party/`
- Integration guide: `docs/integrate-frameworks/about.md`
- Injection points: `docs/integrate-frameworks/adding-scopes.md`
