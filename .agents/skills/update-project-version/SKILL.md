---
name: update-project-version
description: Perform a NeMo Relay project release-version change across project-owned manifests, dependencies, lockfiles, and generated attribution surfaces while keeping just set-version coverage complete. Do not use for dependency-only updates or adding a package when no project version bump is requested.
license: Apache-2.0
---

# Update Project Version

Use this skill for project release-version changes, including prerelease and
build-metadata variants.

## Invariants

- `Cargo.toml` `[workspace.package].version` is the Rust source of truth.
- `just set-version <version>` is the single repository entrypoint for updating
  normal project-version surfaces. Do not hand-edit its existing targets.
- Every new package or plugin that participates in the unified NeMo Relay
  release, plus its project-version dependencies and lockfile entries, must be
  covered by the version automation. If a value does not derive automatically,
  update the automation in the same change that introduces it.
- Python surfaces use PEP 440 translations where required; Cargo, npm, and
  plugin manifests use the repository SemVer form.

## Workflow

Use this sequence:

1. Read the current version from `Cargo.toml` and choose the exact target.
2. Compare project-owned package/workspace members and plugin manifests with
   the helpers called by `set_project_version`, paying particular attention to
   recent or in-scope additions. A new manifest, publishable package, internal
   dependency pinned to the project version, or lockfile package entry is a
   signal that `just set-version` may need to change.
3. Run `just set-version <version>`.
4. Review the diff, then search tracked Cargo, Python, npm, lockfile, and plugin
   manifests for the exact old version. Classify matches rather than replacing
   them blindly: third-party pins and examples can legitimately match.
5. If a project-owned release surface remains on the old version, update the
   appropriate helper and rerun `just set-version <version>`. Read
   `references/version-automation.md` only when the version topology or helper
   implementation changes.
6. Refresh lockfiles or attribution files only when their versioned inputs
   changed. Do not run language suites for a metadata-only bump.
7. Follow the [documentation maintenance review](../../../RELEASING.md#documentation-maintenance)
   in `RELEASING.md`. Regenerate shared prerequisites when their declarations or
   table templates change; a package version bump alone does not require it.

## Verification

Before handoff, verify:

- Confirm every project-owned manifest version and internal dependency resolves
  to the target SemVer or its expected PEP 440 translation, and that unrelated
  versions did not change.
- Rerun `just set-version <version>` when automation changed; it should be
  idempotent and report no missing expected fields.
- Use focused checks for edited helper code and targeted packaging or generated
  output checks only for surfaces affected by the bump.

Temporary versions produced by packaging recipes are not the canonical project
version unless the release workflow explicitly requires that exact suffix.

## References

- `Cargo.toml`
- `justfile`
- `references/version-automation.md` — only for adding, removing, or debugging
  a versioned surface or version helper
