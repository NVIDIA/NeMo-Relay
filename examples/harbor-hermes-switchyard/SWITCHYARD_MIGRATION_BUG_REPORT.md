<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bug report: findings from the Switchyard-native plugin migration

Two bugs were found while migrating this example's plugin configuration to the
Switchyard-native NeMo Relay plugin (Switchyard PR #528) and consolidating
`config/plugins.toml.in` into a single shared template with per-experiment
`config/switchyard/<name>.toml` and `config/pricing/<name>.json` files.

## Bug 1: JSON pricing-catalog file source rejects unknown fields inconsistently with the inline source

**Component:** `crates/core/src/codec/model_pricing.rs` (`PricingCatalog`,
`PricingSourceConfig::File` vs. `PricingSourceConfig::Inline`)

**Severity:** Low/medium — silently breaks any JSON pricing catalog file that
carries extra top-level metadata fields, even though the equivalent inline TOML
catalog source tolerates the same kind of extra fields.

**Summary:** Relay's pricing component supports two source types that both
deserialize into the same `PricingCatalog { version, entries }` struct:
`type = "inline"` (an embedded TOML table) and `type = "file"` (an external JSON
file). Code inspection of `PricingCatalog` shows no `#[serde(deny_unknown_fields)]`
attribute at the struct level. Empirically, however, a `type = "file"` source
rejects any additional top-level JSON key beyond `version`/`entries` at plugin
activation time. We did not confirm the exact mechanism producing the stricter
behavior for the file path only.

**Reproduction:**
1. Configure a Relay pricing component with a `type = "file"` source pointing at
   a JSON catalog (`{"version": 1, "entries": [...]}`).
2. Add any extra top-level key to that JSON file (e.g. a copyright/license
   attribution field).
3. Activate the dynamic plugin that loads this pricing config (or start a Relay
   host that loads it).

**Observed error:**
```
invalid config: invalid model pricing config: invalid model pricing catalog JSON: unknown field `spdx_copyright`, expected `version` or `entries` at line N column M
```

**Expected:** Either both source types should reject unknown fields, or both
should tolerate them — the inconsistency between two code paths deserializing
the same target struct is surprising and, as far as we found, undocumented.

**Workaround applied here:** Removed the extra metadata keys from
`config/pricing/default.json` in this example.

**Suggested fix:** Align the `File` source's JSON deserialization with the same
permissiveness as the `Inline` source's TOML deserialization, or explicitly add
`#[serde(deny_unknown_fields)]` to `PricingCatalog` and document the strict
contract for both source types.

## Bug 2 (fixed in this change): `plugin_config_template_sha256` loses experiment-distinguishing power once multiple cohorts share one template file

**Component:** `examples/harbor-hermes-switchyard/scripts/prepare_runtime.py`,
`examples/harbor-hermes-switchyard/scripts/run_phase2_cohort.py`

**Severity:** Medium — could have caused silent reuse of stale or wrong cached
setup evidence across different experiment cohorts.

**Summary:** Before this migration, each experiment cohort had its own
self-contained `plugins.<name>.toml.in` file, so hashing that one file
(`plugin_config_template_sha256`) reliably detected "did this experiment's
configuration change" for the cache-reuse/change-detection logic in
`run_phase2_cohort.py` (`validate_smoke_evidence`, `validate_offline_evidence`,
and setup-admission provenance).

After consolidating to one shared `plugins.toml.in` template referenced by every
switchyard-routed experiment — with per-experiment content moved to
`config/switchyard/<name>.toml` and `config/pricing/<name>.json` — hashing only
the shared template no longer distinguishes between experiments: two different
cohorts with different models or routing algorithms would compute an identical
`plugin_config_template_sha256`, since the file they both reference is
byte-identical.

**Impact if left unfixed:** Reuse-eligibility checks comparing a stored hash
against a freshly computed one would pass even when the actual experiment
configuration differed, risking silent reuse of setup/smoke evidence from a
different cohort.

**Fix applied:** Added `plugin_config_identity_sha256()` in the new shared
`scripts/plugin_config_paths.py` module. It hashes the template together with
its derived `switchyard/<name>.toml` and `pricing/<name>.json` sibling files,
falling back to just the template hash for direct-baseline configs (which have
no siblings). Every identity/reuse-eligibility call site in `prepare_runtime.py`
and `run_phase2_cohort.py` now uses this combined hash instead of hashing the
template file alone.
