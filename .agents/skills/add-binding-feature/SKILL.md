---
name: add-binding-feature
description: Add or change a public NeMo Relay runtime API across Rust and affected language bindings when no more specific maintainer skill owns the domain. Do not use for middleware additions, internal refactors, binding-local fixes, or docs-only changes.
license: Apache-2.0
---


# Add a Binding Feature

Use this skill when a change affects the public runtime surface and must stay in
parity across the Rust core, FFI, and one or more bindings.

Do not use this skill for:

- Internal-only core refactors with no public API change
- New middleware contracts, which use the middleware-specific workflow
- Binding-local bug fixes that do not change shared behavior
- Docs-only or example-only updates

## Implementation Order

1. **Core Rust**
   Implement the behavior first in `crates/core/src/api/` and
   related core modules such as `crates/core/src/api/runtime/`,
   `crates/core/src/codec/`, or `crates/core/src/json.rs`.
2. **FFI / shared C surface, when affected**
   Update `crates/ffi` and its generated header only when the capability is
   exposed through the C ABI or Go binding.
3. **Language-native bindings**
   Update each binding that exposes the capability; leave unrelated bindings
   unchanged.
4. **Language wrapper helpers**
   Update Python wrapper modules, Go shorthand packages, typed helpers, or
   adaptive/plugin helpers if the new behavior belongs there.
5. **Docs and examples**
   Update reference docs, language-binding docs, and examples when the public
   surface or expected usage changed.
6. **Validation**
   Follow the repository validation policy for the surfaces whose public or
   observable behavior changed.

## Naming Conventions

| Layer       | Convention        | Example                              |
|-------------|-------------------|--------------------------------------|
| Rust        | `snake_case`      | `nemo_relay_tool_call`                |
| C FFI       | `nemo_relay_` prefix | `nemo_relay_tool_call`              |
| Python      | `snake_case`      | `nemo_relay.tools.call`               |
| Go          | `PascalCase`      | `nemo_relay.ToolCall`                 |
| Node.js     | `camelCase`       | `toolCall`                           |

## Parity Checklist

- [ ] Core function with doc comment in `crates/core/src/api/`
- [ ] Runtime callback/state, codec, JSON, or event/tool/LLM/scope types added
      in the relevant core module if needed
- [ ] FFI wrapper and generated header updated if the C ABI changes
- [ ] Python native binding, wrapper, docstring, and stubs updated if exposed
- [ ] Go wrapper and shorthand package updated if the experimental Go surface
      exposes the capability
- [ ] Node.js native binding and wrapper updated if exposed
- [ ] Typed wrapper or adaptive/plugin helper surfaces updated when applicable
- [ ] Meaningful tests added in every affected language surface
- [ ] SPDX license header on any new files
- [ ] Relevant pages under `docs/reference/` updated
- [ ] `README.md`, `docs/getting-started/`, or binding-level READMEs updated if behavior differs by language
- [ ] Relevant getting-started, README, or example docs updated if usage changed

## Decision Points

Lock these before implementing:

- Which bindings actually expose the new surface?
- Is the change part of the plain JSON API, typed wrappers, adaptive/plugin
  helpers, or observability helpers?
- Does the new API need manual lifecycle and managed execute variants, or only
  one of them?
- Does the new behavior change event fields, metadata, or scope expectations?
- If tool execution is affected, does every callback, continuation, managed
  return, and manual end surface use the canonical `ToolExecutionResult`
  contract and preserve its opaque annotation?
- Are docs/examples required because the intended usage changed?

## Key References

- Architecture: `docs/about-nemo-relay/architecture.mdx`
- Reference index: `docs/reference/api/index.mdx`
- Getting started and binding status: `README.md`,
  `docs/getting-started/quick-start/index.mdx`,
  `docs/about-nemo-relay/support-matrix.mdx`
- Typed wrappers and codecs: `docs/integrate-into-frameworks/using-codecs.mdx`,
  `docs/integrate-into-frameworks/provider-codecs.mdx`
- Adaptive config/plugins: `docs/configure-plugins/about.mdx`,
  `docs/build-plugins/about.mdx`,
  `docs/configure-plugins/adaptive/configuration.mdx`
- Existing pattern: follow a surface already implemented across core, FFI,
  Python, Go, and Node.js rather than inventing a new shape
