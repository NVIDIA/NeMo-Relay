<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Native Plugin ABI

`nemo-relay-plugin-abi` defines the application binary interface (ABI) between a
native plugin and its Relay host. Use `nemo-relay-plugin` to write plugins. Use
this crate when implementing or testing the library boundary.

The interface uses `abi_stable` 0.11.3 to check type layouts and `async-ffi` 0.5.1
to pass futures and wakers across libraries. Relay runtime types, Tokio objects,
and `serde_json::Value` stay outside this crate.

## Version Rules

Compatibility uses these version rules:

| Version | Purpose |
|---|---|
| Relay release | The plugin's `compat.relay` range selects supported runtime releases. |
| Native contract, such as `v1.0` | The build SDK sets the minimum host interface required by the plugin. |
| ABI dependency versions | The loader checks shared type layouts, including dependency ABI generations. |

Contract names use dots: `v1.0`, `v1.1`, and `v2.0`. Rust module names use
underscores, such as `v1_0`, because Rust identifiers cannot contain dots.

A plugin built with SDK contract v1.1 requires a host that supports v1.1 or a
later minor in major 1. A host that supports v1.1 also supports v1.0 plugins.
Each minor keeps its published interfaces and behavior. A breaking layout or dependency ABI change
requires a new native major.

The interface crate follows Relay's package release version. That version does
not select the native contract or determine binary compatibility. The explicit
contract in `src/contract.rs` is the source of truth. The build script derives the
SDK descriptor constant and `abi_stable` layout metadata from that contract.
Frozen bootstrap and module headers keep their published contract versions.

## Bootstrap and Typed Values

The bootstrap is the library entry interface. Its layout stays fixed throughout
a native major. It provides the plugin descriptor, module discovery, and factory
access. Descriptor lookup and module discovery do not create components, register
callbacks, or start executors.

Use `LibHeader`'s checked initialization before calling a factory. Select the
plugin's exact contract. Add interfaces through versioned modules. Published
records, enum variants, signatures, marker traits, and ownership rules stay fixed
within each contract.

The recursive `Json` type carries application values without converting them to
JSON text. SDK conversions preserve signed and unsigned integers, finite floating-point
values, Unicode, embedded NULs, nested values, and negative zero. They reject
non-finite numbers and duplicate object keys.

## Loading and Component Cleanup

Native plugins run in the host process and require full trust. Before initializing
a library, the host checks Relay compatibility, loading policy, signatures, and
integrity. The checked loader keeps a private copy of the verified library and
its dependencies. It reads binary imports and runtime search paths to find
package-local libraries. The operating system supplies system libraries.
Package-local dependencies must use loader-relative imports and search paths.
The loader rejects absolute package paths and sidecars whose library names
conflict with different bytes already resident in the process. Use unique
library names for different sidecar implementations. Relay tracks names for
every resident image, including main plugin libraries that fail compatibility
checks. Captured dependencies cannot reuse those names with different bytes.
Resident library identity includes the captured paths and bytes.
Each activation also checks the descriptor against the manifest.

Libraries and their staged dependencies stay loaded until process exit. Each
activation creates fresh component state. Closing a component removes its
registrations, revokes runtime access, cancels work, waits for tasks to finish,
and releases its executor resources. A failed close keeps ownership so the caller
can retry.

The SDK polls user futures inline by default and creates no Tokio runtime.
Components can opt into a shared runtime within their plugin library. Closing
the last component that uses it shuts it down on a separate thread. Concurrent
close calls share that work. If the teardown thread cannot start, the component
keeps runtime ownership. Dropping a foreign future cancels its work, including
when the future has not been polled.

## Dependency Pins

Most workspace dependencies use Cargo SemVer ranges. The native interface uses
exact `=version` requirements for `nemo-relay-plugin-abi`, `abi_stable`, and
`async-ffi`. The interface pin selects the declarations reviewed with that SDK
release. Binary compatibility follows the native contract and checked layouts.
The external dependency pins keep shared types on reviewed ABI generations.
A dependency update must pass the binary compatibility tests.

Fixture workspaces have their own exact dependency pins and lockfiles.
`[package] version = "1.0.0"` names a package release, so it does not use an `=`
prefix inside the value.

## Verify the Interface

Run these commands from the repository root:

```bash
just build-native-abi-test-fixtures
cargo test -p nemo-relay-plugin-abi
cargo test -p nemo-relay-plugin --test abi_value --test abi_executor
just build-test-plugin-fixtures
cargo test -p nemo-relay-plugin-host --test native_checked_loader
```

`just test-rust` also prepares the fixtures. Each fixture builds in a separate
Cargo workspace and target directory. This prevents workspace features from
changing the binary being tested. Keep the frozen interface copies fixed so they
can detect compatibility regressions.

The tests load a v1.0 plugin with v1.0 and v1.1 hosts. They also check minimum
minor versions, major versions, changed signatures, changed `Sync` requirements,
missing modules, descriptor mismatches, and multiple libraries. Hosts and plugins
also load unchanged v1.0 binaries built with different Cargo package major
versions and prereleases.

To refresh fixture lockfiles, run the owning workflow:

```bash
just generate-native-abi-test-lockfiles
```

## Release the Interface

The interface publishes first in the normal Relay release workflow, using the
workspace release version. Relay version-update recipes update its package
version and exact SDK dependency pin together. They leave the native contract
and frozen compatibility fixtures unchanged.

Run `just native-abi-contract` to check the contract and package metadata. Refer
to [Releasing NeMo Relay](../../RELEASING.md) for the maintainer procedure.
