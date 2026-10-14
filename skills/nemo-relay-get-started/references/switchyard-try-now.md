<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard Trial Routing

Read the current [Switchyard quick start](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/switchyard)
and its referenced files. Confirm that the guide supports the installed Relay
release and selected plugin bundle.

Use `nemo-relay-install` only for missing CLI or bundle dependencies.
Resolve their stable releases and compatibility independently; do not keep
release pins, archive names, or platform-minimum tables in this reference.

Follow the guide for integrity checks, manifest/library layout, provider and
routing configuration, attestation requirements, gateway lifecycle, request,
and verification. Preserve provider credentials and do not weaken policy
outside the selected plugin's documented requirements.

Use an isolated trial and avoid colliding with an existing service. Derive
the actual response and routing evidence from the guide; do not substitute a
generic tool/agent proof or claim that a successful provider response alone
shows that routing occurred.

Apply [Guide-Based First-Value Checks](first-value.md). Report what the selected
trial demonstrates without extrapolating routing quality or performance.
