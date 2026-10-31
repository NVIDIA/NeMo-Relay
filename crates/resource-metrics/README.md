<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Resource Metrics

System collectors, measurement configuration, sampling state, polling lifecycle,
and the built-in resource metrics plugin. Use `collect()` for on-demand snapshots.

The `plugin::ResourceMetricsPlugin` implements the shared `Plugin` contract
and tracks runtime cleanup through `PluginRegistrationContext`.
`nemo-relay-plugin-host` supplies a snapshot callback for scope event delivery;
this crate owns collection and polling without depending on the runtime or SDK.

This crate participates in the unified NeMo Relay release. Publish it after
`nemo-relay-types` and `nemo-relay-plugin`, before `nemo-relay-plugin-host`.
