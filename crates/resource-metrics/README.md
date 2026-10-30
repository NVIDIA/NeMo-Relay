<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Resource Metrics

System collectors, measurement configuration, sampling state, polling lifecycle,
and the built-in resource metrics plugin. Use `collect()` for on-demand snapshots.

The `plugin::ResourceMetricsPlugin` requests a snapshot callback through
`ResourceMetricsRegistrar`. The core runtime supplies scope event delivery;
this crate owns collection and polling without depending on core.

This crate participates in the unified NeMo Relay release. Publish it after
`nemo-relay-types` and `nemo-relay-plugin`, before core.
