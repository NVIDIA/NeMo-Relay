<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Model Pricing

Model pricing catalogs, source chains, cost estimation, and the built-in pricing
plugin. Import these APIs from `nemo_relay_model_pricing`.

The `plugin::PricingPlugin` implements the shared `nemo_relay_plugin::component::Plugin`
contract. The core runtime registers the component and retains host lifecycle
and rollback ownership.

This crate participates in the unified NeMo Relay release. Publish it after
`nemo-relay-types` and `nemo-relay-plugin`, before observability and core.
