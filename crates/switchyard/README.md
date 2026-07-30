<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Switchyard Plugin

`nemo-relay-switchyard` is NeMo Relay's experimental in-process integration
with [NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard).
It runs libsy's random router inside Relay through `Algorithm::run_stream`.

Relay owns provider target bindings, credentials, transport, retries, fallback,
and observability. libsy emits routing decisions and requests provider calls;
Relay fulfills each `CallLlm` and returns the real response or error through
`CallLlmRequest::respond`. No Switchyard server or Switchyard LLM client is
used.

## Translation Boundary

`switchyard-translation` is the plugin's only provider wire-format translation
engine:

1. decode the caller request into the Switchyard neutral request;
2. encode each routed call for the selected target protocol;
3. decode the provider response or stream before returning it to libsy;
4. encode `ReturnToAgent` into the caller protocol.

Same-protocol buffered bodies and stream events preserve their original
provider JSON. Cross-protocol routes carry only data represented by the target
protocol and fail when Switchyard reports a lossy conversion. The plugin does
not use NeMo Relay codecs as a translation fallback.

## Configuration

```toml
version = 1

[[components]]
kind = "switchyard"
enabled = true

[components.config]
version = 2
priority = 0
max_retries = 3
enabled_inbound_profiles = ["openai_chat"]

[components.config.algorithm]
kind = "random"
seed = 42

[components.config.default_targets]
openai_chat = "fast"

[components.config.targets.fast]
model = "provider/model"
protocol = "openai_chat"
endpoint = "/v1/chat/completions"
base_url = "https://provider.example.com"
weight = 1

[components.config.targets.fast.header_env]
authorization = "PROVIDER_AUTHORIZATION"
```

Target-map keys are libsy semantic names. Relay configuration remains
authoritative for the physical model, protocol, endpoint, URL, and credentials.
Version-1 Decision API configuration is rejected with a migration error.

Build and test the optional CLI integration with:

```bash
cargo build -p nemo-relay-cli --features switchyard
cargo test -p nemo-relay-switchyard
```

See [`examples/switchyard`](../../examples/switchyard) for a no-service,
fake-provider smoke test.
