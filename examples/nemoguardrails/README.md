<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails Plugin Example

This directory contains an example Python plugin that uses the NeMo Guardrails
Python API from NeMo Relay.

It is intentionally outside the `nemo_relay` package. Applications can copy,
vendor, or package this plugin if they want to use it.

The single-file plugin implementation, runnable agent, and Guardrails config
artifacts live under `example`.

## What It Shows

- Lazy loading of the optional `nemoguardrails` dependency.
- Native NeMo Guardrails config loaded from `config_path` or `config_yaml`.
- A real `example/example_config.yml` with NeMo Guardrails self-check input and
  output rails.
- Input and output checks around non-streaming `llm.execute(...)` calls.
- Optional checks around managed `tools.execute(...)` arguments and results.
- Request and response decoding with NeMo Relay's built-in OpenAI Chat, OpenAI
  Responses, and Anthropic Messages codecs.
- A concrete example agent that exercises the plugin with a live NVIDIA
  OpenAI-compatible chat request.
- A fast live validation lane that uses a deterministic `current_time` tool and
  passthrough Guardrails config.

## Boundaries

This example keeps provider response rewriting out of the plugin. Guardrails can
rewrite LLM input because NeMo Relay request codecs support decode and encode.
If Guardrails returns modified LLM output, the example raises instead of
mutating provider-shaped responses.

The example also does not cover streaming calls or a full `generate_async`
agent-runtime integration. Tool checks use NeMo Relay tool middleware and
serialized JSON payloads.

## Use It

Install NeMo Guardrails in the environment that runs the application:

```bash
pip install nemoguardrails
```

The bundled `example_config.yml` uses NeMo Guardrails'
`nvidia_ai_endpoints` model engine. To run that config as-is, also install the
NVIDIA LangChain provider:

```bash
pip install langchain-nvidia-ai-endpoints
```

Copy `example/plugin.py` into your application, or import it from this example
directory when experimenting locally.

Register and initialize the plugin:

```python
import asyncio

import nemo_relay
import plugin as nemoguardrails_plugin


async def main() -> None:
    nemoguardrails_plugin.register()
    try:
        config = nemo_relay.plugin.PluginConfig(
            components=[
                nemo_relay.plugin.ComponentSpec(
                    kind=nemoguardrails_plugin.DEFAULT_KIND,
                    config={
                        "config_path": "./rails",
                        "codec": "openai_chat",
                    },
                )
            ]
        )
        await nemo_relay.plugin.initialize(config)
    finally:
        nemo_relay.plugin.clear()
        nemoguardrails_plugin.deregister()


asyncio.run(main())
```

## Run the Example Agent

The `example/agent_example.py` script runs a small agent-like flow: it
initializes this plugin, runs a managed `tools.execute(...)` call, and sends the
tool result through a managed `llm.execute(...)` call to NVIDIA-hosted
inference.

Run it from a checkout where NeMo Relay and NeMo Guardrails are installed. The
default lane uses a passthrough Guardrails config and the `current_time` tool.
This is the fastest live validation path because it exercises the real plugin,
real `nemoguardrails` initialization, tool execution, and LLM execution without
running model-backed self-check rails:

```bash
export NVIDIA_API_KEY="<your-key>"
python examples/nemoguardrails/example/agent_example.py
```

To run the inline self-check rails example, load `example/example_config.yml`
as inline `config_yaml`:

```bash
python examples/nemoguardrails/example/agent_example.py --guardrails-config inline
```

The config directory lane uses the bundled
`examples/nemoguardrails/example/rails/config.yml` by default. It
contains the same input and output self-check rails as `example/example_config.yml`:

```bash
python examples/nemoguardrails/example/agent_example.py --guardrails-config path
```

Use `--tool weather` when you want the example to use the weather tool instead
of the default `current_time` tool:

```bash
python examples/nemoguardrails/example/agent_example.py --tool weather
```

Pass `--config-path` when you want the example agent to use your own native
NeMo Guardrails config directory:

```bash
python examples/nemoguardrails/example/agent_example.py \
  --guardrails-config path \
  --config-path ./rails
```

## Tests

The pytest suite injects fake `nemoguardrails` modules into `sys.modules`.
That lets CI verify the plugin behavior without installing the optional
NeMo Guardrails dependency.

The script also accepts `NVIDIA_MODEL`, `NVIDIA_BASE_URL`, and
`NVIDIA_CHAT_COMPLETIONS_URL` for local provider overrides. It also accepts
`NEMO_GUARDRAILS_CONFIG`, `NEMO_GUARDRAILS_CONFIG_PATH`, and
`NEMO_GUARDRAILS_TOOL` as environment variable equivalents for the config lane,
config path, and tool selection.

See [NeMo Guardrails Example Plugin](../../docs/build-plugins/nemoguardrails.md)
for the full configuration and limitation notes.
