<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nexus Examples

This directory contains runnable examples demonstrating NeMo Agent Toolkit Nexus features. Each example is self-contained and includes setup instructions in its docstring.

## Examples

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| [agent_with_logging.py](agent_with_logging.py) | A LangChain ReAct agent using ChatNVIDIA with Nexus event logging and ATIF trajectory export. Demonstrates registering an event subscriber for console logging, setting up an `AtifExporter`, running an agent inside a Nexus scope, and exporting the trajectory to a JSON file. | Python >= 3.11, `uv sync` (installs all dependencies), `NVIDIA_API_KEY` environment variable |

## Running Examples

All Python examples use [uv](https://docs.astral.sh/uv/) for dependency management. From the repository root:

```bash
# Install dependencies (if not already done)
uv sync

# Run an example
cd examples
uv run python agent_with_logging.py
```

## Adding New Examples

When adding a new example:

1. Include the SPDX license header at the top of the file.
2. Add a module-level docstring describing what the example demonstrates, its prerequisites, and how to run it.
3. Update the table in this README with the new example.
4. Keep examples focused on a single feature or integration pattern.
