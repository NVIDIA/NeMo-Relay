# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from google.adk.apps import App
from google.adk.tools import FunctionTool
from google.adk.workflow import Workflow
from weather import get_weather, main

from nemo_relay.integrations.google_adk import NemoRelayPlugin, wrap_node


def pick_city(node_input: object) -> dict[str, str]:
    return {"location": "San Francisco"}


workflow = wrap_node(Workflow(name="weather_pipeline", edges=[("START", pick_city, FunctionTool(get_weather))]))
app = App(name="weather_app", root_agent=workflow, plugins=[NemoRelayPlugin()])


if __name__ == "__main__":
    asyncio.run(main(app))
