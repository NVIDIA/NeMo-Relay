# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.adk.agents import LlmAgent
from google.adk.apps import App

import nemo_relay
from nemo_relay.integrations.google_adk import NemoRelayPlugin

# Print Relay events in the terminal while you try the agent.
nemo_relay.subscribers.register("console", lambda event: print(event))


def get_weather(location: str) -> dict[str, str]:
    """Return sample weather data for a location."""
    return {"location": location, "forecast": "sunny, 22 degrees Celsius"}


root_agent = LlmAgent(name="weather_agent", model="gemini-3.5-flash", tools=[get_weather])
app = App(name="weather_agent", root_agent=root_agent, plugins=[NemoRelayPlugin()])
