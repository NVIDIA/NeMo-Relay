# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import nemo_relay
from nemo_relay.integrations.google_adk import NemoRelayPlugin


def get_weather(location: str) -> dict[str, str]:
    """Return sample weather data for a location."""
    return {"location": location, "forecast": "sunny, 22 degrees Celsius"}


root_agent = LlmAgent(
    name="weather_agent",
    model="gemini-3.5-flash",
    instruction="Use tools when they are relevant. Keep the final answer brief.",
    tools=[get_weather],
)

app = App(name="weather_app", root_agent=root_agent, plugins=[NemoRelayPlugin()])


async def main(application: App = app) -> None:
    async with nemo_relay.plugin.activate({}, "plugins.toml"):
        try:
            async with Runner(app=application, session_service=InMemorySessionService()) as runner:
                session = await runner.session_service.create_session(app_name=application.name, user_id="user-1")
                message = types.UserContent("What is the weather in San Francisco?")
                with nemo_relay.scope.scope("adk-request", nemo_relay.ScopeType.Agent):
                    async for event in runner.run_async(
                        user_id=session.user_id, session_id=session.id, new_message=message
                    ):
                        if event.is_final_response() and event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.text and not part.thought:
                                    print(part.text)
        finally:
            await nemo_relay.subscribers.flush_async()


if __name__ == "__main__":
    asyncio.run(main())
