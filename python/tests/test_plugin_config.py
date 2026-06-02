# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import JsonObject, plugin


class _RecordingPlugin:
    def __init__(self) -> None:
        self.configs: list[JsonObject] = []

    def validate(self, plugin_config: JsonObject) -> list[plugin.ConfigDiagnostic]:
        self.configs.append(plugin_config)
        return []

    def register(self, plugin_config: JsonObject, context: plugin.PluginContext) -> None:
        self.configs.append(plugin_config)


async def test_initialize_layers_code_config_over_project_plugins_toml(tmp_path, monkeypatch):
    plugin_kind = "python.layered.plugin"
    project = tmp_path / "project"
    project_config = project / ".nemo-relay"
    project_config.mkdir(parents=True)
    (project_config / "plugins.toml").write_text(
        f"""
version = 1

[[components]]
kind = "{plugin_kind}"
enabled = true

[components.config]
source = "file"

[components.config.nested]
file = true
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    recorder = _RecordingPlugin()
    plugin.register(plugin_kind, recorder)

    try:
        report = await plugin.initialize(
            {
                "components": [
                    {
                        "kind": plugin_kind,
                        "config": {
                            "source": "code",
                            "nested": {
                                "code": True,
                            },
                        },
                    }
                ]
            }
        )
    finally:
        plugin.clear()
        plugin.deregister(plugin_kind)

    assert report["diagnostics"] == []
    assert recorder.configs == [
        {"source": "code", "nested": {"file": True, "code": True}},
        {"source": "code", "nested": {"file": True, "code": True}},
    ]
