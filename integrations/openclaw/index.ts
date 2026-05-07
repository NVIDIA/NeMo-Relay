// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  definePluginEntry,
  type OpenClawPluginApi,
} from "openclaw/plugin-sdk/plugin-entry";

import { nemoFlowConfigSchema } from "./src/config.js";
import { registerNemoFlowPlugin } from "./src/runtime-state.js";

export default definePluginEntry({
  id: "nemo-flow",
  name: "NeMo Flow Observability",
  description: "ATIF, OpenInference, and OpenTelemetry telemetry through NeMo Flow",
  configSchema: nemoFlowConfigSchema,
  register(api: OpenClawPluginApi) {
    registerNemoFlowPlugin(api);
  },
});
