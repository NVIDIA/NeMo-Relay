// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type HookReplayBackendStatus =
  | { state: "not_initialized"; reason?: string }
  | { state: "disabled"; reason?: string }
  | { state: "ready" }
  | { state: "degraded"; reason: string }
  | { state: "stopping" }
  | { state: "stopped"; reason?: string };

export type NemoFlowHealthSnapshot = {
  id: "nemo-flow";
  backend: "hooks";
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
};

export function createHealthSnapshot(params: {
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
}): NemoFlowHealthSnapshot {
  return {
    id: "nemo-flow",
    backend: "hooks",
    status: params.status,
    initializedPluginHost: params.initializedPluginHost,
  };
}
