// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

func main() {
	handle, err := nat_nexus.PushScope("example-agent", nat_nexus.ScopeTypeAgent)
	if err != nil {
		panic(err)
	}
	defer nat_nexus.PopScope(handle)

	result, err := nat_nexus.ToolCallExecute(
		"search",
		json.RawMessage(`{"query":"hello"}`),
		func(args json.RawMessage) (json.RawMessage, error) {
			return json.Marshal(map[string]any{
				"results": []string{"echo:hello"},
			})
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(result))
}
