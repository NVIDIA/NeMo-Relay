// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

func main() {
	handle, err := nemo_flow.PushScope("example-agent", nemo_flow.ScopeTypeAgent)
	if err != nil {
		panic(err)
	}
	defer nemo_flow.PopScope(handle)

	result, err := nemo_flow.ToolCallExecute(
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
