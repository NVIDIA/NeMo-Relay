<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started: Go

This guide takes you from the FFI build step to a minimal scope, tool call,
and LLM call using the Go binding.

All examples in this guide use:

- an active Nexus scope
- the managed execution APIs (`ToolCallExecute(...)` and `LlmCallExecute(...)`)

This guide intentionally does not use the low-level manual lifecycle APIs.

## Prerequisites

- Go 1.21+
- Rust toolchain

## Build the FFI Layer

From the repository root:

```bash
cargo build --release -p nvidia-nat-nexus-ffi
```

## Minimal Scope and Tool Execution

This example uses the module path defined by `go/nat_nexus/go.mod`.

```go
package main

import (
	"encoding/json"
	"fmt"

	"gitlab-master.nvidia.com/nemo-agent-toolkit/dev/Project-NAT-Nexus/go/nat_nexus"
)

func main() {
	handle, err := nat_nexus.PushScope("quickstart-agent", nat_nexus.ScopeTypeAgent)
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
```

Run it with the FFI library on the loader path:

```bash
cd go/nat_nexus
CGO_LDFLAGS="-L../../target/release" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../target/release" \
go run ./...
```

## Minimal LLM Execution

```go
request := map[string]any{
	"headers": map[string]any{},
	"content": map[string]any{
		"model": "gpt-4",
		"messages": []map[string]any{
			{"role": "user", "content": "Hello"},
		},
	},
}

response, err := nat_nexus.LlmCallExecute(
	"gpt-4",
	request,
	func(_ json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]any{"response": "ok"})
	},
	nat_nexus.WithLLMModelName("gpt-4"),
)
```

## Common Errors

- Dynamic library not found
  Rebuild the FFI crate and make sure `LD_LIBRARY_PATH` includes
  `target/release`.
- Concurrent server integration confusion
  Read [Context Isolation](context-isolation.md) before integrating with
  goroutines or worker pools.

## Next Docs

- [Language Bindings](language-bindings.md#go)
- [Context Isolation](context-isolation.md)
- [API Reference](api-reference.md)
- [Testing](testing.md)
