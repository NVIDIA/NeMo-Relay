<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started: Go

This guide takes you from the FFI build step to a minimal scope, tool call,
and LLM call using the Go binding.

All examples in this guide use:

- an active NeMo Flow scope
- the managed execution APIs (`ToolCallExecute(...)` and `LlmCallExecute(...)`)

This guide intentionally does not use the low-level manual lifecycle APIs.

## Prerequisites

- Go 1.21+
- Rust toolchain

## Build the FFI Layer

From the repository root:

```bash
cargo build --release -p nemo-flow-ffi
```

## Minimal Scope and Tool Execution

This example uses the module path defined by `go/nemo_flow/go.mod`.

```go
package main

import (
	"encoding/json"
	"fmt"

	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
)

func main() {
	handle, err := nemo_flow.PushScope("quickstart-agent", nemo_flow.ScopeTypeAgent)
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
```

Save the example as `main.go` outside `go/nemo_flow/`, then run it with the
FFI library on the loader path:

```bash
CGO_LDFLAGS="-L$(pwd)/target/release" \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(pwd)/target/release" \
go run ./main.go
```

On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.

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

response, err := nemo_flow.LlmCallExecute(
	"gpt-4",
	request,
	func(_ json.RawMessage) (json.RawMessage, error) {
		return json.Marshal(map[string]any{"response": "ok"})
	},
	nemo_flow.WithLLMModelName("gpt-4"),
)
```

## Common Errors

- Dynamic library not found
  Rebuild the FFI crate and make sure `LD_LIBRARY_PATH` includes
  `target/release` on Linux, or `DYLD_LIBRARY_PATH` on macOS.
- Concurrent server integration confusion
  Read [Context Isolation](context-isolation.md) before integrating with
  goroutines or worker pools.

## Next Docs

- [Language Bindings](language-bindings.md#go)
- [Observability with OpenTelemetry](observability-with-opentelemetry.md)
- [Observability with OpenInference](observability-with-openinference.md)
- [Context Isolation](context-isolation.md)
- [API Reference](api-reference.md)
- [Testing](testing.md)
