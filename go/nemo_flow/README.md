<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Flow Go Binding

The Go binding exposes NeMo Flow runtime APIs through CGo and the raw
`nemo-flow-ffi` library. Use it when a Go application or integration needs the
same scope, middleware, lifecycle event, and observability model used by the
Rust runtime.

This binding is experimental and source-first. Rust, Python, and Node.js are the
primary supported surfaces.

## Why Use It?

- 🧭 **Use NeMo Flow from Go**: Group agent, tool, and LLM work into the same
  scope and lifecycle model as the Rust runtime.
- 🔌 **Bridge through CGo and FFI**: Consume the shared runtime through the
  repository-maintained `nemo-flow-ffi` layer.
- 📡 **Observe runtime behavior**: Register subscribers for scope, tool, LLM,
  and mark events emitted by the runtime.
- 🚧 **Evaluate an experimental binding**: Use the source-first Go surface when
  a Go integration needs NeMo Flow semantics.

## What You Get

- ✅ **Scope, tool, and LLM helpers**: Managed lifecycle APIs backed by the
  shared Rust runtime.
- ✅ **Middleware APIs**: Guardrails and intercepts for request rewriting,
  blocking, sanitization, and execution wrapping.
- ✅ **Event subscribers**: Runtime lifecycle callbacks for observability and
  diagnostics.
- ✅ **Convenience subpackages**: Short imports for scopes, tools, LLM calls,
  guardrails, intercepts, subscribers, plugins, and adaptive helpers.
- ✅ **Local source-first workflow**: Build the FFI library locally, then test or
  consume the Go module from the checkout.

## Installation

Build the FFI library from a repository checkout before using the Go binding:

```bash
git clone https://github.com/NVIDIA/NeMo-Flow.git
cd NeMo-Flow
cargo build --release -p nemo-flow-ffi
```

For a Go application that consumes a local checkout, point the module at the
checked-out binding:

```bash
go mod edit -replace github.com/NVIDIA/NeMo-Flow/go/nemo_flow=../NeMo-Flow/go/nemo_flow
go get github.com/NVIDIA/NeMo-Flow/go/nemo_flow
```

## Getting Started

Run the binding tests from the repository checkout to verify the CGo link path
and the FFI library:

```bash
cd go/nemo_flow
go test ./...
```

Then import the package from application code:

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	nemo "github.com/NVIDIA/NeMo-Flow/go/nemo_flow"
	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow/scope"
	"github.com/NVIDIA/NeMo-Flow/go/nemo_flow/tools"
)

func main() {
	if err := nemo.RegisterSubscriber("printer", func(event nemo.Event) {
		fmt.Printf("%s %s\n", event.Kind(), event.Name())
	}); err != nil {
		log.Fatal(err)
	}
	defer nemo.DeregisterSubscriber("printer")

	handle, err := scope.Push("demo-agent", nemo.ScopeTypeAgent)
	if err != nil {
		log.Fatal(err)
	}
	defer scope.Pop(handle)

	if err := scope.Event("initialized"); err != nil {
		log.Fatal(err)
	}

	result, err := tools.Execute("search", json.RawMessage(`{"query":"hello"}`), func(args json.RawMessage) (json.RawMessage, error) {
		return args, nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(result))
}
```
