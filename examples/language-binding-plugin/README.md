<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Binding Plugin

These Rust, Python, and Node.js hosts implement the same application-owned
`documentation-plugin`. Every test owns one behavior and can run by itself;
setup and teardown do not depend on another test having run first.

Run each project from its own directory:

```bash
(cd rust && cargo test)
(cd python && uv run --locked --group test pytest)
(cd node && npm test)
```

The test names separate validation, activation, tool and model policies, request
rewrites, streaming, subscription, and teardown. The `main` program in each
directory is the end-to-end demonstration, while its atomic tests identify the
exact contract that failed.

Run the commands and compare their expected output in the
[Runnable Examples guide](https://docs.nvidia.com/nemo/relay/build-plugins/language-binding/code-examples).
