# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare production native middleware using two local source checkouts."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
from pathlib import Path


def command(args, cwd, *, capture=False):
    print("+ " + " ".join(map(str, args)), flush=True)
    return subprocess.run(
        list(map(str, args)), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE if capture else None
    ).stdout


def prepare(source, directory):
    checked = "nemo-relay-plugin-abi" in (source / "crates/plugin/Cargo.toml").read_text()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "Cargo.toml").write_text("""[workspace]
resolver = "2"
members = ["host", "plugin"]
[profile.release]
opt-level = "s"
lto = "thin"
codegen-units = 1
strip = "debuginfo"
panic = "unwind"
""")
    scripts = Path(__file__).resolve().parent
    for member in ("host", "plugin"):
        src = directory / member / "src"
        src.mkdir(parents=True, exist_ok=True)
        for original, destination in [
            (scripts / "allocation.rs", src / "allocation.rs"),
            (scripts / f"{member}.rs", src / ("main.rs" if member == "host" else "lib.rs")),
        ]:
            content = original.read_bytes()
            if not destination.exists() or destination.read_bytes() != content:
                destination.write_bytes(content)
        dependencies = {"serde_json": {"version": "1"}, "tokio": {"version": "1", "features": ["full"]}}
        if member == "host":
            dependencies |= {
                "nemo-relay-plugin-host": {
                    "path": str(source / "crates/plugin-host"),
                    "features": ["__skip-implicit-config"],
                },
                "nemo-relay-runtime": {"path": str(source / "crates/runtime")},
            }
        else:
            dependencies |= {
                "nemo-relay-plugin": {"path": str(source / "crates/plugin")},
                "futures": {"version": "0.3"},
            }
        manifest = f'[package]\nname = "native-bench-{member}"\nversion = "0.0.0"\nedition = "2024"\n'
        if member == "plugin":
            manifest += '[lib]\ncrate-type = ["cdylib"]\n[features]\nchecked = []\n'
        manifest += "[dependencies]\n" + "".join(
            name + " = { " + ", ".join(f"{key} = {json.dumps(item)}" for key, item in value.items()) + " }\n"
            for name, value in dependencies.items()
        )
        (directory / member / "Cargo.toml").write_text(manifest)
    shutil.copyfile(source / "Cargo.lock", directory / "Cargo.lock")
    command(
        ["cargo", "metadata", "--offline", "--format-version", "1"]
        + (["--features", "native-bench-plugin/checked"] if checked else []),
        directory,
        capture=True,
    )
    args = ["cargo", "build", "--release", "--locked"]
    if checked:
        args += ["--features", "native-bench-plugin/checked"]
    command(args, directory)
    prefix, suffix = ("", ".dll") if os.name == "nt" else ("lib", ".dylib" if platform.system() == "Darwin" else ".so")
    library = directory / "target/release" / f"{prefix}native_bench_plugin{suffix}"
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest = directory / "relay-plugin.toml"
    manifest.write_text(
        f'''manifest_version = 1
[plugin]
id = "bench.native"
kind = "rust_dynamic"
[compat]
relay = ">=1.0.0,<2.0"
native_api = "{"v1.0" if checked else "1"}"
[defaults]
enabled = false
[capabilities]
items = ["plugin_native"]
[integrity]
sha256 = "sha256:{digest}"
[load]
library = {json.dumps(str(library))}
'''
        + ("" if checked else 'symbol = "nemo_relay_register_plugin"\n')
    )
    return {
        "commit": command(["git", "rev-parse", "HEAD"], source, capture=True).strip(),
        "dirty": bool(command(["git", "status", "--porcelain"], source, capture=True).strip()),
        "lock_sha256": hashlib.sha256((directory / "Cargo.lock").read_bytes()).hexdigest(),
        "manifest": str(manifest),
        "binary": str(
            directory / "target/release" / ("native-bench-host.exe" if os.name == "nt" else "native-bench-host")
        ),
    }


def report(data):
    rows = {}
    expected = {
        (p, m, s)
        for p in ("request", "execute")
        for m in ("ready", "pending", "wakers", "fanout")
        for s in ("tiny", "objects", "large")
    }
    for label in ("baseline", "checked"):
        runs = data["runs"][label]
        if not runs:
            raise ValueError("Missing benchmark runs")
        for run in runs:
            keys = [(r["path"], r["mode"], r["payload"]) for r in run["results"]]
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError("Benchmark scenario set differs")
            for result, key in zip(run["results"], keys, strict=True):
                rows.setdefault(key, {}).setdefault(label, []).append(result)
    lines = [
        "# Native Middleware Measurements",
        "",
        f"Platform: {data['platform']}. Compiler: {data['compiler']}.",
        "",
        "Values are medians across process runs. Ratio is checked / baseline; values above 1 mean slower calls.",
        "",
        "| Path | Scheduling | Payload | Baseline µs | Checked µs | Ratio | Host allocations | Plugin allocations |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in sorted(rows.items()):
        base, new = values["baseline"], values["checked"]
        latency = [statistics.median(r["median_ns"] for r in group) / 1000 for group in (base, new)]
        counts = [
            f"{statistics.median(r[field] for r in base):.0f} → {statistics.median(r[field] for r in new):.0f}"
            for field in ("host_allocations", "plugin_allocations")
        ]
        lines.append(
            "| "
            + " | ".join([*key, f"{latency[0]:.2f}", f"{latency[1]:.2f}", f"{latency[1] / latency[0]:.2f}", *counts])
            + " |"
        )
    lines += [
        "",
        (
            "Allocation counters run separately from timing and count allocation/reallocation calls "
            "and requested bytes, including middleware scheduling. They do not measure live memory. "
            "Raw output includes byte counts and all timing rounds."
        ),
        "",
        (
            "Loading and registration happen before timing. Every call checks the full returned value. "
            "Execution includes a real downstream continuation; fanout execution calls it twice concurrently. "
            "Request fanout uses the ready request path. Teardown must remove all registrations. "
            "These are local microbenchmarks, not application throughput measurements."
        ),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Source checkout to compare")
    parser.add_argument("--output", type=Path, default=Path("target/native-plugin-benchmark"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    if min(args.iterations, args.rounds, args.processes) < 1:
        parser.error("iterations, rounds, and processes must be positive")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "platform": platform.platform(),
        "compiler": command(["rustc", "-Vv"], root, capture=True).strip(),
        "sources": {},
        "runs": {"baseline": [], "checked": []},
    }
    for label, source in [("baseline", args.baseline.resolve()), ("checked", root)]:
        data["sources"][label] = prepare(source, output / label)
    os.environ["NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG"] = "1"
    for repetition in range(args.processes):
        for label in ("baseline", "checked") if repetition % 2 == 0 else ("checked", "baseline"):
            prepared = data["sources"][label]
            stdout = command(
                [prepared["binary"], prepared["manifest"], str(args.iterations), str(args.rounds)], output, capture=True
            )
            result = next(
                line.removeprefix("NATIVE_BENCH_RESULT ")
                for line in stdout.splitlines()
                if line.startswith("NATIVE_BENCH_RESULT ")
            )
            data["runs"][label].append(json.loads(result))
            (output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "report.md").write_text(report(data))
    print(output / "report.md")


if __name__ == "__main__":
    main()
