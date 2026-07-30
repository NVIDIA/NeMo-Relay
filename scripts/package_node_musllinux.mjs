#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

const ignoredDirectories = new Set([".git", ".venv", "node_modules", "target", "tmp"]);

function command(name, args, cwd) {
  const result = spawnSync(name, args, { cwd, stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`${name} ${args.join(" ")} failed with exit code ${result.status}`);
  }
}

function argumentsFrom(args) {
  let version;
  let output;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--version") {
      version = args[++index];
    } else if (args[index] === "--out") {
      output = args[++index];
    } else {
      throw new Error(`Unexpected argument: ${args[index]}`);
    }
  }
  if (!version || !output) {
    throw new Error("Usage: package_node_musllinux.mjs --version VERSION --out DIRECTORY");
  }
  return { output: resolve(output), version };
}

function setPackageVersion(sourceDirectory, version) {
  const packagePath = join(sourceDirectory, "crates", "node", "package.json");
  const packageJson = JSON.parse(readFileSync(packagePath, "utf8"));
  packageJson.version = version;
  writeFileSync(packagePath, `${JSON.stringify(packageJson, null, 2)}\n`);

  const lockPath = join(sourceDirectory, "package-lock.json");
  const lock = JSON.parse(readFileSync(lockPath, "utf8"));
  lock.packages["crates/node"].version = version;
  writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`);
}

function main() {
  const { output, version } = argumentsFrom(process.argv.slice(2));
  const repository = process.cwd();
  const temporaryDirectory = mkdtempSync(join(tmpdir(), "nemo-relay-node-musllinux-"));
  const sourceDirectory = join(temporaryDirectory, "source");

  try {
    mkdirSync(output, { recursive: true });
    cpSync(repository, sourceDirectory, {
      filter: (source) => !ignoredDirectories.has(basename(source)),
      recursive: true,
    });
    setPackageVersion(sourceDirectory, version);
    command("npm", ["install", "--workspace=nemo-relay-node", "--ignore-scripts"], sourceDirectory);
    command("npm", ["run", "--workspace=nemo-relay-node", "build"], sourceDirectory);
    command(
      "npm",
      ["pack", "--workspace=nemo-relay-node", "--pack-destination", output],
      sourceDirectory,
    );

    const packages = readdirSync(output).filter((entry) => entry.endsWith(".tgz"));
    if (packages.length !== 1) {
      throw new Error(`Expected one npm package artifact in ${output}, found ${packages.length}`);
    }
  } finally {
    rmSync(temporaryDirectory, { force: true, recursive: true });
  }
}

main();
