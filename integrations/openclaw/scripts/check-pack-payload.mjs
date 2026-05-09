/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Package payload validation for the OpenClaw integration.
 *
 * This script guards the npm package boundary: production source files,
 * generated dist files, and OpenClaw manifest entries must be packed, while
 * tests, maps, and test build output must stay out of the package.
 */
import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const npmExecPath = process.env.npm_execpath;

/** Fail the pack check with a precise validation message. */
function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

/** Run a child command from the package root and surface stderr on failure. */
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: packageRoot,
    encoding: "utf8",
    ...options,
  });
  if (result.status !== 0) {
    if (result.error) {
      process.stderr.write(`${result.error.message}\n`);
    }
    process.stderr.write(result.stderr ?? "");
    throw new Error(`${command} ${args.join(" ")} failed`);
  }
  return result;
}

/** Run npm through the active npm CLI when available, preserving workspace behavior. */
function runNpm(args, options = {}) {
  if (npmExecPath) {
    return run(process.execPath, [npmExecPath, ...args], options);
  }
  return run(npm, args, {
    shell: process.platform === "win32",
    ...options,
  });
}

/** Normalize npm pack paths to POSIX style without a leading ./ prefix. */
function normalizePackagePath(value) {
  return value.replace(/^\.\//, "").replaceAll("\\", "/");
}

/** Recursively list files below a package-local directory. */
function walkFiles(root, prefix = "") {
  const absoluteRoot = path.join(packageRoot, root, prefix);
  const output = [];
  for (const entry of readdirSync(absoluteRoot)) {
    const relative = path.posix.join(prefix, entry);
    const absolute = path.join(packageRoot, root, relative);
    if (statSync(absolute).isDirectory()) {
      output.push(...walkFiles(root, relative));
    } else {
      output.push(path.posix.join(root, relative));
    }
  }
  return output.sort();
}

runNpm(["run", "build"], { stdio: "inherit" });

const pack = runNpm(["pack", "--dry-run", "--json", "--ignore-scripts"]);
const packInfo = JSON.parse(pack.stdout)[0];
assert(packInfo, "npm pack did not return package metadata");

const productionSources = walkFiles("src").filter(
  (file) => file.endsWith(".ts") && !file.includes("/__tests__/") && !file.endsWith(".test.ts"),
);
const packedFiles = new Set(packInfo.files.map((file) => normalizePackagePath(file.path)));
const packageJson = JSON.parse(readFileSync(path.join(packageRoot, "package.json"), "utf8"));
const declaredFiles = new Set(packageJson.files ?? []);

for (const entry of declaredFiles) {
  assert(
    !(entry.startsWith("src/") && entry.includes("*")),
    `package files should explicitly allowlist production sources, not ${entry}`,
  );
}

for (const source of productionSources) {
  assert(declaredFiles.has(source), `package files allowlist is missing ${source}`);
  assert(packedFiles.has(source), `packed package is missing source file ${source}`);
}

const requiredFiles = [
  "package.json",
  "README.md",
  "index.ts",
  "openclaw.plugin.json",
  "dist/index.js",
  "dist/index.d.ts",
];

for (const file of requiredFiles) {
  assert(packedFiles.has(file), `packed package is missing ${file}`);
}

for (const entry of packageJson.openclaw?.extensions ?? []) {
  const file = normalizePackagePath(entry);
  assert(packedFiles.has(file), `openclaw.extensions entry ${entry} is not packed`);
}

for (const entry of packageJson.openclaw?.runtimeExtensions ?? []) {
  const file = normalizePackagePath(entry);
  assert(packedFiles.has(file), `openclaw.runtimeExtensions entry ${entry} is not packed`);
}

assert(packageJson.openclaw?.compat?.pluginApi, "openclaw.compat.pluginApi is required");
assert(packageJson.openclaw?.compat?.minGatewayVersion, "openclaw.compat.minGatewayVersion is required");
assert(packageJson.openclaw?.build?.openclawVersion, "openclaw.build.openclawVersion is required");
assert(packageJson.openclaw?.build?.pluginSdkVersion, "openclaw.build.pluginSdkVersion is required");

for (const file of packedFiles) {
  assert(!file.includes("__tests__"), `packed package includes test artifact ${file}`);
  assert(!file.startsWith(".test-dist/"), `packed package includes test output ${file}`);
  assert(!file.endsWith(".map"), `packed package includes source/declaration map ${file}`);
}

const builtDistFiles = new Set(walkFiles("dist"));
for (const file of builtDistFiles) {
  assert(packedFiles.has(file), `built dist file ${file} is not packed`);
}

for (const file of packedFiles) {
  if (file.startsWith("dist/")) {
    assert(builtDistFiles.has(file), `packed dist file ${file} was not produced by the fresh build`);
  }
}

console.log(`pack payload ok: ${packedFiles.size} files`);
