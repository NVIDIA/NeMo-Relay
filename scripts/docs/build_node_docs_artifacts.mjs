// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SPDX_HEADER = [
  '<!--',
  'SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
  'SPDX-License-Identifier: Apache-2.0',
  '-->',
];

const REQUIRED_SPHINX_ENV = [
  'NEMO_RELAY_SPHINX_JS_MAIN_TS',
  'NEMO_RELAY_SPHINX_JS_IMPORT_HOOK',
  'NEMO_RELAY_SPHINX_JS_TSX_TSCONFIG',
];

const MODULES = [
  {
    declaration: 'index.d.ts',
    deppath: './declarations/index.d',
    entryTarget: './declarations/index',
    pageName: 'runtime',
    required: true,
    title: 'Runtime',
  },
  {
    declaration: 'typed.d.ts',
    deppath: './declarations/typed.d',
    entryTarget: './declarations/typed',
    pageName: 'typed',
    title: 'Typed Helpers',
  },
  {
    declaration: 'plugin.d.ts',
    deppath: './declarations/plugin.d',
    entryTarget: './declarations/plugin',
    pageName: 'plugin',
    title: 'Plugins',
  },
  {
    declaration: 'adaptive.d.ts',
    deppath: './declarations/adaptive.d',
    entryTarget: './declarations/adaptive',
    pageName: 'adaptive',
    title: 'Adaptive',
  },
  {
    declaration: 'observability.d.ts',
    deppath: './declarations/observability.d',
    entryTarget: './declarations/observability',
    pageName: 'observability',
    title: 'Observability',
  },
];

// These rewrites are a docs-only compatibility layer for TypeDoc/sphinx-js
// when it walks declaration files rather than authored `.ts` sources.
const DECLARATION_REWRITES = new Map([
  [
    'plugin.d.ts',
    [
      {
        original: "import type { Json } from './index';",
        replacement: 'type Json = import("./index").Json;',
      },
      {
        original: 'export interface ComponentSpec {',
        replacement: 'interface ComponentSpecShape {',
      },
      {
        original: '): ComponentSpec;',
        replacement: '): ComponentSpecShape;',
      },
    ],
  ],
  [
    'typed.d.ts',
    [
      {
        original: "import { ScopeHandle, LlmStream } from './index';",
        replacement: 'type ScopeHandle = import("./index").ScopeHandle;\ntype LlmStream = import("./index").LlmStream;',
      },
      {
        original: 'export interface JsonArray extends Array<JsonValue> {}',
        replacement: 'export type JsonArray = JsonValue[];',
      },
    ],
  ],
  [
    'adaptive.d.ts',
    [
      {
        original: "import type { Json } from './index';",
        replacement: 'type Json = import("./index").Json;',
      },
      {
        original:
          "import type { ConfigPolicy, ConfigDiagnostic, ConfigReport } from './plugin';\n\nexport { ConfigPolicy, ConfigDiagnostic, ConfigReport };",
        replacement: [
          'export type ConfigPolicy = import("./plugin").ConfigPolicy;',
          'export type ConfigDiagnostic = import("./plugin").ConfigDiagnostic;',
          'export type ConfigReport = import("./plugin").ConfigReport;',
        ].join('\n'),
      },
      {
        original: 'export interface ComponentSpec {',
        replacement: 'interface ComponentSpecShape {',
      },
      {
        original: '): ComponentSpec;',
        replacement: '): ComponentSpecShape;',
      },
    ],
  ],
  [
    'observability.d.ts',
    [
      {
        original: "import type { Json } from './index';",
        replacement: 'type Json = import("./index").Json;',
      },
      {
        original:
          "import type { ConfigPolicy, ConfigDiagnostic, ConfigReport } from './plugin';\n\nexport { ConfigPolicy, ConfigDiagnostic, ConfigReport };",
        replacement: [
          'export type ConfigPolicy = import("./plugin").ConfigPolicy;',
          'export type ConfigDiagnostic = import("./plugin").ConfigDiagnostic;',
          'export type ConfigReport = import("./plugin").ConfigReport;',
        ].join('\n'),
      },
      {
        original: 'export interface ComponentSpec {',
        replacement: 'interface ComponentSpecShape {',
      },
      {
        original: '): ComponentSpec;',
        replacement: '): ComponentSpecShape;',
      },
    ],
  ],
]);

const PUBLIC_NAME_REWRITES = new Map([['ComponentSpecShape', 'ComponentSpec']]);

const repoRoot = process.env.NEMO_RELAY_DOCS_REPO_ROOT
  ? path.resolve(process.env.NEMO_RELAY_DOCS_REPO_ROOT)
  : path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const nodePackageDir = path.join(repoRoot, 'crates', 'node');
const docsDir = process.env.NEMO_RELAY_DOCS_DIR
  ? path.resolve(process.env.NEMO_RELAY_DOCS_DIR)
  : path.join(repoRoot, 'docs');
const nodeDocsGeneratedDir = path.join(docsDir, 'reference', 'api', 'nodejs', '_generated');
const nodeDocsSourceDir = path.join(nodeDocsGeneratedDir, 'source');
const nodeDocsDeclDir = path.join(nodeDocsSourceDir, 'declarations');
const nodeApiJsonPath = path.join(nodeDocsGeneratedDir, 'node-api.json');

function main() {
  runCommand('npm', ['run', 'build-debug'], { cwd: nodePackageDir });
  resetGeneratedDirectories();
  ensureRequiredArtifacts(['index.js', 'index.d.ts']);
  const modules = availableModules();
  stageDeclarationSources(modules);
  writeModuleEntrypoints(modules);
  writeDocsPackageManifest();
  runTypeDoc();
  writeModulePages(readTypedocItems(), modules);
}

function runCommand(command, args, { cwd, env = process.env } = {}) {
  const result = spawnSync(command, args, {
    cwd,
    env,
    stdio: 'inherit',
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`);
  }
}

function resetGeneratedDirectories() {
  // This script is intentionally self-contained so it can be rerun outside the
  // Sphinx hook without depending on `docs/conf.py` to clean up first.
  rmSync(nodeDocsGeneratedDir, { recursive: true, force: true });
  mkdirSync(nodeDocsDeclDir, { recursive: true });
}

function ensureRequiredArtifacts(artifacts) {
  for (const artifact of artifacts) {
    const artifactPath = path.join(nodePackageDir, artifact);
    if (!existsSync(artifactPath)) {
      throw new Error(`Expected generated Node docs artifact missing: ${artifactPath}`);
    }
  }
}

function declarationPath(module) {
  return path.join(nodePackageDir, module.declaration);
}

function availableModules() {
  return MODULES.filter((module) => {
    if (existsSync(declarationPath(module))) {
      return true;
    }
    if (module.required) {
      throw new Error(`Expected generated Node docs artifact missing: ${declarationPath(module)}`);
    }

    console.warn(`Skipping Node API docs module with no declaration artifact: ${module.declaration}`);
    return false;
  });
}

function normalizeDeclaration(filename, contents) {
  const normalized = filename === 'index.d.ts' ? stripInternalTestHelpers(contents) : contents;
  const rewrites = DECLARATION_REWRITES.get(filename);
  if (!rewrites) {
    return normalized;
  }

  return rewrites.reduce((current, rewrite) => applyRewrite(filename, current, rewrite), normalized);
}

function stripInternalTestHelpers(contents) {
  return contents.replaceAll(
    /\/\*\* Internal test helper:[\s\S]*?\*\/\r?\nexport declare function __testClosed[^\r\n]*(?:\r?\n)?/g,
    '',
  );
}

function applyRewrite(filename, contents, { original, replacement }) {
  // Drift here usually means the exported `.d.ts` surface changed and the
  // docs-normalization shim needs to be updated alongside it.
  if (contents.includes(replacement)) {
    return contents;
  }
  if (!contents.includes(original)) {
    throw new Error(`Declaration rewrite drifted for ${filename}: ${original}`);
  }
  return contents.replace(original, replacement);
}

function stageDeclarationSources(modules) {
  for (const module of modules) {
    const contents = readUtf8(declarationPath(module));
    writeUtf8(path.join(nodeDocsDeclDir, module.declaration), normalizeDeclaration(module.declaration, contents));
  }
}

function writeModuleEntrypoints(modules) {
  // TypeDoc works more predictably when each documented surface has its own
  // tiny entrypoint instead of pointing it at a directory of declarations.
  for (const module of modules) {
    writeUtf8(path.join(nodeDocsSourceDir, `${module.pageName}.ts`), `export * from "${module.entryTarget}";\n`);
  }
}

function writeDocsPackageManifest() {
  writeUtf8(
    path.join(nodeDocsSourceDir, 'package.json'),
    `${JSON.stringify({ name: 'nemo-relay-node-api-docs', private: true }, null, 2)}\n`,
  );
}

function requireEnv(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function runTypeDoc() {
  const [mainTs, importHook, tsxTsconfig] = REQUIRED_SPHINX_ENV.map(requireEnv);

  runCommand(
    'npx',
    [
      'tsx@4.15.8',
      '--tsconfig',
      tsxTsconfig,
      '--import',
      importHook,
      mainTs,
      '--entryPointStrategy',
      'expand',
      '--options',
      path.join(docsDir, 'typedoc.node.json'),
      '--tsconfig',
      path.join(docsDir, 'typedoc.node.tsconfig.json'),
      '--basePath',
      nodeDocsSourceDir,
      '--json',
      nodeApiJsonPath,
      nodeDocsSourceDir,
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        TYPEDOC_NODE_MODULES: path.join(nodePackageDir, 'node_modules'),
      },
    },
  );
}

function readTypedocItems() {
  // sphinx-js/TypeDoc currently emits a nested array shape for this pipeline.
  // Accept the flat form too so minor upstream JSON changes are easier to absorb.
  const rawIr = JSON.parse(readUtf8(nodeApiJsonPath));
  if (Array.isArray(rawIr) && Array.isArray(rawIr[0])) {
    return rawIr[0];
  }
  if (Array.isArray(rawIr)) {
    return rawIr;
  }
  throw new Error(`Unexpected TypeDoc JSON structure in ${nodeApiJsonPath}`);
}

function writeModulePages(irObjects, modules) {
  const grouped = new Map(modules.map((module) => [module.deppath, []]));

  for (const item of irObjects) {
    if (!isDocumentedNodeApiItem(item)) {
      continue;
    }
    if (grouped.has(item.deppath)) {
      grouped.get(item.deppath).push(item);
    }
  }

  for (const module of modules) {
    writeModulePage(module, grouped.get(module.deppath) ?? []);
  }
}

function isDocumentedNodeApiItem(item) {
  if (!item?.deppath || !item.documentation_root) {
    return false;
  }

  // The Node binding exports a few underscored helpers for native callback
  // tests. Keep them callable from tests, but out of public API reference
  // pages.
  return !String(item.name ?? '').startsWith('__test');
}

function writeModulePage(module, items) {
  const rendered = items.map(renderItem).filter(Boolean).join('\n');
  writeUtf8(
    path.join(nodeDocsGeneratedDir, `${module.pageName}.md`),
    [
      ...SPDX_HEADER,
      '',
      `# ${module.title}`,
      '',
      `Generated from \`crates/node/${module.declaration}\` via \`sphinx-js\`.`,
      '',
      rendered,
      '',
    ].join('\n'),
  );
}

function renderInline(items) {
  if (!items) {
    return '';
  }
  if (typeof items === 'string') {
    return items;
  }

  return items
    .map((item) => {
      if (typeof item === 'string') {
        return item;
      }
      if (item.type === 'code') {
        return `\`${item.code}\``;
      }
      return item.text ?? publicName(item.name) ?? '';
    })
    .join('');
}

function publicName(name) {
  if (!name) {
    return name;
  }
  return PUBLIC_NAME_REWRITES.get(name) ?? name;
}

function renderType(items) {
  const value = renderInline(items)
    .trim()
    .replaceAll(/\bComponentSpecShape\b/g, 'ComponentSpec');
  return value ? `\`${value}\`` : '`unknown`';
}

function renderList(title, values) {
  if (!values?.length) {
    return '';
  }
  return [`### ${title}`, '', ...values.map((value) => `- ${value}`), ''].join('\n');
}

function renderTypedDescription(name, typeItems, description, { optional = false } = {}) {
  const renderedType = renderType(typeItems);
  const renderedName = optional ? `${publicName(name)}?` : publicName(name);
  if (!description) {
    return `\`${renderedName}\` ${renderedType}`;
  }
  return `\`${renderedName}\` ${renderedType}: ${description}`;
}

function renderFunction(item) {
  const params = (item.params ?? []).map((param) => {
    const description = renderInline(param.description).trim();
    return renderTypedDescription(param.name, param.type, description, {
      optional: Boolean(param.is_optional),
    });
  });
  const returns = (item.returns ?? []).map((ret) => {
    const description = renderInline(ret.description).trim();
    return description ? `${renderType(ret.type)}: ${description}` : renderType(ret.type);
  });
  const remarks = (item.block_tags?.remarks ?? []).map((remark) => renderInline(remark));

  return [
    `## \`${publicName(item.name)}\``,
    '',
    renderInline(item.description) || 'No description available.',
    '',
    `Kind: \`${item.kind}\``,
    '',
    renderList('Parameters', params),
    renderList('Returns', returns),
    renderList('Remarks', remarks),
  ]
    .filter(Boolean)
    .join('\n');
}

function renderAttribute(item) {
  return [
    `## \`${publicName(item.name)}\``,
    '',
    renderInline(item.description) || 'No description available.',
    '',
    `Type: ${renderType(item.type)}`,
    '',
  ].join('\n');
}

function renderTypeLike(item) {
  const members = (item.members ?? []).map((member) => {
    const description = renderInline(member.description).trim();
    return renderTypedDescription(member.name, member.type, description, {
      optional: Boolean(member.is_optional),
    });
  });
  const supers = (item.supers ?? []).map((sup) => renderType(sup));

  return [
    `## \`${publicName(item.name)}\``,
    '',
    renderInline(item.description) || 'No description available.',
    '',
    `Kind: \`${item.kind}\``,
    '',
    renderList('Extends', supers),
    renderList('Members', members),
  ]
    .filter(Boolean)
    .join('\n');
}

function renderAlias(item) {
  return [
    `## \`${publicName(item.name)}\``,
    '',
    renderInline(item.description) || 'No description available.',
    '',
    `Alias Of: ${renderType(item.type)}`,
    '',
  ].join('\n');
}

function renderItem(item) {
  switch (item.kind) {
    case 'function':
      return renderFunction(item);
    case 'attribute':
      return renderAttribute(item);
    case 'class':
    case 'interface':
      return renderTypeLike(item);
    case 'typeAlias':
      return renderAlias(item);
    default:
      return '';
  }
}

function readUtf8(filePath) {
  return readFileSync(filePath, 'utf8');
}

function writeUtf8(filePath, contents) {
  writeFileSync(filePath, contents, 'utf8');
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
