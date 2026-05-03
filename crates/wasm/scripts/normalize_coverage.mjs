// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';

const [inputPath, outputPath = inputPath] = process.argv.slice(2);

if (!inputPath) {
  console.error('Usage: node scripts/normalize_coverage.mjs <input-cobertura.xml> [output-cobertura.xml]');
  process.exit(1);
}

const filenameMap = new Map([
  ['pkg/adaptive.js', 'wrappers/nodejs/adaptive.js'],
  ['pkg/index.js', 'wrappers/nodejs/index.js'],
  ['pkg/plugin.js', 'wrappers/nodejs/plugin.js'],
  ['pkg/typed.js', 'wrappers/nodejs/typed.js'],
  ['wrappers/nodejs/adaptive.js', 'wrappers/nodejs/adaptive.js'],
  ['wrappers/nodejs/index.js', 'wrappers/nodejs/index.js'],
  ['wrappers/nodejs/plugin.js', 'wrappers/nodejs/plugin.js'],
  ['wrappers/nodejs/typed.js', 'wrappers/nodejs/typed.js'],
]);

function getAttribute(xml, name) {
  const match = xml.match(new RegExp(`${name}="([^"]*)"`));
  return match?.[1] ?? '';
}

function setAttribute(xml, name, value) {
  const escaped = String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;');
  const pattern = new RegExp(`${name}="[^"]*"`);
  return xml.replace(pattern, `${name}="${escaped}"`);
}

function rate(covered, valid) {
  return valid === 0 ? '1' : String(covered / valid);
}

function normalizeCoverageFilename(filename) {
  return filename.replaceAll('\\', '/').replace(/^(?:\.\/)+/, '');
}

function summarizeClass(xml) {
  let linesValid = 0;
  let linesCovered = 0;
  let branchesValid = 0;
  let branchesCovered = 0;

  for (const lineMatch of xml.matchAll(/<line\b[^>]*>/g)) {
    const line = lineMatch[0];
    linesValid += 1;
    if (Number(getAttribute(line, 'hits')) > 0) {
      linesCovered += 1;
    }

    const branchMatch = getAttribute(line, 'condition-coverage').match(/\((\d+)\/(\d+)\)/);
    if (branchMatch) {
      branchesCovered += Number(branchMatch[1]);
      branchesValid += Number(branchMatch[2]);
    }
  }

  return {
    linesValid,
    linesCovered,
    branchesValid,
    branchesCovered,
  };
}

function sumCoverage(items) {
  return items.reduce(
    (total, item) => ({
      linesValid: total.linesValid + item.linesValid,
      linesCovered: total.linesCovered + item.linesCovered,
      branchesValid: total.branchesValid + item.branchesValid,
      branchesCovered: total.branchesCovered + item.branchesCovered,
    }),
    { linesValid: 0, linesCovered: 0, branchesValid: 0, branchesCovered: 0 },
  );
}

const input = fs.readFileSync(inputPath, 'utf8');
const classes = [];

for (const classMatch of input.matchAll(/<class\b[\s\S]*?<\/class>/g)) {
  let classXml = classMatch[0];
  const filename = normalizeCoverageFilename(getAttribute(classXml, 'filename'));
  const normalizedFilename = filenameMap.get(filename);

  if (!normalizedFilename) {
    continue;
  }

  const summary = summarizeClass(classXml);
  classXml = setAttribute(classXml, 'filename', normalizedFilename);
  classXml = setAttribute(classXml, 'line-rate', rate(summary.linesCovered, summary.linesValid));
  classXml = setAttribute(classXml, 'branch-rate', rate(summary.branchesCovered, summary.branchesValid));
  classes.push({ xml: classXml, ...summary });
}

if (classes.length === 0) {
  throw new Error(`no checked-in WebAssembly wrapper coverage classes found in ${inputPath}`);
}

const total = sumCoverage(classes);
let output = input;
output = output.replace(/<coverage\b[^>]*>/, (match) => {
  let updated = match;
  updated = setAttribute(updated, 'lines-valid', total.linesValid);
  updated = setAttribute(updated, 'lines-covered', total.linesCovered);
  updated = setAttribute(updated, 'line-rate', rate(total.linesCovered, total.linesValid));
  updated = setAttribute(updated, 'branches-valid', total.branchesValid);
  updated = setAttribute(updated, 'branches-covered', total.branchesCovered);
  updated = setAttribute(updated, 'branch-rate', rate(total.branchesCovered, total.branchesValid));
  return updated;
});
output = output.replace(/<package\b[^>]*>/, (match) => {
  let updated = match;
  updated = setAttribute(updated, 'name', 'wasm_wrappers');
  updated = setAttribute(updated, 'line-rate', rate(total.linesCovered, total.linesValid));
  updated = setAttribute(updated, 'branch-rate', rate(total.branchesCovered, total.branchesValid));
  return updated;
});
output = output.replace(
  /<classes>[\s\S]*?<\/classes>/,
  `<classes>\n${classes.map((item) => item.xml).join('\n')}\n      </classes>`,
);

fs.writeFileSync(outputPath, output);
