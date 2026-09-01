/** Claims the README makes about the package rather than about the maths.
 * Each of these rots silently: a dependency arrives with an npm install, a
 * CI leg is dropped, a version is bumped and the install line is not. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const read = (p: string): string => readFileSync(join(root, p), 'utf8');

test('zero runtime dependencies', () => {
  const pkg = JSON.parse(read('package.json')) as Record<string, unknown>;
  for (const field of [
    'dependencies',
    'peerDependencies',
    'optionalDependencies',
    'bundledDependencies'
  ]) {
    assert.deepEqual(pkg[field] ?? {}, {}, `${field} is not empty`);
  }
});

test('CI runs the suite on the Node versions the README names', () => {
  assert.match(read('.github/workflows/test.yml'), /node:\s*\[22,\s*24,\s*26\]/);
});

test('every version the README pins is the version this package is', () => {
  const { version } = JSON.parse(read('package.json')) as { version: string };
  const pinned = read('README.md')
    .split('probe-heads#v')
    .slice(1)
    .map((tail) => tail.split('`')[0]);
  assert.deepEqual(pinned, [version], 'the install line pins a stale tag');
});

test('every test CLAIMS.md names exists under that name', () => {
  const refs = [...read('CLAIMS.md').matchAll(/`(test\/[a-z]+\.test\.ts)::([^`]+)`/g)];
  assert.ok(refs.length > 30, `CLAIMS.md only references ${refs.length} tests`);
  const sources = new Map<string, string>();
  for (const [, file, name] of refs) {
    if (!sources.has(file)) sources.set(file, read(file));
    assert.ok(
      (sources.get(file) as string).includes("test('" + name + "'"),
      `${file} has no test called "${name}" - CLAIMS.md is out of date`
    );
  }
});
