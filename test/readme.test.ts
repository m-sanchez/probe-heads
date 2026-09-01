/** The README publishes a demo table as if it were output. This asserts it
 * is output: the block in README.md has to be, byte for byte, what
 * `npm run demo` prints on this checkout. A changed default, a changed
 * metric, or a number that drifted between Node versions all show up here
 * as a stale README rather than as a claim nobody checked. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const lf = (s: string): string => s.replace(/\r\n/g, '\n');

test('the README demo block is what npm run demo prints', () => {
  const printed = lf(
    execFileSync(process.execPath, ['demo/demo.ts'], { cwd: root, encoding: 'utf8' })
  ).trim();
  const readme = lf(readFileSync(join(root, 'README.md'), 'utf8'));
  assert.ok(
    readme.includes(printed),
    `the README's demo block is stale. npm run demo prints:\n\n${printed}\n`
  );
});
