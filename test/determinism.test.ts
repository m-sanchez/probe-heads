/** Cross-version determinism, for real.
 *
 * The README's strongest claim is that the same data and hyperparameters
 * give bit-identical weights every run, so a regression is a real
 * regression. The test that used to back it called `fit` twice in one
 * process and deep-equalled the results, which cannot fail for code with
 * no randomness in it - it is a tautology, and it would not notice a
 * changed default, let alone drift in V8's Math.exp between Node majors.
 *
 * test/fixtures/golden.json is a frozen input matrix and the exact doubles
 * `fit` produced for it. The assertion is exact equality, and CI runs it on
 * Node 22, 24 and 26, so any of those failure modes surfaces as a
 * reviewable diff instead of quietly invalidating every probe number
 * anybody has stored. Do not regenerate the fixture to make this pass. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fit } from '../src/probe.ts';
import type { Example, FitOptions, ProbeModel } from '../src/probe.ts';

interface Block {
  features: number[][];
  labels: boolean[][];
}

interface GoldenCase {
  name: string;
  options: FitOptions;
  train: Block;
  val: Block;
  model: ProbeModel;
}

interface Golden {
  note: string;
  generatedWith: string;
  cases: GoldenCase[];
}

const golden = JSON.parse(
  readFileSync(fileURLToPath(new URL('./fixtures/golden.json', import.meta.url)), 'utf8')
) as Golden;

const examples = (b: Block): Example[] =>
  b.features.map((features, i) => ({ features, labels: b.labels[i] }));

function reproduce(index: number, name: string): void {
  const c = golden.cases[index];
  assert.equal(c.name, name, 'the fixture case moved out from under this test');
  const model = fit(examples(c.train), examples(c.val), c.options);
  // exact, not approximate: every weight, bias, threshold, scaler statistic
  // and convergence number, to the last bit
  assert.deepStrictEqual(model, c.model);
}

test('the fixture covers both a default run and a wider, retuned one', () => {
  assert.equal(golden.cases.length, 2);
  assert.deepEqual(golden.cases[0].options, {});
  assert.equal(golden.cases[0].model.dim, 6);
  assert.equal(golden.cases[1].model.dim, 32);
  assert.notDeepEqual(golden.cases[1].options, {});
});

test('fit reproduces the golden fixture exactly: 6 features, 2 labels, every default', () => {
  reproduce(0, '6 features, 2 labels, every default');
});

test('fit reproduces the golden fixture exactly: 32 features, 1 label, non-default hyperparameters', () => {
  reproduce(1, '32 features, 1 label, non-default hyperparameters');
});
