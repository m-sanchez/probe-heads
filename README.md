# probe-heads

![TypeScript](https://img.shields.io/badge/TypeScript-erasable_syntax-3178C6?logo=typescript&logoColor=white)
![Node](https://img.shields.io/badge/node-%3E%3D22.18-5FA04E?logo=nodedotjs&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-B45309)
[![CI](https://github.com/m-sanchez/probe-heads/actions/workflows/test.yml/badge.svg)](https://github.com/m-sanchez/probe-heads/actions/workflows/test.yml)
![License](https://img.shields.io/badge/license-MIT-6E6E6E)
[![npm](https://img.shields.io/npm/v/@m-sanchez/probe-heads?color=CB3837&logo=npm&logoColor=white)](https://www.npmjs.com/package/@m-sanchez/probe-heads)

> **In plain English:** this tests what a model actually learned by training a tiny classifier on its internals, keeping the test data strictly separate so the result is not cheating.

Reproducible probing of frozen embeddings: deterministic one-vs-rest
logistic heads, leakage-safe scaling, thresholds chosen on validation, the
holdout kept out of training, and every F1 reported against a
permuted-label control so you can tell it from noise. Zero dependencies.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics)

*Provenance: a fresh, dependency-free implementation of standard methods,
written to test the systems the other tools came from. First published
2026-08-31.*

A linear probe answers "is this property linearly readable from the
embedding?" It is a small, honest experiment - and small honest
experiments are where evaluation hygiene quietly breaks: the scaler is fit
over the whole dataset, the decision threshold is tuned on the test set, or
a shuffle makes the run unreproducible. probe-heads wires the discipline
into the shape of the API so those mistakes are hard to make.

```ts
import { controlFit, evaluate, fit } from '@m-sanchez/probe-heads';

// fit sees train and val only: it standardises on train, trains a
// logistic head per label on train, and picks each label's threshold
// on val
const model = fit(train, val, { epochs: 300 });

// the same pipeline on the same features, with the labels permuted: what
// this probe scores when there is no signal in the labels at all
const control = controlFit(train, val, { epochs: 300 });

// the holdout is not an argument to fit; it goes here, once, at the end
const report = evaluate(model, holdout, control);
report.perLabel;        // precision, recall, f1, support, baselineF1, selectivity
report.macroF1;         // and micro, macroBaselineF1, macroSelectivity
model.convergence;      // per head: epochs, finalLoss, finalGradNorm, converged
```

`npm run demo` probes two labels off a frozen 6-D embedding:

```
two labels probed off a frozen 6-D embedding (balanced, rare)

label  thr   P      R      F1     floor  control  selectivity  support
0      0.30  0.639  0.927  0.756  0.646  0.646    0.110        191
1      0.35  0.630  0.670  0.649  0.381  0.381    0.269        94

macro F1 0.703   floor 0.513   control 0.513   selectivity 0.189
heads stopped at gradient norm 1.2e-5 and 6.9e-4 after 300 and 300 epochs
thresholds were chosen on validation; this is the first look at the holdout.
read each F1 against its floor and its control, never against zero.
```

Read the macro row across, not down. **0.703** is the headline; **0.513**
is what this same pipeline scores on the same features when the labels are
permuted so no signal survives; **0.189** is the difference, and it is the
only part that is about the embedding. Published unflattering because it is
the number that means something.

The gap is not a quirk of this dataset. An F1-maximising threshold search
on a label it cannot predict collapses to the lowest cut in the grid,
predicts everything positive, and collects recall 1.0 at a precision equal
to the base rate. Measured on this package's own test data: a probe trained
on labels drawn independently of the features reaches holdout F1 **0.671**,
against real signal's 0.787. One number in isolation cannot tell linear
readability from class balance, which is the only question a probe is
asked.

## The discipline, made structural

- **Every F1 comes with its control and its floor.** `controlFit` runs the
  identical pipeline on the identical features with the labels
  deterministically permuted, and `evaluate(model, holdout, control)`
  reports `selectivity` - this probe's F1 minus that one's. `baselineF1` is
  what a classifier that answers "yes" to everything scores. The
  permutation is arithmetic, `p(i) = (stride * i + 1) mod n`, so no RNG
  enters and the control is as reproducible as the probe.
- **Leakage-safe scaling.** The feature scaler is fit on the training
  split and applied unchanged to val and holdout, so the holdout's
  distribution never informs training.
- **Deterministic training.** Each head is full-batch gradient descent
  from zero init with no shuffling and no randomness: the same data and
  hyperparameters give bit-identical weights every run. A regression is a
  real regression, not a reseed. `test/fixtures/golden.json` holds a fixed
  input matrix and the exact doubles `fit` produced for it, asserted for
  exact equality on Node 22, 24 and 26 - so a changed default, or float
  drift between engines, shows up as a reviewable diff instead of quietly
  invalidating every probe number you have stored.
- **Thresholds on validation, not the test set.** A logistic head emits a
  score - not a calibrated probability, since the head is L2-regularised
  and the cut is deliberately moved off 0.5, so rank it and threshold it
  rather than reading it as a likelihood. The yes/no cut is chosen to
  maximise F1 on validation (0.5 is rarely right for a rare label), never
  on train and never on the holdout.
- **Fail closed on shape.** The model records the feature width it was fit
  on, and every split is checked against it. A holdout of the wrong width
  used to return `macroF1: 0` - a shape bug served as the conclusion "not
  linearly readable". It now throws, as does a ragged feature matrix, a
  prediction vector of the wrong length, an empty validation split, and a
  validation label with only one class present (there is no
  F1-maximising cut to select, and the search would return an artefact of
  the grid).
- **The holdout is kept out of training.** It is not a parameter of `fit`:
  scaling, head training, and threshold selection are confined to train
  and validation, so the holdout cannot inform the model. `evaluate` is a
  separate, explicit step - a stateless function you call when you are
  ready to measure. (It does not *enforce* a single call; the guarantee is
  that nothing the holdout contains can change the model, not that you can
  only look once.)

## Honest limits

- This is a *linear* probe: it measures what a linear read of the frozen
  features can recover, which is exactly the question probes are for. It is
  not a classifier to ship, and a low score means "not linearly readable",
  not "not present".
- **The default epoch budget is sized for small problems.** 300 epochs of
  full-batch descent leave the demo's two 6-D heads at gradient norms
  1.2e-5 and 6.9e-4. At d=64 the norm is 1.1e-3, and at d=768 - the width
  of a sentence embedding - it is 2.4e-3, two hundred times further from
  stationarity. Those are underfit heads, not verdicts about the
  embedding. `fit` reports `convergence` per head and takes a `tolerance`
  it stops at; the d=64 case reaches 1e-6 in about 1,100 epochs. A head
  whose `converged` is false has not measured linear readability, it has
  run out of epochs.
- One-vs-rest per label; it does not model label correlations.
- You bring the embeddings and the splits. probe-heads does the training,
  threshold selection, and scoring; it does not compute features or choose
  your split sizes.

## Run

```bash
npm install       # dev-only: typescript
npm test
npm run demo
npm run typecheck
```

Install: `npm install @m-sanchez/probe-heads` (or a pinned git tag,
`github:m-sanchez/probe-heads#v2.0.0`; CI proves the packed tarball
imports). Node 22.18+, zero runtime dependencies.

## The tests are the point

| Test | Claim |
| :-- | :-- |
| a probe on pure noise scores like a result and clears nothing | an F1 on its own is not evidence, and the report now says so |
| a probe on signal clears its control | selectivity is the part of a score that is about the embedding |
| the control permutation is a bijection and never the identity | the control is a permutation, not a resample: class balance is exact |
| the control is not a restatement of the floor | the floor and the control are two different numbers, both worth printing |
| fit reproduces the golden fixture exactly | the same doubles across Node versions and across time |
| fit standardises on train only: a shifted val cannot move the scaler | no split outside train enters the scaler, asserted on `fit` |
| fit trains heads on train only: flipping every val label moves no weight | no label outside train can move a weight |
| nothing in the holdout can change the trained model | the guarantee, tested by mutating the holdout and re-checking |
| the returned head is stationary to the tolerance that was asked for | a low score is a finding, not an unfinished optimisation |
| evaluate refuses a holdout of the wrong feature width | a shape bug is an error, never a result |
| the probe learns a separable label to high holdout F1 | the training actually works |
| the README demo block is what npm run demo prints | the numbers above are output, not decoration |

[CLAIMS.md](CLAIMS.md) maps every falsifiable claim in this README, and in
the package description, to the test that enforces it - file and test name.
Anything that could not be enforced was narrowed or removed rather than
left standing.
