# probe-heads

![TypeScript](https://img.shields.io/badge/TypeScript-erasable_syntax-3178C6?logo=typescript&logoColor=white)
![Node](https://img.shields.io/badge/node-%3E%3D22.18-5FA04E?logo=nodedotjs&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-B45309)
[![CI](https://github.com/m-sanchez/probe-heads/actions/workflows/test.yml/badge.svg)](https://github.com/m-sanchez/probe-heads/actions/workflows/test.yml)
![License](https://img.shields.io/badge/license-MIT-6E6E6E)

Reproducible probing of frozen embeddings: deterministic one-vs-rest
logistic heads, leakage-safe scaling, thresholds chosen on validation, the
holdout touched once. Zero dependencies.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics)

A linear probe answers "is this property linearly readable from the
embedding?" It is a small, honest experiment - and small honest
experiments are where evaluation hygiene quietly breaks: the scaler is fit
over the whole dataset, the decision threshold is tuned on the test set, or
a shuffle makes the run unreproducible. probe-heads wires the discipline
into the shape of the API so those mistakes are hard to make.

```ts
import { fit, evaluate } from 'probe-heads';

// fit sees train and val only: it standardises on train, trains a
// logistic head per label on train, and picks each label's threshold
// on val
const model = fit(train, val, { epochs: 300 });

// the holdout is not an argument to fit; it goes here, once, at the end
const report = evaluate(model, holdout);
report.perLabel;   // { precision, recall, f1, support } per label
report.macroF1;    // and micro
```

`npm run demo` probes two labels off a frozen 6-D embedding:

```
label   threshold   P      R      F1     support
0       0.30        0.64   0.93   0.76   191
1       0.35        0.63   0.67   0.65   94
macro F1 0.703, micro F1 0.725
thresholds were chosen on validation; this is the first look at the holdout.
```

## The discipline, made structural

- **Leakage-safe scaling.** The feature scaler is fit on the training
  split and applied unchanged to val and holdout, so the holdout's
  distribution never informs training.
- **Deterministic training.** Each head is full-batch gradient descent
  from zero init with no shuffling and no randomness: the same data and
  hyperparameters give bit-identical weights every run. A regression is a
  real regression, not a reseed.
- **Thresholds on validation, not the test set.** A logistic head emits a
  probability; the yes/no cut is chosen to maximise F1 on validation
  (0.5 is rarely right for a rare label), never on train and never on the
  holdout.
- **The holdout is spent once.** It is not a parameter of `fit`. The type
  keeps the final measurement out of the loop where it would get peeked.

## Honest limits

- This is a *linear* probe: it measures what a linear read of the frozen
  features can recover, which is exactly the question probes are for. It is
  not a classifier to ship, and a low score means "not linearly readable",
  not "not present".
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

Install: `npm install github:m-sanchez/probe-heads#v1.0.0` (not yet on npm;
CI proves the packed tarball imports). Node 22.18+, zero runtime
dependencies.

## The tests are the point

| Test | Claim |
| :-- | :-- |
| the probe learns a separable label to high holdout F1 | the training actually works |
| same data in, bit-identical model out | reproducible by construction, not by luck |
| the scaler is fit on train only | the holdout's statistics never leak into training |
| threshold selection beats a fixed 0.5 on a rare label | the cut is chosen where it matters |
| the holdout is not an argument to fit | the discipline is enforced by the signature |
| a constant feature does not divide by zero | the scaler is numerically safe |
| sigmoid is stable at the extremes | no overflow between the model and the metric |
