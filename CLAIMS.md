# Claims

Every externally falsifiable claim in `README.md` and in the `description`
field of `package.json`, mapped to the executable test that enforces it.
A claim with no test either got one in this pass or was narrowed until it
was true; the narrowings are listed at the bottom rather than quietly
dropped.

Run the whole set with `npm test`. Test names are exact - `node --test
--test-name-pattern "..."` will run any single row.

## Behaviour

| Claim | Where it is said | Enforced by |
| :-- | :-- | :-- |
| deterministic one-vs-rest logistic heads | package description; README lead | `test/determinism.test.ts::fit reproduces the golden fixture exactly: 6 features, 2 labels, every default` |
| the same data and hyperparameters give bit-identical weights every run | README, *Deterministic training* | `test/determinism.test.ts::fit reproduces the golden fixture exactly: 32 features, 1 label, non-default hyperparameters` |
| `fit` is pure - no state carries between calls | README, *Deterministic training* | `test/probe.test.ts::training is deterministic: same data in, bit-identical model out` |
| the golden fixture is asserted exactly on Node 22, 24 and 26 | README, *Deterministic training* | `test/determinism.test.ts::the fixture covers both a default run and a wider, retuned one` + `test/package.test.ts::CI runs the suite on the Node versions the README names` |
| leakage-safe scaling: the scaler is fit on train and applied unchanged to val and holdout | package description; README, *Leakage-safe scaling* | `test/probe.test.ts::fit standardises on train only: a shifted val cannot move the scaler` |
| head training is confined to train | README, *The holdout is kept out of training* | `test/probe.test.ts::fit trains heads on train only: flipping every val label moves no weight` |
| thresholds are chosen on validation, never on train and never on the holdout | package description; README, *Thresholds on validation* | `test/probe.test.ts::threshold selection beats a fixed 0.5 on a rare label` + `test/probe.test.ts::fit trains heads on train only: flipping every val label moves no weight` (the thresholds move when val labels do, the weights do not) |
| the holdout is kept out of training; nothing it contains can change the model | package description; README lead and *The holdout is kept out of training*; plain-English line | `test/probe.test.ts::nothing in the holdout can change the trained model` |
| `evaluate` is stateless - looking more than once costs nothing | README, *The holdout is kept out of training* | `test/probe.test.ts::nothing in the holdout can change the trained model` (second assertion) |
| one-vs-rest per label; it does not model label correlations | README, *Honest limits* | `test/probe.test.ts::one-vs-rest: no label can move another label head` |
| `predict` thresholds each label at its own selected cut | README code sample | `test/probe.test.ts::predict thresholds each label independently at its selected cut` |
| the training actually works on a separable label | README test table | `test/probe.test.ts::the probe learns a separable label to high holdout F1` |
| a constant feature does not divide by zero | README test table | `test/probe.test.ts::a constant feature does not divide by zero` |
| sigmoid is stable at the extremes | README test table | `test/probe.test.ts::sigmoid is stable and bounded at the extremes` |

## The control, the floor, and what a score means

| Claim | Where it is said | Enforced by |
| :-- | :-- | :-- |
| every F1 is reported against a permuted-label control | package description; README lead | `test/control.test.ts::a probe on signal clears its control` |
| a probe on labels drawn independently of the features reaches holdout F1 0.671, against real signal's 0.787 | README, after the demo block | `test/control.test.ts::a probe on pure noise scores like a result and clears nothing` (0.671) + `test/control.test.ts::a probe on signal clears its control` (0.787) |
| the mechanism: the threshold search collapses to the lowest cut, predicts everything positive, and lands at precision = base rate | README, after the demo block | `test/control.test.ts::a probe on pure noise scores like a result and clears nothing` (control F1 equals the always-positive floor to six decimals) |
| `selectivity` is this probe's F1 minus the control's | README, *Every F1 comes with its control and its floor* | `test/control.test.ts::a probe on signal clears its control` |
| selectivity is reported as `null`, not 0, when no control was run | README code sample | `test/control.test.ts::selectivity is null, not zero, when no control was run` |
| `baselineF1` is what a classifier answering "yes" to everything scores | README, *Every F1 comes with its control and its floor* | `test/control.test.ts::the always-positive floor is 2b/(1+b) on the holdout base rate` |
| the control is `p(i) = (stride * i + 1) mod n` - a permutation, so class balance is exact | README, *Every F1 comes with its control and its floor* | `test/control.test.ts::the control permutation is a bijection and never the identity` + `test/control.test.ts::the control permutes rows, so no label loses or gains a positive` |
| no RNG enters, so the control is as reproducible as the probe | README, *Every F1 comes with its control and its floor* | `test/control.test.ts::the control introduces no randomness: two control fits are identical` |
| the floor and the control are two different numbers, both worth printing | README test table | `test/control.test.ts::the control is not a restatement of the floor` |
| a control probe of the wrong shape is refused | API behaviour behind the claim above | `test/control.test.ts::evaluate refuses a control probe of a different shape` |
| the demo block in the README is what `npm run demo` prints - macro F1 0.703, floor 0.513, control 0.513, selectivity 0.189, and both per-label rows | README, demo block and the paragraph reading it | `test/readme.test.ts::the README demo block is what npm run demo prints` |

## Convergence

| Claim | Where it is said | Enforced by |
| :-- | :-- | :-- |
| 300 epochs leave the demo's two 6-D heads at gradient norms 1.2e-5 and 6.9e-4 | README, *Honest limits* (and printed by the demo) | `test/readme.test.ts::the README demo block is what npm run demo prints` |
| at d=64 the gradient norm after the default budget is 1.1e-3 | README, *Honest limits* | `test/convergence.test.ts::fit says so when the epoch budget runs out short of the tolerance` |
| at d=768 it is 2.4e-3 | README, *Honest limits* | `test/convergence.test.ts::at embedding width the default budget is nowhere near stationary` |
| `fit` takes a `tolerance` and stops at it; the d=64 case reaches 1e-6 in about 1,100 epochs | README, *Honest limits* | `test/convergence.test.ts::the returned head is stationary to the tolerance that was asked for` |
| a head whose `converged` is false has run out of epochs rather than measured readability | README, *Honest limits* | `test/convergence.test.ts::fit says so when the epoch budget runs out short of the tolerance` |
| the reported `finalGradNorm` is the gradient at the weights actually returned | README code sample (`model.convergence`) | `test/convergence.test.ts::the reported gradient norm is the real one at the returned weights` (recomputed in the test from the public API) |
| `convergence` carries epochs, finalLoss, finalGradNorm and converged, per head | README code sample | `test/convergence.test.ts::convergence is recorded per head` + `test/convergence.test.ts::trainHead reports where it stopped` |
| the default tolerance changes no existing number | README, *Honest limits* (implied by the unchanged demo output) | `test/convergence.test.ts::the default tolerance of 0 leaves v1 numbers untouched` |

## Fail closed

Every row here is a measured failure mode that used to return a plausible
number instead of an error.

| Claim | Where it is said | Enforced by |
| :-- | :-- | :-- |
| the model records the feature width it was fit on | README, *Fail closed on shape* | `test/shapes.test.ts::the model records the feature width it was fit on` |
| a holdout of the wrong width throws instead of returning `macroF1: 0` | README, *Fail closed on shape* | `test/shapes.test.ts::evaluate refuses a holdout of the wrong feature width` |
| a prediction vector of the wrong length throws | README, *Fail closed on shape* | `test/shapes.test.ts::predict refuses a feature vector that is too short` + `test/shapes.test.ts::predict refuses a feature vector that is too long` |
| a ragged feature matrix throws | README, *Fail closed on shape* | `test/shapes.test.ts::fit refuses ragged training features` + `test/shapes.test.ts::fitScaler refuses ragged rows instead of producing NaN statistics` |
| a scaler will not transform a vector it was not fit for | README, *Fail closed on shape* | `test/shapes.test.ts::transform refuses a vector the scaler was not fit for` + `test/shapes.test.ts::score and trainHead refuse vectors of the wrong width` |
| an empty validation split throws | README, *Fail closed on shape* | `test/shapes.test.ts::fit refuses an empty validation split rather than defaulting the cut` |
| a validation label with only one class present throws | README, *Fail closed on shape* | `test/shapes.test.ts::fit refuses a validation label with no positives` + `test/shapes.test.ts::fit refuses a validation label with no negatives` + `test/shapes.test.ts::selectThreshold refuses a degenerate label column` |
| an empty holdout throws rather than reporting zeros | *Fail closed on shape*, same stance | `test/shapes.test.ts::evaluate refuses an empty holdout` |
| mismatched score and label arrays throw | *Fail closed on shape*, same stance | `test/shapes.test.ts::f1At refuses mismatched score and label arrays` |

## The package

| Claim | Where it is said | Enforced by |
| :-- | :-- | :-- |
| zero runtime dependencies | README lead, badge, *Run* | `test/package.test.ts::zero runtime dependencies` |
| CI runs on Node 22, 24 and 26 | README, *Deterministic training* | `test/package.test.ts::CI runs the suite on the Node versions the README names` |
| the pinned git tag in the install line is this version | README, *Run* | `test/package.test.ts::every version the README pins is the version this package is` |
| Node 22.18+ | README, badge, *Run* | `package.json` `engines`, exercised by the Node 22 leg of the CI matrix |
| CI proves the packed tarball imports | README, *Run* | `.github/workflows/test.yml`, the `npm pack` and `install proof` steps (a CI step, not a unit test) |
| the test names in this file are exact and still exist | CLAIMS.md preamble | `test/package.test.ts::every test CLAIMS.md names exists under that name` |

## Narrowed or removed rather than left unenforced

- **"A logistic head emits a probability"** became "emits a score - not a
  calibrated probability". The scores are raw sigmoid outputs of an
  L2-regularised head whose decision cut is then deliberately moved off
  0.5; run through a calibration measure they are meaningfully
  miscalibrated (expected calibration error of roughly 0.07 and 0.03 on the
  demo's two labels). Nothing here measures calibration, so the word
  "probability" was removed from the README, from `Prediction.scores` in
  `src/probe.ts`, and from the header comment in `src/threshold.ts`, rather
  than backed by a test. If you need the probability reading, calibrate the
  scores first.
- **"macro F1 0.703"** as a standalone headline. It is still the headline,
  but it is never printed alone: the demo and the README show it beside the
  0.513 floor, the 0.513 control, and the 0.189 selectivity, because a
  probe on permuted labels reaches 0.671 on this shape of data.
- **"the scaler is fit on train only"** used to be backed by a test that
  called `fitScaler` on a two-row literal and never invoked `fit`. Fitting
  the scaler on train-union-val left the whole suite green. That row is now
  two tests that assert on `fit`.
- **Wall-clock timings** ("30ms", "2.4s") were drafted into *Honest limits*
  and then removed: they are machine-dependent and no test can hold them.
  The gradient norms next to them stayed, because tests do.
- **Ad-hoc measurement numbers.** An intermediate draft of *Honest limits*
  quoted gradient norms of 1.9e-13, 8.7e-4 and 5.6e-3, measured on
  throwaway data that no test reproduces. They were replaced with 1.2e-5 /
  6.9e-4, 1.1e-3 and 2.4e-3 - the numbers the suite actually pins.

## Deliberately not claims

Prose that guides interpretation or states scope, and is not the kind of
thing a test can settle: "a low score means *not linearly readable*, not
*not present*"; "it is not a classifier to ship"; "you bring the embeddings
and the splits"; the provenance note; and the framing sentence about the
API shape making mistakes hard to make.
