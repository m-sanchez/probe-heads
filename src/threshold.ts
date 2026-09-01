/** Per-label decision thresholds, selected on the validation split.
 *
 * A logistic head outputs a score; turning it into a yes/no needs a
 * threshold, and 0.5 is rarely the right one for an imbalanced label. The
 * threshold is chosen to maximise F1 on VALIDATION, never on training
 * (which would overfit) and never on the holdout (which must stay
 * untouched until the final evaluation). That split of duties is the whole
 * point: fit on train, choose thresholds on val, report on holdout.
 *
 * A label column with no positives has no F1-maximising cut - every grid
 * point scores 0. `selectThreshold` used to hand back the first grid point
 * in that case, a head that fires on everything, indistinguishable from a
 * chosen cut. It now refuses. */

export interface F1Point {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
}

export function f1At(scores: number[], labels: number[], threshold: number): F1Point {
  if (scores.length !== labels.length) {
    throw new Error(`got ${scores.length} scores and ${labels.length} labels`);
  }
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < scores.length; i++) {
    const pred = scores[i] >= threshold ? 1 : 0;
    if (pred === 1 && labels[i] === 1) tp++;
    else if (pred === 1 && labels[i] === 0) fp++;
    else if (pred === 0 && labels[i] === 1) fn++;
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { threshold, precision, recall, f1 };
}

/** Grid-search the threshold that maximises validation F1. Ties break to
 * the lower threshold (higher recall), deterministically. Requires a label
 * column with at least one positive: without one there is nothing to
 * maximise and any returned cut would be an artefact of the grid. */
export function selectThreshold(
  scores: number[],
  labels: number[],
  grid: number[] = defaultGrid()
): F1Point {
  if (scores.length === 0) throw new Error('cannot select a threshold on an empty split');
  if (grid.length === 0) throw new Error('cannot select a threshold from an empty grid');
  if (!labels.some((y) => y === 1)) {
    throw new Error('cannot select a threshold: the label column has no positive examples');
  }
  let best = f1At(scores, labels, grid[0]);
  for (let i = 1; i < grid.length; i++) {
    const point = f1At(scores, labels, grid[i]);
    if (point.f1 > best.f1) best = point;
  }
  return best;
}

export function defaultGrid(): number[] {
  const grid: number[] = [];
  for (let t = 0.05; t <= 0.951; t += 0.05) grid.push(Math.round(t * 100) / 100);
  return grid;
}
