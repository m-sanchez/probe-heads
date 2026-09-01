/** Feature standardisation, fit on training data only.
 *
 * The single most common leak in a probe is standardising with statistics
 * computed over the whole dataset - the model then "knows" the test set's
 * mean before it is evaluated. The scaler is fit from the training split
 * and applied unchanged to validation and holdout, so the holdout's
 * distribution never touches training.
 *
 * Both functions fail closed on shape. A scaler fit on 768 columns and
 * handed 769 has no meaningful answer, and the silent one it used to give
 * (NaN, or a score computed from a prefix of the vector) reads exactly
 * like a real result. */

export interface Scaler {
  mean: number[];
  /** standard deviation, floored so a constant feature does not divide by 0 */
  std: number[];
}

export function fitScaler(features: number[][]): Scaler {
  if (features.length === 0) throw new Error('cannot fit a scaler on no examples');
  const dim = features[0].length;
  if (dim === 0) throw new Error('cannot fit a scaler on zero-width feature vectors');
  for (let i = 0; i < features.length; i++) {
    if (features[i].length !== dim) {
      throw new Error(
        `ragged features: row 0 has width ${dim}, row ${i} has width ${features[i].length}`
      );
    }
  }
  const mean = new Array(dim).fill(0);
  for (const x of features) for (let j = 0; j < dim; j++) mean[j] += x[j];
  for (let j = 0; j < dim; j++) mean[j] /= features.length;
  const std = new Array(dim).fill(0);
  for (const x of features) for (let j = 0; j < dim; j++) std[j] += (x[j] - mean[j]) ** 2;
  for (let j = 0; j < dim; j++) std[j] = Math.max(Math.sqrt(std[j] / features.length), 1e-8);
  return { mean, std };
}

export function transform(scaler: Scaler, x: number[]): number[] {
  if (x.length !== scaler.mean.length) {
    throw new Error(
      `feature vector has ${x.length} values, the scaler expects ${scaler.mean.length}`
    );
  }
  return x.map((v, j) => (v - scaler.mean[j]) / scaler.std[j]);
}
