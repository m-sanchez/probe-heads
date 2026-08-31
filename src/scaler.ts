/** Feature standardisation, fit on training data only.
 *
 * The single most common leak in a probe is standardising with statistics
 * computed over the whole dataset - the model then "knows" the test set's
 * mean before it is evaluated. The scaler is fit from the training split
 * and applied unchanged to validation and holdout, so the holdout's
 * distribution never touches training. */

export interface Scaler {
  mean: number[];
  /** standard deviation, floored so a constant feature does not divide by 0 */
  std: number[];
}

export function fitScaler(features: number[][]): Scaler {
  if (features.length === 0) throw new Error('cannot fit a scaler on no examples');
  const dim = features[0].length;
  const mean = new Array(dim).fill(0);
  for (const x of features) for (let j = 0; j < dim; j++) mean[j] += x[j];
  for (let j = 0; j < dim; j++) mean[j] /= features.length;
  const std = new Array(dim).fill(0);
  for (const x of features) for (let j = 0; j < dim; j++) std[j] += (x[j] - mean[j]) ** 2;
  for (let j = 0; j < dim; j++) std[j] = Math.max(Math.sqrt(std[j] / features.length), 1e-8);
  return { mean, std };
}

export function transform(scaler: Scaler, x: number[]): number[] {
  return x.map((v, j) => (v - scaler.mean[j]) / scaler.std[j]);
}
