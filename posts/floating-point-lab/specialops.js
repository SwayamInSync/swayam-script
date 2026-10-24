// Operations on the special values (±0, ±∞, NaN) and the IEEE 754 exception flags they raise.
// Arithmetic uses JavaScript's numbers, which follow IEEE 754 binary64 here; the flags are IEEE 754's rules for
// quiet-NaN inputs (JavaScript has no way to read the hardware flags). Tested against NumPy (values and flags)
// and Python (verify/test_specialops.mjs).

import { bitsOfNumber } from "./fpbig.js";

export const VALUES = [-Infinity, -1, -0, 0, 1, Infinity, NaN];

const isInf = (x) => x === Infinity || x === -Infinity;
const isZero = (x) => x === 0;
const neg0 = (x) => Object.is(x, -0);

/** IEEE 754-2019 minimum: NaN if either is NaN; −0 below +0. */
export function minimum(x, y) {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (isZero(x) && isZero(y)) return neg0(x) || neg0(y) ? -0 : 0;
  return x < y ? x : y;
}
export function maximum(x, y) {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (isZero(x) && isZero(y)) return neg0(x) && neg0(y) ? -0 : 0;
  return x > y ? x : y;
}
/** IEEE 754-2019 minimumNumber: a number beats a (quiet) NaN; −0 below +0. */
export function minimumNumber(x, y) {
  if (Number.isNaN(x)) return y;
  if (Number.isNaN(y)) return x;
  return minimum(x, y);
}
export function maximumNumber(x, y) {
  if (Number.isNaN(x)) return y;
  if (Number.isNaN(y)) return x;
  return maximum(x, y);
}
/** Python's min(x, y) and max(x, y): keep the first unless the second is strictly smaller (larger). */
export const pyMin = (x, y) => (y < x ? y : x);
export const pyMax = (x, y) => (y > x ? y : x);

/** IEEE 754 totalOrder(x, y): x ≤ y in the order −NaN < −∞ < … < −0 < +0 < … < +∞ < +NaN (this NaN is +NaN). */
export function totalOrder(x, y) {
  const rank = (v) => {
    if (Number.isNaN(v)) return [bitsOfNumber(v) >> 63n ? -2 : 2, 0];
    if (isZero(v)) return [0, neg0(v) ? -1 : 1];
    return [v < 0 ? -1 : 1, v];
  };
  const [a, b] = [rank(x), rank(y)];
  if (a[0] !== b[0]) return a[0] < b[0];
  return a[1] <= b[1];
}

/** Each operation: its JavaScript result, and the IEEE 754 flags it raises (quiet-NaN inputs). */
export const OPS = {
  add: { label: "x + y", f: (x, y) => x + y, flags: (x, y) => (isInf(x) && isInf(y) && x !== y ? ["invalid"] : []) },
  sub: { label: "x − y", f: (x, y) => x - y, flags: (x, y) => (isInf(x) && isInf(y) && x === y ? ["invalid"] : []) },
  mul: { label: "x × y", f: (x, y) => x * y, flags: (x, y) => ((isZero(x) && isInf(y)) || (isInf(x) && isZero(y)) ? ["invalid"] : []) },
  div: {
    label: "x ÷ y", f: (x, y) => x / y,
    flags: (x, y) => ((isZero(x) && isZero(y)) || (isInf(x) && isInf(y)) ? ["invalid"] : isZero(y) && Number.isFinite(x) && !isZero(x) ? ["divideByZero"] : []),
  },
  eq: { label: "x == y", f: (x, y) => x === y, flags: () => [] },
  lt: { label: "x < y", f: (x, y) => x < y, flags: (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? ["invalid"] : []) },
  minimum: { label: "minimum(x, y) (IEEE 754-2019)", f: minimum, flags: () => [] },
  minimumNumber: { label: "minimumNumber(x, y) (IEEE 754-2019)", f: minimumNumber, flags: () => [] },
  jsMin: { label: "Math.min(x, y) (JavaScript)", f: Math.min, flags: () => [] },
  pyMin: { label: "min(x, y) (Python)", f: pyMin, flags: () => [] },
  maximum: { label: "maximum(x, y) (IEEE 754-2019)", f: maximum, flags: () => [] },
  maximumNumber: { label: "maximumNumber(x, y) (IEEE 754-2019)", f: maximumNumber, flags: () => [] },
  jsMax: { label: "Math.max(x, y) (JavaScript)", f: Math.max, flags: () => [] },
  pyMax: { label: "max(x, y) (Python)", f: pyMax, flags: () => [] },
  totalOrder: { label: "totalOrder(x, y) (IEEE 754)", f: totalOrder, flags: () => [] },
};
