// "Are these two doubles close?", the way each library actually answers. Every test runs in binary64 with the
// same operations, in the same order, as the library's own code, so its True/False is the library's.
// Tested against CPython's math.isclose, NumPy's isclose / assert_allclose and pytest.approx
// (verify/test_closeness.mjs). Julia's isapprox and GoogleTest's DOUBLE_EQ follow their source.

import { bitsOfNumber } from "./fpbig.js";

export const EPS = 2 ** -52; // Number.EPSILON, sys.float_info.epsilon, DBL_EPSILON

/** The double's bits as an integer key in which neighbours differ by 1 and ±0 both map to 0 (Act 11). */
export function key(x) {
  const i = BigInt.asIntN(64, bitsOfNumber(x));
  return i >= 0n ? i : -(i & 0x7fffffffffffffffn);
}

/** Steps from a to b along the doubles (null if either is NaN). */
export function ulpDistance(a, b) {
  if (Number.isNaN(a) || Number.isNaN(b)) return null;
  const d = key(a) - key(b);
  return d < 0n ? -d : d;
}

/** The folk test: abs(a - b) < epsilon. */
export const epsEqual = (a, b) => Math.abs(a - b) < EPS;

/** CPython's math.isclose (Modules/mathmodule.c). */
export function pyIsclose(a, b, relTol = 1e-9, absTol = 0) {
  if (a === b) return true;
  if (a === Infinity || a === -Infinity || b === Infinity || b === -Infinity) return false;
  const diff = Math.abs(b - a);
  return diff <= Math.abs(relTol * b) || diff <= Math.abs(relTol * a) || diff <= absTol;
}

/** numpy.isclose(a, b): |a - b| <= atol + rtol*|b|, and b finite, or a == b. Asymmetric: b is the reference. */
export function npIsclose(a, b, rtol = 1e-5, atol = 1e-8, equalNan = false) {
  let r = (Math.abs(a - b) <= atol + rtol * Math.abs(b) && Number.isFinite(b)) || a === b;
  if (equalNan) r = r || (Number.isNaN(a) && Number.isNaN(b));
  return r;
}

/** numpy.testing.assert_allclose(actual, desired) passes: isclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True). */
export const npAssertAllclose = (actual, desired, rtol = 1e-7, atol = 0) => npIsclose(actual, desired, rtol, atol, true);

/** actual == pytest.approx(expected), with the default tolerances: max(1e-6 * |expected|, 1e-12). */
export function pytestApprox(actual, expected, rel = 1e-6, abs = 1e-12) {
  if (actual === expected) return true;
  if (Number.isNaN(expected)) return false; // nan_ok defaults to False
  if (!Number.isFinite(expected)) return false;
  return Math.abs(expected - actual) <= Math.max(rel * Math.abs(expected), abs);
}

/** Julia's isapprox(x, y) with default keywords: rtol = √eps(Float64) = 2^-26 when atol = 0. */
export function juliaIsapprox(x, y, atol = 0, rtol = atol > 0 ? 0 : Math.sqrt(EPS)) {
  return x === y || (Number.isFinite(x) && Number.isFinite(y) && Math.abs(x - y) <= Math.max(atol, rtol * Math.max(Math.abs(x), Math.abs(y))));
}

/** GoogleTest's EXPECT_DOUBLE_EQ: neither is NaN and they're at most 4 ulps apart. */
export function gtestDoubleEq(a, b) {
  const d = ulpDistance(a, b);
  return d !== null && d <= 4n;
}
