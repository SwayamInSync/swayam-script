// Summation algorithms in binary64, for the summation race. Each is the textbook loop, run with JavaScript's
// numbers (IEEE 754 binary64, round to nearest even), so each returns exactly what the same loop returns in
// C or Python. Tested against NumPy, math.fsum and verify/act7.py (verify/test_summation.mjs).

import { FORMATS, roundTo, bitsOfNumber } from "./fpbig.js";

/** Left to right, one rounding per addition. */
export function naive(xs, n = xs.length) {
  let s = 0;
  for (let i = 0; i < n; i++) s += xs[i];
  return s;
}

/** NumPy's np.sum on a contiguous float64 array: pairwise summation with an 8-way unrolled base case of up to
 *  128 elements (numpy/_core/src/umath/loops_utils.h.src), over the whole array, added to the identity 0. */
export function numpySum(xs, n = xs.length) {
  return 0 + pairwise(xs, 0, n);
}
function pairwise(a, lo, n) {
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r += a[lo + i];
    return r;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
}

/** Kahan's compensated summation. */
export function kahan(xs, n = xs.length) {
  let s = 0;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const y = xs[i] - c;
    const t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

/** Neumaier's variant (what Python 3.12's sum() of floats uses). */
export function neumaier(xs, n = xs.length) {
  let s = 0;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const v = xs[i];
    const t = s + v;
    c += Math.abs(s) >= Math.abs(v) ? (s - t) + v : (v - t) + s;
    s = t;
  }
  return s + c;
}

// --- the exact sum ---------------------------------------------------------------------------------------------
const dv = new DataView(new ArrayBuffer(8));
/** A finite double as {M, E} with value M × 2^E (M an integer, possibly negative). */
function parts(x) {
  const bits = bitsOfNumber(x);
  const e = Number((bits >> 52n) & 0x7ffn);
  let M = bits & 0xfffffffffffffn;
  let E;
  if (e === 0) E = -1074;
  else {
    M |= 1n << 52n;
    E = e - 1075;
  }
  if (bits >> 63n) M = -M;
  return { M, E };
}

/** The exact sum of the first n doubles, as a rational {n, d} (d a power of two). Inputs must be finite. */
export function exactSum(xs, n = xs.length) {
  let minE = Infinity;
  const ps = new Array(n);
  for (let i = 0; i < n; i++) {
    ps[i] = parts(xs[i]);
    if (ps[i].M !== 0n && ps[i].E < minE) minE = ps[i].E;
  }
  if (minE === Infinity) return { n: 0n, d: 1n };
  let S = 0n;
  for (let i = 0; i < n; i++) if (ps[i].M !== 0n) S += ps[i].M << BigInt(ps[i].E - minE);
  return minE >= 0 ? { n: S << BigInt(minE), d: 1n } : { n: S, d: 1n << BigInt(-minE) };
}

/** The exact sum rounded once to the nearest double: what math.fsum returns (barring intermediate overflow). */
export function roundedExact(sum) {
  if (sum.n === 0n) return 0;
  const neg = sum.n < 0n;
  const r = roundTo(FORMATS.binary64, { neg, mag: neg ? { n: -sum.n, d: sum.d } : sum }, "RNE");
  if (r.inf) return r.inf * Infinity;
  dv.setBigUint64(0, r.bits);
  return dv.getFloat64(0);
}

/** A running exact sum, for checkpoints along a long race (each term is k / 2^53 or any double). */
export class ExactAccumulator {
  constructor() {
    this.S = 0n;
    this.E = 0; // S × 2^E is the exact sum so far
  }
  add(x) {
    const { M, E } = parts(x);
    if (M === 0n) return;
    if (E < this.E) {
      this.S <<= BigInt(this.E - E);
      this.E = E;
    }
    this.S += M << BigInt(E - this.E);
  }
  get value() {
    return this.E >= 0 ? { n: this.S << BigInt(this.E), d: 1n } : { n: this.S, d: 1n << BigInt(-this.E) };
  }
}

/** A small seeded generator (mulberry32), so a race can be replayed: next53() gives k in [0, 2^53) as a number,
 *  from two 32-bit draws (26 + 27 bits), and k / 2^53 is then an exact double in [0, 1). */
export function seeded(seed) {
  let a = seed | 0;
  const u32 = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };
  return { u32, next53: () => (u32() >>> 6) * 134217728 + (u32() >>> 5) };
}
