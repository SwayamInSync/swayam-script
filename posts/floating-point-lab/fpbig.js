// Arbitrary binary floating-point formats, in exact BigInt arithmetic.
//
// Values are exact rationals {n, d} (from fpcore.js). A format is described by its exponent width e,
// trailing-significand width m and special-value rule; formats up to binary256 work. Nothing here is
// ever rounded by JavaScript's own doubles: JS numbers enter and leave only through their exact bits.

import { rat, cmp, add, sub, div, neg, rabs, pow2, toDecimal } from "./fpcore.js";

const one = rat(1);
const half = rat(1, 2);
const mul = (a, b) => rat(a.n * b.n, a.d * b.d);
const isZero = (a) => a.n === 0n;

/** floor(log2(a)) for a positive rational, exactly. */
export function floorLog2(a) {
  let e = a.n.toString(2).length - a.d.toString(2).length;
  // 2^e <= a < 2^(e+1)  <=>  adjust e
  const ge = (k) => (k >= 0 ? a.n >= (a.d << BigInt(k)) : (a.n << BigInt(-k)) >= a.d);
  while (!ge(e)) e--;
  while (ge(e + 1)) e++;
  return e;
}

/** floor of a rational, as a BigInt. */
const floorRat = (a) => {
  const q = a.n / a.d;
  return a.n < 0n && q * a.d !== a.n ? q - 1n : q;
};

/**
 * A binary format: 1 sign bit (if hasSign), e exponent bits, m trailing-significand bits.
 * specials: "ieee" (top exponent = ±∞ / NaN), "e4m3" (OCP: only S.1…1.1…1 is NaN), "none" (all finite).
 */
export function binaryFormat({ name = "", e, m, specials = "ieee", hasSign = true, bias }) {
  const B = bias ?? (1 << (e - 1)) - 1;
  const top = (1 << e) - 1;
  const emin = 1 - B;
  const emaxField = specials === "ieee" ? top - 1 : top;
  const mmax = (1n << BigInt(m)) - 1n;
  const maxSig = specials === "e4m3" ? mmax - 1n : mmax; // e4m3: the all-ones mantissa in the top row is NaN
  const emax = emaxField - B;
  const maxFinite = mul(rat((1n << BigInt(m)) + maxSig, 1n << BigInt(m)), pow2(emax));
  return {
    name, e, m, p: m + 1, bias: B, specials, hasSign, top, emin, emax,
    bits: (hasSign ? 1 : 0) + e + m,
    maxFinite,
    minNormal: pow2(emin),
    minSub: m > 0 ? pow2(emin - m) : pow2(emin),
    epsilon: pow2(-m),
  };
}

export const FORMATS = {
  binary16: binaryFormat({ name: "binary16", e: 5, m: 10 }),
  bfloat16: binaryFormat({ name: "bfloat16", e: 8, m: 7 }),
  binary32: binaryFormat({ name: "binary32", e: 8, m: 23 }),
  binary64: binaryFormat({ name: "binary64", e: 11, m: 52 }),
  binary128: binaryFormat({ name: "binary128", e: 15, m: 112 }),
  binary256: binaryFormat({ name: "binary256", e: 19, m: 236 }),
  e4m3: binaryFormat({ name: "FP8 E4M3", e: 4, m: 3, specials: "e4m3" }),
  e5m2: binaryFormat({ name: "FP8 E5M2", e: 5, m: 2 }),
  toy: binaryFormat({ name: "toy 1·4·3", e: 4, m: 3 }),
};

/** Decode a bit pattern (BigInt). Returns {s, E, M, cls, value} with value null for ∞/NaN. */
export function decode(fmt, bits) {
  bits = BigInt(bits);
  const s = fmt.hasSign ? Number((bits >> BigInt(fmt.e + fmt.m)) & 1n) : 0;
  const E = Number((bits >> BigInt(fmt.m)) & BigInt(fmt.top));
  const M = bits & ((1n << BigInt(fmt.m)) - 1n);
  const base = { s, E, M };
  if (fmt.specials === "ieee" && E === fmt.top) return { ...base, cls: M === 0n ? "inf" : "nan", value: null };
  if (fmt.specials === "e4m3" && E === fmt.top && M === (1n << BigInt(fmt.m)) - 1n) return { ...base, cls: "nan", value: null };
  let mag;
  let cls;
  if (E === 0) {
    mag = mul(rat(M, 1n << BigInt(fmt.m)), pow2(fmt.emin));
    cls = M === 0n ? "zero" : "sub";
  } else {
    mag = mul(rat((1n << BigInt(fmt.m)) + M, 1n << BigInt(fmt.m)), pow2(E - fmt.bias));
    cls = "normal";
  }
  return { ...base, cls, value: s ? neg(mag) : mag };
}

/** Encode a finite magnitude already on the format's grid (no rounding) plus a sign. */
export function encode(fmt, negative, mag) {
  const sbit = negative && fmt.hasSign ? 1n << BigInt(fmt.e + fmt.m) : 0n;
  if (isZero(mag)) return sbit;
  const ex = Math.max(floorLog2(mag), fmt.emin);
  const q = pow2(ex - fmt.m);
  const n = div(mag, q);
  if (n.d !== 1n) throw new RangeError("value is not on the format's grid");
  if (ex === fmt.emin && n.n < 1n << BigInt(fmt.m)) return sbit | n.n; // subnormal
  const E = BigInt(ex + fmt.bias);
  return sbit | (E << BigInt(fmt.m)) | (n.n - (1n << BigInt(fmt.m)));
}

export const MODES = {
  RNE: "nearest, ties to even",
  RNA: "nearest, ties away from zero",
  RTP: "toward +∞",
  RTN: "toward −∞",
  RTZ: "toward zero",
  RTO: "to odd",
};

/**
 * Bracket a non-negative exact magnitude: lo <= mag <= hi on the format's grid, ignoring the top of the
 * range (hi can be above maxFinite). Returns {lo, hi, exact, frac} with frac = (mag − lo) / (hi − lo).
 */
export function bracket(fmt, mag) {
  if (isZero(mag)) return { lo: rat(0), hi: rat(0), exact: true, frac: rat(0), q: fmt.minSub };
  const ex = Math.max(floorLog2(mag), fmt.emin);
  const q = pow2(ex - fmt.m);
  const k = floorRat(div(mag, q));
  const lo = mul(rat(k), q);
  const exact = cmp(lo, mag) === 0;
  const hi = exact ? lo : mul(rat(k + 1n), q);
  return { lo, hi, exact, k, q, frac: exact ? rat(0) : div(sub(mag, lo), q) };
}

/**
 * Round the exact real x = {neg, mag} into fmt in rounding mode `mode`, with IEEE 754 overflow rules
 * for "ieee" formats (§7.4) and saturation (OCP SAT) for "e4m3"/"none" formats.
 * Returns {value, bits, cls: "exact"|"rounded"|"overflow"|"underflow-zero", inf}.
 */
export function roundTo(fmt, x, mode = "RNE") {
  const { neg: negative, mag } = x;
  const b = bracket(fmt, mag);
  let pick;
  if (b.exact) pick = b.lo;
  else {
    const c = cmp(b.frac, half);
    const kOdd = (b.k & 1n) === 1n;
    const up = {
      RNE: c > 0 || (c === 0 && kOdd),
      RNA: c >= 0,
      RTZ: false,
      RTP: !negative,
      RTN: negative,
      RTO: !kOdd, // pick whichever neighbour has an odd last bit
    }[mode];
    pick = up ? b.hi : b.lo;
  }
  const signed = (v) => (negative ? neg(v) : v);
  if (cmp(pick, fmt.maxFinite) > 0) {
    // overflow: IEEE 754 §7.4 (rounded with unbounded exponent beyond the largest finite number)
    const toInf =
      fmt.specials === "ieee" &&
      ({ RNE: true, RNA: true, RTZ: false, RTP: !negative, RTN: negative, RTO: false }[mode]);
    if (toInf) {
      const bits = (negative ? 1n << BigInt(fmt.e + fmt.m) : 0n) | (BigInt(fmt.top) << BigInt(fmt.m));
      return { value: null, inf: negative ? -1 : 1, bits, cls: "overflow" };
    }
    return { value: signed(fmt.maxFinite), bits: encode(fmt, negative, fmt.maxFinite), cls: "overflow" };
  }
  const cls = b.exact ? "exact" : isZero(pick) ? "underflow-zero" : "rounded";
  return { value: signed(pick), bits: encode(fmt, negative, pick), cls };
}

/** Probability that stochastic rounding picks the upper neighbour (by magnitude). */
export function srUpProbability(fmt, mag) {
  const b = bracket(fmt, mag);
  return b.exact ? rat(0) : b.frac;
}

// --- binary64 <-> JS numbers, through their bits ------------------------------------------------------
const dv = new DataView(new ArrayBuffer(8));
export function bitsOfNumber(x) {
  dv.setFloat64(0, x);
  return dv.getBigUint64(0);
}
export function numberOfBits(bits) {
  dv.setBigUint64(0, BigInt.asUintN(64, BigInt(bits)));
  return dv.getFloat64(0);
}
/** Exact rational value of a finite JS number. */
export function exactOfNumber(x) {
  if (!Number.isFinite(x)) return null;
  return decode(FORMATS.binary64, bitsOfNumber(x)).value;
}

/** floor(log10(a)) for a positive rational, exactly. */
export function log10floor(a) {
  const tenPow = (k) => (k >= 0 ? rat(10n ** BigInt(k)) : rat(1n, 10n ** BigInt(-k)));
  let k = a.n.toString().length - a.d.toString().length;
  const ge = (j) => cmp(a, tenPow(j)) >= 0;
  while (!ge(k)) k--;
  while (ge(k + 1)) k++;
  return k;
}

/** The decimal with k significant digits nearest to the positive rational v (ties to an even last digit,
 *  as C's printf does with the exact value). Returns {c, exp, value} with value = c × 10^exp. */
export function nearestDecimal(v, k) {
  let exp = log10floor(v) - k + 1;
  const unit = (e) => (e >= 0 ? rat(10n ** BigInt(e)) : rat(1n, 10n ** BigInt(-e)));
  let q = div(v, unit(exp));
  let c = floorRat(q);
  const r = sub(q, rat(c));
  const h = cmp(r, half);
  if (h > 0 || (h === 0 && (c & 1n) === 1n)) c += 1n;
  if (c.toString().length > k) {
    // rounded up to the next power of ten: 9.99… -> 10.0…, still k digits at the next exponent
    c /= 10n;
    exp += 1;
  }
  return { c, exp, value: mul(rat(c), unit(exp)) };
}

/**
 * The interval of reals that round (to nearest, ties to even) to the positive binary64 x, and the shortest
 * decimal inside it (ECMAScript Number::toString's choice: fewest digits, then closest to x, then even).
 * Returns {x, lo, hi, inclusive, candidates: [{k, c, exp, value, inside}], best, bestAll} where candidates[k-1]
 * is the k-digit decimal nearest x and bestAll lists every decimal with best.k digits inside the interval.
 */
export function shortestInterval(x) {
  const fmt = FORMATS.binary64;
  const bits = bitsOfNumber(x);
  const d = decode(fmt, bits);
  if (d.value === null || d.s || isZero(d.value)) throw new RangeError("positive finite numbers only");
  const v = d.value;
  const up = decode(fmt, bits + 1n).value; // may be null at the top: treat as 2^1024
  const vUp = up ?? pow2(1024);
  const vDown = d.E === 0 && d.M === 1n ? rat(0) : decode(fmt, bits - 1n).value;
  const lo = div(add(v, vDown), rat(2));
  const hi = div(add(v, vUp), rat(2));
  const inclusive = (d.M & 1n) === 0n;
  const inside = (c) => (inclusive ? cmp(c, lo) >= 0 && cmp(c, hi) <= 0 : cmp(c, lo) > 0 && cmp(c, hi) < 0);
  const tenPow = (k) => (k >= 0 ? rat(10n ** BigInt(k)) : rat(1n, 10n ** BigInt(-k)));
  const log10floor = (a) => {
    let k = Math.floor(Math.log10(Number(a.n) / Number(a.d)) || 0);
    if (!Number.isFinite(k)) k = 0;
    const ge = (j) => cmp(a, tenPow(j)) >= 0;
    while (!ge(k)) k--;
    while (ge(k + 1)) k++;
    return k;
  };
  const candidates = [];
  let best = null;
  let bestAll = [];
  for (let k = 1; k <= 17 && !best; k++) {
    const found = [];
    for (const top of new Set([log10floor(lo.n === 0n ? v : lo), log10floor(hi)])) {
      const step = tenPow(top - k + 1);
      const cLo = floorRat(div(lo, step));
      for (let c = cLo; c <= cLo + 2n + floorRat(div(sub(hi, lo), step)); c++) {
        if (c <= 0n) continue;
        const val = mul(rat(c), step);
        if (inside(val) && c.toString().length <= k) found.push({ c, exp: top - k + 1, value: val });
      }
    }
    // the decimal with k significant digits nearest to x, for display even when it misses the interval
    const nd = nearestDecimal(v, k);
    candidates.push({ k, ...nd, inside: inside(nd.value) });
    if (found.length) {
      const seen = new Set();
      bestAll = found.filter((f) => {
        const key = `${f.value.n}/${f.value.d}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).sort((a, b2) => cmp(a.value, b2.value));
      found.sort((a, b2) => {
        const da = rabs(sub(a.value, v));
        const db = rabs(sub(b2.value, v));
        return cmp(da, db) || (a.c % 2n === 0n ? -1 : 1);
      });
      best = { k, ...found[0] };
    }
  }
  return { x, v, lo, hi, inclusive, candidates, best, bestAll };
}

/** Decimal string for a value like {c, exp}: c × 10^exp, written plainly or in e-notation. */
export function decimalString(c, exp) {
  const s = c.toString().replace(/0+$/, "") || "0";
  const exp2 = exp + (c.toString().length - s.length);
  const n = s.length;
  const pointPos = n + exp2; // digits before the decimal point
  if (pointPos > 21 || pointPos < -5) {
    const mant = s.length > 1 ? s[0] + "." + s.slice(1) : s;
    const e10 = pointPos - 1;
    return mant + "e" + (e10 >= 0 ? "+" : "-") + Math.abs(e10);
  }
  if (pointPos <= 0) return "0." + "0".repeat(-pointPos) + s;
  if (pointPos >= n) return s + "0".repeat(pointPos - n);
  return s.slice(0, pointPos) + "." + s.slice(pointPos);
}

export { toDecimal, rat, cmp, add, sub, div, neg, rabs, pow2, mul, one };
