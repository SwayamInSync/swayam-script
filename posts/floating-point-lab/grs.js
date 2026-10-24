// A hardware-style floating-point adder with a choice of extra bits (guard, round, sticky), in exact BigInt
// arithmetic. A line-for-line port of verify/grs.py's add_hw, plus a trace of every step for the adder widget.
// Numbers are {s, e, M}: sign, exponent and a p-bit significand M in [2^(p-1), 2^p), value M × 2^(e-p+1).
// The exponent range is unbounded: this is about alignment and rounding, not overflow or underflow.

import { rat, cmp, sub, div, pow2 } from "./fpcore.js";

export const MODES = ["RNE", "RTZ", "RTP", "RTN"];
export const CONFIGS = {
  none: { extra: 0, sticky: false, label: "no extra bits" },
  G: { extra: 1, sticky: false, label: "guard only" },
  GR: { extra: 2, sticky: false, label: "guard + round" },
  GS: { extra: 2, sticky: true, label: "guard + sticky" },
  GRS: { extra: 3, sticky: true, label: "guard + round + sticky" },
};

const half = rat(1, 2);
const mulR = (a, b) => rat(a.n * b.n, a.d * b.d);

/** The exact value of {s, e, M} at precision p. */
export function value({ s, e, M }, p) {
  const v = mulR(rat(M), pow2(e - p + 1));
  return s ? { n: -v.n, d: v.d } : v;
}

/** Correctly round a nonzero rational to p significant bits (unbounded exponent). */
export function roundExact(x, p, mode) {
  const s = x.n < 0n;
  const a = s ? { n: -x.n, d: x.d } : x;
  let e = a.n.toString(2).length - a.d.toString(2).length;
  if (cmp(pow2(e), a) > 0) e -= 1;
  else if (cmp(pow2(e + 1), a) <= 0) e += 1;
  const scaled = div(a, pow2(e - p + 1)); // in [2^(p-1), 2^p)
  let q = scaled.n / scaled.d;
  const r = sub(scaled, rat(q));
  const rc = cmp(r, half);
  const up = {
    RNE: rc > 0 || (rc === 0 && (q & 1n) === 1n),
    RNA: rc >= 0,
    RTZ: false,
    RTP: r.n !== 0n && !s,
    RTN: r.n !== 0n && s,
  }[mode];
  if (up) q += 1n;
  if (q === 1n << BigInt(p)) {
    q >>= 1n;
    e += 1;
  }
  return { s, e, M: q };
}

/**
 * Hardware-style a + b keeping `extra` bits beyond the p-bit significand; if `sticky`, the last of them is a
 * sticky bit (the OR of everything shifted past it). Returns {result, trace}; result is null for an exact zero.
 */
export function addHW(a, b, p, mode, extra, sticky) {
  const P = BigInt(p);
  const X = BigInt(extra);
  const trace = { p, extra, sticky, mode };
  // make |a| >= |b|
  if (a.e < b.e || (a.e === b.e && a.M < b.M)) [a, b] = [b, a];
  trace.big = a;
  trace.small = b;
  const d = a.e - b.e;
  trace.d = d;

  const shr = (x, n) => {
    const N = BigInt(n);
    const lost = n ? x & ((1n << N) - 1n) : 0n;
    const y = x >> N;
    return sticky && lost ? y | 1n : y;
  };

  const A = a.M << X;
  const B0 = b.M << X;
  trace.A = A;
  trace.B0 = B0;
  trace.lost = d ? B0 & ((1n << BigInt(d)) - 1n) : 0n; // the bits of B that fall off the register
  const B = shr(B0, d);
  trace.B = B;
  const subtract = a.s !== b.s;
  trace.subtract = subtract;
  let S = subtract ? A - B : A + B;
  trace.S0 = S;
  if (S === 0n) {
    trace.zero = true;
    return { result: null, trace };
  }
  let e = a.e;
  const top = 1n << (P - 1n + X);
  trace.shifts = [];
  while (S >= 2n * top) {
    // carry out: shift right (the bit shifted out goes into the sticky position, if there is one)
    trace.shifts.push({ dir: "right", lostBit: S & 1n });
    S = shr(S, 1);
    e += 1;
  }
  while (S < top) {
    // cancellation: shift left, zeros come in
    trace.shifts.push({ dir: "left" });
    S <<= 1n;
    e -= 1;
  }
  trace.S = S;
  let q = S >> X;
  const rem = S & ((1n << X) - 1n);
  const halfX = extra ? 1n << (X - 1n) : 0n;
  trace.q = q;
  trace.rem = rem;
  const neg = a.s;
  const up = {
    RNE: extra > 0 && (rem > halfX || (rem === halfX && (q & 1n) === 1n)),
    RTZ: false,
    RTP: rem !== 0n && !neg,
    RTN: rem !== 0n && neg,
  }[mode];
  trace.up = up;
  trace.tie = extra > 0 && rem === halfX;
  if (up) q += 1n;
  if (q === 1n << P) {
    q >>= 1n;
    e += 1;
    trace.roundCarry = true;
  }
  return { result: { s: neg, e, M: q }, trace };
}

/** Is the hardware result the correctly rounded exact sum? */
export function isCorrect(a, b, p, mode, result) {
  const exact = (() => {
    const va = value(a, p);
    const vb = value(b, p);
    return rat(va.n * vb.d + vb.n * va.d, va.d * vb.d);
  })();
  if (exact.n === 0n) return result === null;
  if (result === null) return false;
  return cmp(value(result, p), value(roundExact(exact, p, mode), p)) === 0;
}
