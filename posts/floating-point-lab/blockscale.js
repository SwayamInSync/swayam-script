// Block-scaled quantization, exactly: a port of verify/blockscale.py, tested against its vectors
// (verify/test_blockscale.mjs). Values are exact rationals {n, d}; every FP32 step is rounded exactly as binary32.
//
// - Inputs are first rounded to float32.
// - MX (OCP MX v1.0 §6.3): X = largest power of two <= max|V|, divided by the largest power of two in the element
//   format, clamped to E8M0's smallest value 2^-127; P = V / X rounded to nearest-even, clamped to the element's
//   max. The "up" rule instead takes the smallest power of two X with max|V| / X <= the element's max.
// - NVFP4 (NVIDIA, "Pretraining LLMs with NVFP4", App. B): an FP32 tensor scale, an E4M3 scale per 16 elements,
//   FP4 E2M1 elements; decoded as f32(f32(q * s_b) * s_dec).
// - Per-tensor FP8: one FP32 scale 448 / max|x|; decoded by dividing by it.

import { FORMATS, binaryFormat, roundTo, floorLog2, rat, cmp, div, mul, pow2 } from "./fpbig.js";

export const FP4 = binaryFormat({ name: "FP4 E2M1", e: 2, m: 1, specials: "none" });
export const E4M3 = FORMATS.e4m3;
const B32 = FORMATS.binary32;
// largest power of two representable in each element type (OCP MX v1.0 §6.3, step 1)
const EMAX_POW2 = new Map([[FP4, 2], [E4M3, 8]]);

const ZERO = rat(0);
const isZero = (x) => x.n === 0n;
const abs = (x) => (x.n < 0n ? { n: -x.n, d: x.d } : x);
const split = (x) => ({ neg: x.n < 0n, mag: abs(x) });
const maxAbs = (xs) => xs.reduce((m, v) => (cmp(abs(v), m) > 0 ? abs(v) : m), ZERO);

/** Round to binary32 (RNE). Returns null on overflow. */
export function f32(x) {
  const r = roundTo(B32, split(x), "RNE");
  return r.value;
}
const F32_MAX = B32.maxFinite;
const f32c = (x) => f32(x) ?? F32_MAX; // clamped, for the scale computations (see blockscale.py)

/** Round to an element format: nearest, ties to even, saturating at its max. */
export const quantize = (x, fmt) => (isZero(x) ? ZERO : roundTo(fmt, split(x), "RNE").value);

function toF32(values) {
  return values.map((v) => {
    const r = f32(v);
    if (r === null) throw new RangeError("input outside float32's range");
    return r;
  });
}

/** One MX block: {X, P, decoded}. */
export function mxBlock(values, fmt, rule = "spec") {
  const vs = toF32(values);
  const amax = maxAbs(vs);
  let X;
  if (isZero(amax)) X = rat(1);
  else if (rule === "spec") X = pow2(floorLog2(amax) - EMAX_POW2.get(fmt));
  else {
    const vmax = fmt.maxFinite;
    let k = floorLog2(amax) - floorLog2(vmax);
    while (cmp(div(amax, pow2(k)), vmax) > 0) k++;
    while (cmp(div(amax, pow2(k - 1)), vmax) <= 0) k--;
    X = pow2(k);
  }
  if (cmp(X, pow2(-127)) < 0) X = pow2(-127);
  const P = vs.map((v) => quantize(div(v, X), fmt));
  return { X, P, decoded: P.map((p) => mul(X, p)) };
}

export function mx(values, fmt, k = 32, rule = "spec") {
  const out = [];
  const scales = [];
  for (let i = 0; i < values.length; i += k) {
    const b = mxBlock(values.slice(i, i + k), fmt, rule);
    out.push(...b.decoded);
    scales.push(b.X);
  }
  return { out, scales };
}

export function nvfp4(values, k = 16) {
  const xs = toF32(values);
  const amax = maxAbs(xs);
  if (isZero(amax)) return { out: xs.map(() => ZERO), scales: [], sEnc: null };
  const sEnc = f32c(div(rat(6 * 448), amax)); // eq. (1)
  const sDec = f32c(div(rat(1), sEnc));
  const out = [];
  const scales = [];
  for (let i = 0; i < xs.length; i += k) {
    const blk = xs.slice(i, i + k);
    const amaxB = maxAbs(blk);
    const sDecB = f32c(div(amaxB, rat(6))); // eq. (2)
    const sB = quantize(f32c(mul(sDecB, sEnc)), E4M3); // eq. (3)
    scales.push(sB);
    if (isZero(sB)) {
      out.push(...blk.map(() => ZERO));
      continue;
    }
    const sEncB = f32c(div(rat(1), f32c(mul(sB, sDec))));
    for (const v of blk) {
      const q = quantize(f32c(mul(v, sEncB)), FP4); // eq. (4)
      out.push(f32c(mul(f32c(mul(q, sB)), sDec)));
    }
  }
  return { out, scales, sEnc };
}

export function perTensorFP8(values, fmt = E4M3) {
  const xs = toF32(values);
  const amax = maxAbs(xs);
  if (isZero(amax)) return { out: xs.map(() => ZERO), scales: [] };
  const sEnc = f32c(div(fmt.maxFinite, amax));
  return { out: xs.map((v) => f32c(div(quantize(f32c(mul(v, sEnc)), fmt), sEnc))), scales: [sEnc] };
}

export const SCHEMES = {
  pt8: { label: "per-tensor FP8 E4M3", bits: "8 (+ one FP32 per tensor)", block: 0, run: (v) => perTensorFP8(v) },
  mx8: { label: "MXFP8 (E4M3, blocks of 32)", bits: "8.25", block: 32, run: (v) => mx(v, E4M3, 32) },
  mx4: { label: "MXFP4 (blocks of 32, the spec's scale)", bits: "4.25", block: 32, run: (v) => mx(v, FP4, 32) },
  mx4up: { label: "MXFP4 (blocks of 32, scale rounded up)", bits: "4.25", block: 32, run: (v) => mx(v, FP4, 32, "up") },
  nv4: { label: "NVFP4 (blocks of 16, E4M3 scales)", bits: "4.5 (+ one FP32 per tensor)", block: 16, run: (v) => nvfp4(v) },
};
