// Exact minifloat arithmetic for the Floating Point Lab widgets.
//
// Every value is an exact rational {n, d} with BigInt parts and d > 0, so the
// widgets never show a number that was rounded by JavaScript's own doubles.

const babs = (a) => (a < 0n ? -a : a);

function gcd(a, b) {
  a = babs(a);
  b = babs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

export function rat(n, d = 1n) {
  n = BigInt(n);
  d = BigInt(d);
  if (d === 0n) throw new RangeError("zero denominator");
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const g = gcd(n, d) || 1n;
  return { n: n / g, d: d / g };
}

export const cmp = (a, b) => {
  const l = a.n * b.d;
  const r = b.n * a.d;
  return l < r ? -1 : l > r ? 1 : 0;
};
export const add = (a, b) => rat(a.n * b.d + b.n * a.d, a.d * b.d);
export const sub = (a, b) => rat(a.n * b.d - b.n * a.d, a.d * b.d);
export const div = (a, b) => rat(a.n * b.d, a.d * b.n);
export const neg = (a) => ({ n: -a.n, d: a.d });
export const rabs = (a) => ({ n: babs(a.n), d: a.d });

export function pow2(k) {
  return k >= 0 ? { n: 1n << BigInt(k), d: 1n } : { n: 1n, d: 1n << BigInt(-k) };
}

/** Approximate a rational as a JS number (display/geometry only, never for results). */
export function toNumber(r) {
  if (r.n === 0n) return 0;
  // scale big parts down first so huge BigInts don't become Infinity / 0
  const shift = Math.max(0, Math.max(r.n.toString(2).length, r.d.toString(2).length) - 1000);
  const s = BigInt(shift);
  return Number(r.n >> s) / Number(r.d >> s);
}

/**
 * Parse "0.1", "-2.75e-3", "1/3", "+.5", "-0".
 * Returns {neg, mag} with mag an exact non-negative rational, or null.
 */
export function parseNumber(str) {
  const m = /^\s*([+-])?(?:(\d+)\s*\/\s*(\d+)|(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?)\s*$/.exec(str);
  if (!m) return null;
  const negSign = m[1] === "-";
  if (m[2] !== undefined) {
    if (BigInt(m[3]) === 0n) return null;
    return { neg: negSign, mag: rat(BigInt(m[2]), BigInt(m[3])) };
  }
  const ip = m[4] || "";
  const fp = m[5] || "";
  if (!ip && !fp) return null;
  const ex = m[6] ? parseInt(m[6], 10) : 0;
  if (Math.abs(ex) > 5000) return null;
  let n = BigInt((ip || "0") + fp);
  let d = 10n ** BigInt(fp.length);
  if (ex >= 0) n *= 10n ** BigInt(ex);
  else d *= 10n ** BigInt(-ex);
  return { neg: negSign, mag: rat(n, d) };
}

/**
 * Exact decimal expansion of a rational. Terminates for dyadic values; for others
 * (like 1/3) stops after maxSig significant digits and reports exact=false.
 */
export function toDecimal(r, maxSig = 80) {
  const negative = r.n < 0n;
  let n = babs(r.n);
  const d = r.d;
  const ip = n / d;
  let rem = n % d;
  let digits = "";
  let sig = ip === 0n ? 0 : ip.toString().length;
  while (rem !== 0n && sig < maxSig) {
    rem *= 10n;
    const q = rem / d;
    rem %= d;
    digits += q.toString();
    if (sig > 0 || q !== 0n) sig++;
  }
  const text = (negative ? "-" : "") + ip.toString() + (digits ? "." + digits : "");
  return { text, exact: rem === 0n };
}

/** Short label for a grid cell: exact when it fits, otherwise ≈ with 3 significant digits. */
export function shortLabel(r, maxChars = 9) {
  const { text, exact } = toDecimal(r, 40);
  if (exact && text.length <= maxChars) return text;
  const x = toNumber(r);
  const s = x.toPrecision(3);
  return "≈" + (s.includes("e") ? s.replace(/e([+-])(\d+)/, (_, sg, ex) => "e" + (sg === "-" ? "-" : "") + ex) : s);
}

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

/**
 * An 8-bit-style binary format: 1 sign bit, e exponent bits, m mantissa bits.
 *   specials: "ieee" -> top exponent code is ±∞ (M=0) and NaN (M≠0)
 *             "e4m3" -> no ∞; only S.1…1.1…1 is NaN (OCP FP8 E4M3 rule)
 *             "none" -> every code is a finite number (OCP FP6/FP4 rule)
 *   subnormals: false -> hypothetical flush-to-zero format: E=0 codes all mean 0
 */
export function makeFormat({ e, m, bias = (1 << (e - 1)) - 1, specials = "ieee", subnormals = true }) {
  const top = (1 << e) - 1;
  const mMask = (1 << m) - 1;
  const bits = 1 + e + m;
  const signBit = 1 << (e + m);
  const emax = specials === "ieee" ? top - 1 - bias : top - bias;
  const emin = 1 - bias;

  function decode(code) {
    const s = (code >> (e + m)) & 1;
    const E = (code >> m) & top;
    const M = code & mMask;
    const base = { code, s, E, M };
    if (specials === "ieee" && E === top) {
      return M === 0 ? { ...base, cls: "inf", value: null } : { ...base, cls: "nan", value: null };
    }
    if (specials === "e4m3" && E === top && M === mMask) return { ...base, cls: "nan", value: null };
    let mag;
    let cls;
    if (E === 0) {
      if (M === 0) {
        mag = rat(0);
        cls = "zero";
      } else if (!subnormals) {
        mag = rat(0);
        cls = "flushed";
      } else {
        mag = rat(BigInt(M) * pow2(emin - m).n, pow2(emin - m).d);
        cls = "sub";
      }
    } else {
      const p = pow2(E - bias - m);
      mag = rat(BigInt((1 << m) + M) * p.n, p.d);
      cls = "normal";
    }
    return { ...base, cls, value: s ? neg(mag) : mag };
  }

  const all = [];
  for (let c = 0; c < 1 << bits; c++) all.push(decode(c));

  // distinct non-negative finite values, ascending, each with its canonical (positive) code
  const positives = all
    .filter((x) => x.s === 0 && x.value !== null && x.cls !== "flushed")
    .sort((a, b) => cmp(a.value, b.value));

  const max = positives[positives.length - 1];
  const minNormal = pow2(emin);
  const minSub = subnormals && m > 0 ? pow2(emin - m) : null;
  const infCode = specials === "ieee" ? top << m : null;

  const count = (cls) => all.filter((x) => x.cls === cls).length;

  return {
    e, m, bias, specials, subnormals, bits, top, mMask, signBit, emax, emin,
    all, positives, max, minNormal, minSub, infCode, decode,
    counts: {
      zero: count("zero"), sub: count("sub"), normal: count("normal"),
      inf: count("inf"), nan: count("nan"), flushed: count("flushed"),
    },
  };
}

/**
 * Round an exact real x = {neg, mag} into fmt with roundTiesToEven.
 * Overflow: "ieee" -> ±∞; "e4m3"/"none" -> saturate to ±max (OCP SAT behaviour;
 * E4M3 in OVF mode would give NaN instead). Flush formats send |x| < 2^emin to 0.
 */
export function roundRNE(fmt, x) {
  const sign = x.neg ? fmt.signBit : 0;
  const P = fmt.positives;
  const max = P[P.length - 1];
  const out = (code, kind, extra = {}) => ({ code: code | sign, kind, ...extra });

  if (x.mag.n === 0n) return out(0, "exact");
  if (!fmt.subnormals && cmp(x.mag, fmt.minNormal) < 0) return out(0, "flushed", { lo: P[0], hi: P[1] });

  if (cmp(x.mag, max.value) > 0) {
    const half = pow2(fmt.emax - fmt.m - 1); // half an ulp in the top binade
    const c = cmp(sub(x.mag, max.value), half);
    const overflow = c > 0 || (c === 0 && (max.code & 1) === 1);
    if (!overflow) return out(max.code, "rounded", { lo: max, hi: null });
    if (fmt.specials === "ieee") return out(fmt.infCode, "overflow-inf", { lo: max, hi: null });
    return out(max.code, "overflow-sat", { lo: max, hi: null });
  }

  // binary search: largest i with P[i] <= |x|
  let lo = 0;
  let hi = P.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (cmp(P[mid].value, x.mag) <= 0) lo = mid;
    else hi = mid - 1;
  }
  const L = P[lo];
  if (cmp(L.value, x.mag) === 0) return out(L.code, "exact", { lo: L, hi: L });
  const H = P[lo + 1];
  const c = cmp(sub(x.mag, L.value), sub(H.value, x.mag));
  const tie = c === 0;
  const pick = c < 0 ? L : c > 0 ? H : (L.code & 1) === 0 ? L : H;
  const kind = pick.code === 0 ? "underflow-zero" : "rounded";
  return out(pick.code, kind, { lo: L, hi: H, tie });
}

/** Next representable neighbours (non-negative side) of a positive code, for ulp display. */
export function neighbours(fmt, posCode) {
  const P = fmt.positives;
  const i = P.findIndex((p) => p.code === posCode);
  if (i < 0) return { down: null, up: null };
  return { down: i > 0 ? P[i - 1] : null, up: i < P.length - 1 ? P[i + 1] : null };
}
