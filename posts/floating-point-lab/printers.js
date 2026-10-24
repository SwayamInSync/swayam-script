// How different languages print a double, computed exactly from its bits (no host printf involved).
// Tested against Python's repr and %-formatting and C's printf (verify/test_printers.mjs).

import { FORMATS, decode, bitsOfNumber, shortestInterval, nearestDecimal, decimalString } from "./fpbig.js";

const digitsOf = (c) => {
  const s = c.toString().replace(/0+$/, "");
  return { s, exp: c.toString().length - s.length };
};

/** Shortest round-trip digits of |x| (x finite, nonzero): {s, point} where |x| ≈ 0.s × 10^point. */
function shortestDigits(x) {
  const r = shortestInterval(Math.abs(x));
  const { s, exp } = digitsOf(r.best.c);
  return { s, point: s.length + r.best.exp + exp };
}

/** JavaScript's String(x) (ECMAScript Number::toString), from the shortest digits. */
export function jsString(x) {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x > 0 ? "Infinity" : "-Infinity";
  if (x === 0) return "0"; // −0 prints as "0" too
  const r = shortestInterval(Math.abs(x));
  return (x < 0 ? "-" : "") + decimalString(r.best.c, r.best.exp);
}

/** Python's repr(x) for a float: shortest digits; exponent form when the point is at -4 or less, or past 16. */
export function pyRepr(x) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const sign = x < 0 || Object.is(x, -0) ? "-" : "";
  if (x === 0) return sign + "0.0";
  const { s, point } = shortestDigits(x);
  if (point <= -4 || point > 16) {
    const e = point - 1;
    const mant = s.length > 1 ? s[0] + "." + s.slice(1) : s;
    return sign + mant + "e" + (e < 0 ? "-" : "+") + String(Math.abs(e)).padStart(2, "0");
  }
  if (point <= 0) return sign + "0." + "0".repeat(-point) + s;
  if (point >= s.length) return sign + s + "0".repeat(point - s.length) + ".0";
  return sign + s.slice(0, point) + "." + s.slice(point);
}

/** C's printf("%.Pg", x): P significant digits, correctly rounded (ties to even on the exact value), trailing
 *  zeros removed; exponent form when the exponent is below -4 or at least P. */
export function cG(x, P = 6) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const sign = x < 0 || Object.is(x, -0) ? "-" : "";
  if (x === 0) return sign + "0";
  const v = decode(FORMATS.binary64, bitsOfNumber(Math.abs(x))).value;
  const nd = nearestDecimal(v, P);
  const full = nd.c.toString(); // exactly P digits
  const X = full.length + nd.exp - 1; // decimal exponent of the rounded value
  if (X < -4 || X >= P) {
    let mant = full[0] + (P > 1 ? "." + full.slice(1) : "");
    mant = mant.includes(".") ? mant.replace(/0+$/, "").replace(/\.$/, "") : mant;
    return sign + mant + "e" + (X < 0 ? "-" : "+") + String(Math.abs(X)).padStart(2, "0");
  }
  // fixed: P - 1 - X digits after the point
  const point = X + 1;
  let t;
  if (point <= 0) t = "0." + "0".repeat(-point) + full;
  else if (point >= full.length) t = full + "0".repeat(point - full.length);
  else t = full.slice(0, point) + "." + full.slice(point);
  if (t.includes(".")) t = t.replace(/0+$/, "").replace(/\.$/, "");
  return sign + t;
}

/** C's printf("%a", x) for zero and normal doubles (C libraries differ on subnormals, so those return null). */
export function cA(x) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  const bits = bitsOfNumber(x);
  const sign = bits >> 63n ? "-" : "";
  const E = Number((bits >> 52n) & 0x7ffn);
  const M = bits & ((1n << 52n) - 1n);
  if (E === 0 && M === 0n) return sign + "0x0p+0";
  if (E === 0) return null;
  const hex = M.toString(16).padStart(13, "0").replace(/0+$/, "");
  const e = E - 1023;
  return sign + "0x1" + (hex ? "." + hex : "") + "p" + (e < 0 ? "-" : "+") + Math.abs(e);
}
