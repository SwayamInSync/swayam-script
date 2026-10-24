// Typed input -> binary64, exactly: decimals and fractions are rounded to the nearest double by fpbig.js (never
// through JavaScript's own parser), and "a op b" applies one IEEE 754 binary64 operation, which JavaScript's
// + - * / are (correctly rounded, round to nearest even), so the result is what any conforming machine gives.

import { FORMATS, roundTo, numberOfBits } from "./fpbig.js";
import { parseNumber } from "./fpcore.js";

const SPECIAL = { inf: Infinity, "+inf": Infinity, infinity: Infinity, "+infinity": Infinity, "∞": Infinity, "+∞": Infinity,
  "-inf": -Infinity, "-infinity": -Infinity, "−inf": -Infinity, "−infinity": -Infinity, "-∞": -Infinity, "−∞": -Infinity, nan: NaN };

/** The double nearest a typed number (or ±inf / nan), or null. */
export function toDouble(text) {
  const t = text.trim().toLowerCase();
  if (Object.hasOwn(SPECIAL, t)) return SPECIAL[t]; // (not `in`: "constructor" would match)
  const x = parseNumber(t.replace(/^−/, "-"));
  if (!x) return null;
  return numberOfBits(roundTo(FORMATS.binary64, x, "RNE").bits);
}

const OPS = {
  "+": (a, b) => a + b, "-": (a, b) => a - b, "−": (a, b) => a - b,
  "*": (a, b) => a * b, "×": (a, b) => a * b, "/": (a, b) => a / b, "÷": (a, b) => a / b,
};

/** "x" or "a op b" (spaces around op): {x, a?, b?, op?}, or null. */
export function evaluate(text) {
  const m = /^\s*(\S+)\s+([+\-−*×/÷])\s+(\S+)\s*$/.exec(text);
  if (m) {
    const a = toDouble(m[1]);
    const b = toDouble(m[3]);
    if (a === null || b === null) return null;
    return { x: OPS[m[2]](a, b), a, b, op: m[2] };
  }
  const x = toDouble(text);
  return x === null ? null : { x };
}
