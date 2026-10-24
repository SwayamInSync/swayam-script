// The special-values table: every operation on ±0, ±1, ±∞ and NaN, with the IEEE 754 flag it raises.
// Mounts into <div id="fp-specials">. Hash key: so=<operation>. window.fpSpecials drives it.
// Results and flags come from specialops.js (tested against NumPy and Python).

import { VALUES, OPS } from "./specialops.js";
import { el, hashParams, writeHash, captureMode, own, scrollToWidget } from "./common.js";

const GROUPS = [
  ["arithmetic", ["add", "sub", "mul", "div"]],
  ["comparison", ["eq", "lt", "totalOrder"]],
  ["min", ["minimum", "minimumNumber", "jsMin", "pyMin"]],
  ["max", ["maximum", "maximumNumber", "jsMax", "pyMax"]],
];
const NOTES = {
  add: "∞ + (−∞) has no sensible answer, so it's NaN and raises invalid. An exact zero sum is +0, except that (−0) + (−0) is −0 (and, in round-toward-negative mode, which this table doesn't use, x − x is −0 too).",
  sub: "∞ − ∞ is invalid. x − x is +0 for every finite x, in the default rounding mode, even when x is −0.",
  mul: "The sign of a product is the XOR of the signs, even for zeros and infinities. 0 × ∞ is invalid.",
  div: "x ÷ 0 for finite, nonzero x is ±∞ and raises divideByZero. The ∞'s sign is the XOR of the two signs, so the zero's sign matters: 1 ÷ −0 is −∞ and −1 ÷ −0 is +∞. 0 ÷ 0 and ∞ ÷ ∞ are invalid instead.",
  eq: "NaN is unequal to everything, itself included; −0 == +0. Equality never raises a flag on quiet NaNs.",
  lt: "Every ordered comparison with NaN is false, and IEEE 754's <, <=, > and >= raise invalid when they meet one (Act 9). −0 < +0 is false: they're equal.",
  totalOrder: "totalOrder puts everything in one line, −NaN < −∞ < … < −0 < +0 < … < +∞ < +NaN, so it can sort anything. (The NaN here is a positive one.)",
  minimum: "IEEE 754-2019's minimum propagates NaN and treats −0 as below +0. NumPy's np.minimum gives the same answers on arm64 (Apple silicon); on x86-64 its answer for a pair of zeros can depend on the argument order.",
  minimumNumber: "IEEE 754-2019's minimumNumber prefers the number when one input is NaN, and treats −0 as below +0. NumPy's np.fmin gives the same answers on arm64 (Apple silicon); on x86-64 its answer for a pair of zeros can depend on the argument order.",
  jsMin: "JavaScript's Math.min propagates NaN and treats −0 as below +0.",
  pyMin: "Python's min keeps the first argument unless the second is strictly smaller, so NaN's position and the order of ±0 decide the answer.",
  maximum: "IEEE 754-2019's maximum propagates NaN and treats +0 as above −0. NumPy's np.maximum gives the same answers on arm64 (Apple silicon); on x86-64 its answer for a pair of zeros can depend on the argument order.",
  maximumNumber: "IEEE 754-2019's maximumNumber prefers the number when one input is NaN, and treats +0 as above −0. NumPy's np.fmax gives the same answers on arm64 (Apple silicon); on x86-64 its answer for a pair of zeros can depend on the argument order.",
  jsMax: "JavaScript's Math.max propagates NaN and treats +0 as above −0.",
  pyMax: "Python's max keeps the first argument unless the second is strictly larger.",
};
const KEYS = ["so"];
const state = { op: "div", cell: null };
const ui = {};

const show = (v) => {
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "+∞";
  if (v === -Infinity) return "−∞";
  if (v === 0) return Object.is(v, -0) ? "−0" : "+0";
  return v < 0 ? "−" + String(-v) : "+" + String(v);
};
const cls = (v) => (typeof v === "boolean" ? (v ? "fpw-yes" : "fpw-no") : Number.isNaN(v) ? "sp-nan" : !Number.isFinite(v) ? "sp-inf" : v === 0 ? "sp-zero" : "sp-num");

function render() {
  for (const [k, b] of Object.entries(ui.opBtns)) b.setAttribute("aria-pressed", k === state.op ? "true" : "false");
  const op = OPS[state.op];
  const head = el("tr", {}, el("th", { class: "sp-corner" }, op.label.split(" (")[0].replace(/\bx\b/, "x↓").replace(/\by\b/, "y→")), VALUES.map((v) => el("th", {}, show(v))));
  const rows = VALUES.map((x, i) => el("tr", {}, el("th", {}, show(x)), VALUES.map((y, j) => {
    const r = op.f(x, y);
    const flags = op.flags(x, y);
    const on = state.cell && state.cell[0] === i && state.cell[1] === j;
    const pick = () => { state.cell = [i, j]; render(); ui.grid.querySelector(`[data-fk="cell-${i}-${j}"]`)?.focus(); };
    return el("td", { class: `sp-cell ${cls(r)}${on ? " sp-on" : ""}`, tabindex: "0", role: "button", "data-fk": `cell-${i}-${j}`,
      "aria-label": `${show(x)}, ${show(y)}: ${show(r)}${flags.length ? `, raises ${flags.join(" and ")}` : ""}`, onclick: pick,
      onkeydown: (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } }, title: `${show(x)}, ${show(y)}` },
      el("span", { class: "sp-v" }, show(r)), flags.length ? el("span", { class: "sp-flag" }, flags.map((f) => (f === "invalid" ? "invalid" : "÷0")).join(" ")) : null);
  })));
  ui.grid.replaceChildren(el("table", { class: "sp-table" }, el("thead", {}, head), el("tbody", {}, rows)));
  ui.note.textContent = NOTES[state.op];
  if (state.cell) {
    const [i, j] = state.cell;
    const x = VALUES[i];
    const y = VALUES[j];
    const r = op.f(x, y);
    const flags = op.flags(x, y);
    const call = op.label.split(" (")[0].replace(/\bx\b/, show(x)).replace(/\by\b/, show(y));
    ui.detail.textContent = `${call} = ${show(r)}${flags.length ? `, and raises ${flags.join(" and ")}` : ", no flag"}.`;
  } else ui.detail.textContent = "Click any cell for its full sentence.";
}

function update(partial) {
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { so: state.op });
}

function mount(root) {
  ui.opBtns = {};
  const groups = GROUPS.map(([g, ks]) => el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small", style: "min-width:6rem" }, g + ":"),
    ks.map((k) => (ui.opBtns[k] = el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ op: k, cell: null }) }, OPS[k].label)))));
  ui.grid = el("div", { style: "overflow-x:auto" });
  ui.note = el("p", { class: "fpw-note" });
  ui.detail = el("p", { class: "fpw-num", style: "margin:0.5rem 0 0;font-size:0.9em", "aria-live": "polite" });
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" }, groups),
    el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, el("span", {}, "x down the side, y along the top"),
      el("span", { class: "fpu-small" }, "invalid / ÷0: the IEEE 754 flag the operation raises")), ui.grid, ui.detail, ui.note,
      el("p", { class: "fpw-note" }, "Values are JavaScript's (IEEE 754 binary64). Flags are IEEE 754's rules for quiet NaNs; JavaScript itself can't read them, so they're computed from the rules. For +, −, × and ÷ they match what NumPy reports from the hardware; NumPy reports no flag for comparisons, but C's <, <=, > and >= raise invalid on a NaN when compiled with FENV_ACCESS on (Act 9)."))));
}

const root = document.getElementById("fp-specials");
if (root) {
  const capturing = captureMode("specials", root);
  const h = hashParams();
  if (own(OPS, h.get("so"))) state.op = h.get("so");
  mount(root);
  render();
  if (!capturing && h.has("so")) scrollToWidget(root);
  window.fpSpecials = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
