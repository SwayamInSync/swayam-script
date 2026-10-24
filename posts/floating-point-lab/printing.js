// The printing machine: why a double prints the digits it does. Type a number (or a + b, a - b, a * b, a / b);
// see the double you get, the interval of reals it owns, the shortest decimal inside that interval, and how
// several languages print it. Mounts into <div id="fp-printing">. Hash key: pv=<input>. window.fpPrinting drives it.

import { FORMATS, decode, bitsOfNumber, shortestInterval, toDecimal, cmp, sub, div } from "./fpbig.js";
import { toNumber } from "./fpcore.js";
import { evaluate } from "./doubles.js";
import { jsString, pyRepr, cG, cA } from "./printers.js";
import { el, sv, hashParams, writeHash, captureMode, scrollToWidget } from "./common.js";

const B64 = FORMATS.binary64;
const PRESETS = [
  ["0.1", "0.1"], ["0.1 + 0.2", "0.1 + 0.2"], ["0.3", "0.3"], ["1e23", "1e23"], ["2⁵³ + 1", "9007199254740993"],
  ["1", "1"], ["5e-324", "5e-324"], ["largest double", "1.7976931348623157e308"], ["1/3", "1/3"],
];
const KEYS = ["pv"];
const state = { v: "0.1 + 0.2" };
const ui = {};

const exactOf = (x) => decode(B64, bitsOfNumber(x)).value;
const hex64 = (x) => "0x" + bitsOfNumber(x).toString(16).toUpperCase().padStart(16, "0");

/** Exact decimal of a rational, cut after `sig` significant digits with "…" if more follow. */
function exactText(r, sig = 40) {
  const t = toDecimal(r, sig);
  return t.text + (t.exact ? "" : "…");
}
const sigDigits = (r) => {
  const t = toDecimal(r, 2000).text.replace("-", "").replace(".", "").replace(/^0+/, "");
  return t.length;
};

function compute() {
  const e = evaluate(state.v);
  if (!e) return null;
  const { x } = e;
  if (!Number.isFinite(x) || x === 0) return { ...e, special: true };
  const r = shortestInterval(Math.abs(x));
  return { ...e, r, neg: x < 0, v: exactOf(Math.abs(x)) };
}

// --- rendering -----------------------------------------------------------------------------------------------

function renderPrinters(c) {
  const x = c.x;
  const a = cA(x);
  const nanC = "nan, or -nan if the sign bit is set (glibc; Apple's prints nan)";
  const nan = Number.isNaN(x);
  const rows = [
    ["JavaScript", "String(x)", jsString(x)],
    ["Python", "repr(x)", pyRepr(x)],
    ["C", 'printf("%g", x)', nan ? nanC : cG(x, 6)],
    ["C", 'printf("%.17g", x)', nan ? nanC : cG(x, 17)],
    ["C", 'printf("%a", x)', nan ? nanC : a ?? "(subnormal: C libraries print these differently)"],
  ];
  ui.printers.replaceChildren(el("table", { class: "fpw-table fpw-stack" },
    el("thead", {}, el("tr", {}, ["language", "call", "prints"].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.map(([l, call, out]) => el("tr", {},
      el("td", { "data-label": "language" }, l),
      el("td", { class: "fpw-num", "data-label": "call" }, call),
      el("td", { class: "fpw-num", "data-label": "prints" }, el("strong", {}, out)))))));
}

function renderStats(c) {
  const items = [];
  if (c.op) items.push(["computed", `${jsString(c.a)} ${c.op} ${jsString(c.b)}, each side rounded to a double, then one correctly rounded binary64 operation`]);
  if (c.special) {
    items.push(["the double", Number.isNaN(c.x) ? "NaN" : !Number.isFinite(c.x) ? (c.x > 0 ? "+∞" : "−∞") : Object.is(c.x, -0) ? "−0" : "+0"]);
  } else {
    const n = sigDigits(c.v);
    items.push(["the double, exactly", `${c.neg ? "−" : ""}${exactText(c.v, 60)}${n > 60 ? `  (${n} significant digits in all)` : ""}`]);
    items.push(["bits", hex64(c.x)]);
  }
  ui.stats.replaceChildren(...items.map(([k, v]) =>
    el("div", { class: "fpu-stat", style: "flex-basis:100%" }, el("span", { class: "fpu-k" }, k), el("span", { class: "fpu-v fpw-num", style: "overflow-wrap:anywhere" }, v))));
}

function renderInterval(c) {
  const { r } = c;
  const sgn = c.neg ? "−" : "";
  const open = r.inclusive ? ["[", "]"] : ["(", ")"];
  // enough digits to show where the shortest candidates sit
  const sig = Math.max(20, r.best.k + 3);
  ui.interval.replaceChildren(
    el("p", { class: "fpw-num", style: "margin:0;overflow-wrap:anywhere" },
      `${open[0]}${sgn}${exactText(c.neg ? r.hi : r.lo, sig)},  ${sgn}${exactText(c.neg ? r.lo : r.hi, sig)}${open[1]}`),
    el("p", { class: "fpw-note" }, r.inclusive
      ? "Both ends belong to this double: they're ties, and its last bit is even, so ties-to-even sends them here."
      : Math.abs(c.x) === Number.MAX_VALUE
        ? "Neither end belongs to this double: they're ties, and its last bit is odd. Ties-to-even sends the lower end to the neighbour below, and the upper one overflows to ∞."
        : "Neither end belongs to this double: they're ties, and its last bit is odd, so ties-to-even sends them to the neighbours."),
  );
  renderLine(c);
}

function renderLine(c) {
  const { r } = c;
  const W = Math.max(320, Math.round(ui.line.clientWidth || 900));
  const H = 136; // room for the "chosen" label above the count line
  const L = 18;
  const R = W - 18;
  const lo = r.lo;
  const span = sub(r.hi, r.lo);
  // the interval fills the middle 76% of the width; marks are placed by their exact fraction of it
  const at = (v) => L + (0.12 + 0.76 * toNumber(div(sub(v, lo), span))) * (R - L);
  const flip = (px) => (c.neg ? L + R - px : px);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "fpw-svg", role: "img", "aria-label": "The interval this double owns, with the shortest decimals in it" });
  const base = 70;
  svg.append(sv("rect", { x: Math.min(flip(at(r.lo)), flip(at(r.hi))), y: base - 22, width: Math.abs(at(r.hi) - at(r.lo)), height: 22, class: "fpw-band" }));
  svg.append(sv("line", { x1: L, x2: R, y1: base, y2: base, class: "fpw-line" }));
  for (const v of [r.lo, r.hi]) {
    svg.append(sv("line", { x1: flip(at(v)), x2: flip(at(v)), y1: base - 26, y2: base + 6, class: "fpw-tick", ...(r.inclusive ? {} : { "stroke-dasharray": "3 3" }) }));
  }
  const t = (x, y, s, cls, anchor = "middle") => {
    const n = sv("text", { x, y, "text-anchor": anchor, class: cls });
    n.textContent = s;
    svg.append(n);
  };
  // every shortest candidate, and the chosen one
  for (const f of r.bestAll) {
    const px = flip(at(f.value));
    const chosen = cmp(f.value, r.best.value) === 0;
    svg.append(sv("line", { x1: px, x2: px, y1: base - 14, y2: base + 14, class: chosen ? "fpw-x" : "fpw-mid" }));
  }
  const px = flip(at(c.v));
  svg.append(sv("circle", { cx: px, cy: base, r: 5, class: "fpw-dot" }));
  t(px, base - 32, "the double", "fpw-strong");
  t(flip(at(r.best.value)), base + 32, `${r.bestAll.length > 1 ? "chosen: " : ""}${jsString(c.x)}`, "fpw-xt");
  t(c.neg ? R : L, H - 6, `${r.bestAll.length} decimal${r.bestAll.length > 1 ? "s" : ""} with ${r.best.k} significant digit${r.best.k > 1 ? "s" : ""} inside`, "", c.neg ? "end" : "start");
  ui.line.replaceChildren(svg);
}

function renderCandidates(c) {
  const { r } = c;
  // at a power of two the interval is lopsided, so the nearest k-digit decimal can miss it while another k-digit one is inside
  const rows = r.candidates.map((c0) => (c0.k === r.best.k && !c0.inside ? { ...r.best, inside: true } : c0)).map((cand) => el("tr", { class: cand.inside ? "" : "fpw-dim" },
    el("td", { "data-label": "digits" }, String(cand.k)),
    el("td", { class: "fpw-num", "data-label": "nearest decimal" }, (c.neg ? "−" : "") + decimal(cand)),
    el("td", { "data-label": "reads back as this double?" }, cand.inside ? el("span", { class: "fpw-yes" }, "yes") : el("span", { class: "fpw-no" }, "no"))));
  const all = r.bestAll.map((f) => (c.neg ? "−" : "") + decimal(f) + (cmp(f.value, r.best.value) === 0 ? " ← nearest the double, so it's printed" : ""));
  ui.cands.replaceChildren(
    el("table", { class: "fpw-table fpw-stack" },
      el("thead", {}, el("tr", {}, ["digits", "nearest decimal", "reads back as this double?"].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows)),
    el("p", { class: "fpw-note" }, (r.best.k === 1
      ? `One digit is enough${r.bestAll.length > 1 ? `, and ${r.bestAll.length} one-digit decimals land in the interval` : ""}`
      : `No decimal with ${r.best.k - 1} or fewer significant digits lands in the interval. With ${r.best.k}, ${r.bestAll.length === 1 ? "exactly one does" : `${r.bestAll.length} do`}`) + (r.bestAll.length > 1 ? ":" : ".")),
    ...(r.bestAll.length > 1 ? [el("ul", { class: "fpw-num", style: "margin:0.2rem 0 0;font-size:0.9em" }, all.map((s) => el("li", {}, s)))] : []),
  );
}

/** A decimal candidate {c, exp} = c × 10^exp written with all of c's digits (so "0.30" has 2 significant digits),
 *  in e-notation when the point would be far from the digits. */
function decimal(f) {
  const d = f.c.toString();
  const point = d.length + f.exp; // digits before the decimal point
  if (point > 21 || point < -5) return d[0] + (d.length > 1 ? "." + d.slice(1) : "") + "e" + (point - 1 < 0 ? "−" : "+") + Math.abs(point - 1);
  if (point <= 0) return "0." + "0".repeat(-point) + d;
  if (point >= d.length) return d + "0".repeat(point - d.length);
  return d.slice(0, point) + "." + d.slice(point);
}

function render() {
  if (document.activeElement !== ui.xIn) ui.xIn.value = state.v;
  for (const [i, b] of ui.presetBtns.entries()) b.setAttribute("aria-pressed", PRESETS[i][1] === state.v ? "true" : "false");
  const c = compute();
  ui.err.textContent = c ? "" : "Type a number like 0.1, 1e23 or 1/3, or a sum like 0.1 + 0.2 (spaces around the operator).";
  ui.body.hidden = !c;
  if (!c) return;
  renderStats(c);
  renderPrinters(c);
  ui.rest.hidden = !!c.special;
  if (c.special) return;
  renderInterval(c);
  renderCandidates(c);
}

function update(partial) {
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { pv: state.v });
}

function mount(root) {
  ui.xIn = el("input", { type: "text", class: "fpu-xin", spellcheck: "false", "aria-label": "A number or a + b", oninput: (ev) => update({ v: ev.target.value }) });
  ui.xIn.style.width = "min(24rem, 100%)";
  ui.presetBtns = PRESETS.map(([label, v]) => el("button", { type: "button", class: "fpw-chip", onclick: () => update({ v }) }, label));
  ui.err = el("p", { class: "fpw-note", "aria-live": "polite" });
  ui.stats = el("div", { class: "fpu-stats" });
  ui.printers = el("div", {});
  ui.interval = el("div", {});
  ui.line = el("div", {});
  ui.cands = el("div", {});
  ui.rest = el("div", { class: "fpw", style: "margin:0" },
    el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "The reals this double owns (everything that rounds to it)"), ui.interval, ui.line),
    el("div", { class: "fpu-panel fpw-part-search" }, el("div", { class: "fpu-panel-head" }, "The search: the shortest decimal inside"), ui.cands));
  ui.body = el("div", { class: "fpw", style: "margin:0" },
    ui.stats,
    el("div", { class: "fpu-panel fpw-part-printers" }, el("div", { class: "fpu-panel-head" }, "How it prints"), ui.printers,
      el("p", { class: "fpw-note" }, "JavaScript and Python print the shortest decimal that reads back as the same double (except that JavaScript prints −0 as 0). C's %g rounds to 6 significant digits and %.17g to 17 (both dropping trailing zeros), and %a prints the exact value in hex. Every string here is computed exactly from the bits.")),
    ui.rest);
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" },
      el("label", { class: "fpu-knob" }, "x = ", ui.xIn),
      el("div", { class: "fpu-presets fpw-part-presets" }, el("span", { class: "fpu-small" }, "try:"), ui.presetBtns),
      ui.err),
    ui.body));
}

const root = document.getElementById("fp-printing");
if (root) {
  const capturing = captureMode("printing", root);
  const h = hashParams();
  if (h.has("pv")) state.v = h.get("pv");
  mount(root);
  render();
  if (!capturing && h.has("pv")) scrollToWidget(root);
  let lastW = root.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastW) < 2) return;
    lastW = root.clientWidth;
    render();
  }).observe(root);
  window.fpPrinting = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
