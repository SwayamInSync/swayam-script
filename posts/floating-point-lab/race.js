// The summation race: the same terms summed five ways, measured against the exact sum as the count grows.
// Mounts into <div id="fp-race">. Hash keys: rs=<terms>, rn=<log10 n>, rseed=<seed>. window.fpRace drives it.
// Sums run in binary64 (summation.js, tested against NumPy and Python); errors are computed exactly.

import { naive, numpySum, kahan, neumaier, roundedExact, ExactAccumulator, seeded } from "./summation.js";
import { exactOfNumber } from "./fpbig.js";
import { toDecimal, toNumber, sub, div, rabs } from "./fpcore.js";
import { jsString } from "./printers.js";
import { el, sv, sup, hashParams, writeHash, captureMode, own, scrollToWidget } from "./common.js";

const U = 2 ** -53;
const TERMS = {
  random: { label: "random numbers in [0, 1)" },
  tenth: { label: "0.1, over and over" },
  tricky: { label: "1, 10¹⁰⁰, 1, −10¹⁰⁰" },
};
const METHODS = [
  { key: "naive", label: "plain loop", fn: naive, cls: "m-naive" },
  { key: "numpy", label: "np.sum (pairwise)", fn: numpySum, cls: "m-numpy" },
  { key: "kahan", label: "Kahan", fn: kahan, cls: "m-kahan" },
  { key: "neumaier", label: "Neumaier (Python 3.12+ sum of floats)", fn: neumaier, cls: "m-neumaier" },
  { key: "fsum", label: "math.fsum (exact, rounded once)", fn: null, cls: "m-fsum" },
];
const KEYS = ["rs", "rn", "rseed"];
const state = { terms: "tenth", logn: 6, seed: 754 };
const ui = {};
let last = null;

function makeTerms(n) {
  if (state.terms === "tricky") return Float64Array.from([1, 1e100, 1, -1e100]);
  const xs = new Float64Array(n);
  if (state.terms === "tenth") xs.fill(0.1);
  else {
    const g = seeded(state.seed);
    for (let i = 0; i < n; i++) xs[i] = g.next53() * U;
  }
  return xs;
}

const relErr = (v, exact) => {
  if (exact.n === 0n) return v === 0 ? 0 : Infinity;
  if (!Number.isFinite(v)) return Infinity;
  return toNumber(rabs(div(sub(exactOfNumber(v), exact), exact)));
};

/** Sum the first n terms every way, at log-spaced checkpoints (10^(j/8)) up to n. */
function race() {
  const n = state.terms === "tricky" ? 4 : 10 ** state.logn;
  const xs = makeTerms(n);
  const checkpoints = state.terms === "tricky" ? [4] : [...new Set(Array.from({ length: 8 * state.logn - 7 }, (_, j) => Math.round(10 ** ((j + 8) / 8))))];
  const acc = new ExactAccumulator();
  const rows = [];
  let i = 0;
  for (const c of checkpoints) {
    for (; i < c; i++) acc.add(xs[i]);
    const exact = acc.value;
    const row = { n: c, exact };
    for (const m of METHODS) {
      const v = m.fn ? m.fn(xs, c) : roundedExact(exact);
      row[m.key] = { v, rel: relErr(v, exact) };
    }
    rows.push(row);
  }
  return { n, rows };
}

// --- rendering -------------------------------------------------------------------------------------------------

function renderTable(res) {
  const final = res.rows[res.rows.length - 1];
  const ex = final.exact;
  const t = toDecimal(ex, 30);
  ui.exact.textContent = `exact sum of all ${res.n.toLocaleString("en-US")} terms: ${t.text}${t.exact ? "" : "…"}`;
  ui.table.replaceChildren(el("table", { class: "fpw-table fpw-stack" },
    el("thead", {}, el("tr", {}, ["method", "result", "relative error", "in units of u = 2⁻⁵³"].map((h) => el("th", {}, h)))),
    el("tbody", {}, METHODS.map((m) => {
      const r = final[m.key];
      const units = r.rel / U;
      return el("tr", {},
        el("td", { "data-label": "method" }, el("i", { class: `fpw-sw ${m.cls}` }), " ", m.label),
        el("td", { class: "fpw-num", "data-label": "result" }, jsString(r.v)),
        el("td", { class: "fpw-num", "data-label": "relative error" }, r.rel === 0 ? "0 (exact)" : r.rel === Infinity ? "∞" : r.rel.toPrecision(3)),
        el("td", { class: "fpw-num", "data-label": "in units of u" }, r.rel === 0 ? "0" : r.rel === Infinity ? "∞" : units >= 100 ? Math.round(units).toLocaleString("en-US") : units.toPrecision(3)));
    }))));
}

function renderChart(res) {
  if (res.rows.length < 2) {
    ui.chartWrap.hidden = true;
    return;
  }
  ui.chartWrap.hidden = false;
  const W = Math.max(320, Math.round(ui.chart.clientWidth || 900));
  const H = 300;
  const m = { l: 58, r: 14, t: 14, b: 40 };
  const all = res.rows.flatMap((r) => METHODS.map((mm) => r[mm.key].rel)).filter((v) => v > 0 && Number.isFinite(v));
  const lo = -18;
  const hi = Math.max(-14, Math.ceil(Math.log10(Math.max(...all, 1e-15)) + 0.3));
  const x = (n) => m.l + (Math.log10(n) - 1) / (state.logn - 1) * (W - m.l - m.r);
  const y = (v) => {
    const lv = v <= 0 ? lo : Math.max(lo, Math.log10(v));
    return m.t + (hi - lv) / (hi - lo) * (H - m.t - m.b);
  };
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "fpw-svg", role: "img", "aria-label": "Relative error against the exact sum as the number of terms grows" });
  for (let k = lo; k <= hi; k++) {
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: y(10 ** k), y2: y(10 ** k), class: "fpa-grid" }));
    const t = sv("text", { x: m.l - 6, y: y(10 ** k) + 4, "text-anchor": "end" });
    t.textContent = k === lo ? `≤ 10${sup(k)}` : `10${sup(k)}`; // exact sums (error 0) are drawn on this line too
    svg.append(t);
  }
  for (let k = 1; k <= state.logn; k++) {
    const t = sv("text", { x: x(10 ** k), y: H - m.b + 18, "text-anchor": "middle" });
    t.textContent = `10${sup(k)}`;
    svg.append(t);
  }
  const ax = sv("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" });
  ax.textContent = "number of terms";
  svg.append(ax);
  // u, for scale
  svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: y(U), y2: y(U), class: "fpw-mid" }));
  const ut = sv("text", { x: W - m.r - 4, y: y(U) - 5, "text-anchor": "end", class: "fpw-halo" });
  ut.textContent = "u = 2⁻⁵³";
  svg.append(ut);
  // draw the exact-sum line first, so the methods that land on it stay visible on top
  for (const mm of [...METHODS].sort((p, q) => (p.key === "fsum" ? -1 : q.key === "fsum" ? 1 : 0))) {
    const pts = res.rows.map((r) => `${x(r.n).toFixed(1)},${y(r[mm.key].rel).toFixed(1)}`).join(" ");
    svg.append(sv("polyline", { points: pts, class: `fpw-run ${mm.cls}` }));
  }
  ui.chart.replaceChildren(svg);
}

function render(rerun = true) {
  for (const [k, b] of Object.entries(ui.termBtns)) b.setAttribute("aria-pressed", k === state.terms ? "true" : "false");
  for (const [k, b] of Object.entries(ui.nBtns)) {
    b.setAttribute("aria-pressed", Number(k) === state.logn ? "true" : "false");
    b.disabled = state.terms === "tricky";
  }
  ui.seedIn.value = String(state.seed);
  ui.seedWrap.hidden = state.terms !== "random";
  if (rerun || !last) last = race();
  renderTable(last);
  renderChart(last);
  ui.note.textContent = state.terms === "tricky"
    ? "Kahan's correction is lost when a term dwarfs the running sum; Neumaier's variant keeps it. (In Python, write the 1s as 1.0: on 3.12 and 3.13, sum() doesn't compensate int terms, and sum([1, 1e100, 1, -1e100]) is 0.0.)"
    : state.terms === "random"
      ? `Terms are k / 2⁵³ for random integers k (mulberry32, seed ${state.seed}), so every term is an exact double, and the exact sum is an integer over 2⁵³. They aren't the post's NumPy draws, so the exact numbers differ from the post's chart.`
      : "0.1 is the double nearest one tenth, so the exact sum is n times that double, not n/10.";
}

function update(partial) {
  // the 10⁷ race takes a couple of seconds: don't run it again for a click that changes nothing
  const changed = Object.entries(partial).some(([k, v]) => state[k] !== v);
  Object.assign(state, partial);
  render(changed);
  writeHash(KEYS, { rs: state.terms, rn: state.logn, rseed: state.terms === "random" ? state.seed : "" });
}

function mount(root) {
  ui.termBtns = Object.fromEntries(Object.entries(TERMS).map(([k, t]) => [k, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ terms: k }) }, t.label)]));
  ui.nBtns = Object.fromEntries([3, 4, 5, 6, 7].map((k) => [k, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ logn: k }) }, `10${sup(k)}${k === 7 ? " (slow)" : ""}`)]));
  // seeds are 32-bit unsigned (the generator reduces them mod 2³² anyway), so the box shows the seed that ran
  ui.seedIn = el("input", { type: "number", class: "fpu-xin", style: "width:7rem", min: "0", max: "4294967295", step: "1", "aria-label": "Seed",
    onchange: (e) => update({ seed: (Math.trunc(Number(e.target.value)) || 0) >>> 0 }) });
  ui.seedWrap = el("label", { class: "fpu-knob" }, "seed ", ui.seedIn);
  ui.exact = el("p", { class: "fpw-num", style: "margin:0 0 0.4rem;overflow-wrap:anywhere" });
  ui.table = el("div", {});
  ui.note = el("p", { class: "fpw-note" });
  ui.chart = el("div", {});
  ui.chartWrap = el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "Relative error as the terms pile up"), ui.chart,
    el("div", { class: "fpu-legend" }, METHODS.map((mm) => el("span", { class: "fpu-leg" }, el("i", { class: `fpw-sw ${mm.cls}` }), mm.label))));
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" },
      el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small" }, "terms:"), Object.values(ui.termBtns), ui.seedWrap),
      el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small" }, "how many:"), Object.values(ui.nBtns))),
    el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "The finish line"), ui.exact, ui.table, ui.note),
    ui.chartWrap));
}

const root = document.getElementById("fp-race");
if (root) {
  const capturing = captureMode("race", root);
  const h = hashParams();
  if (own(TERMS, h.get("rs"))) state.terms = h.get("rs");
  if (h.has("rn")) state.logn = Math.min(7, Math.max(3, parseInt(h.get("rn"), 10) || 6));
  if (h.has("rseed")) state.seed = (Math.trunc(Number(h.get("rseed"))) || 0) >>> 0;
  mount(root);
  render(true);
  if (!capturing && KEYS.some((k) => h.has(k))) scrollToWidget(root);
  let lastW = root.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastW) < 2) return;
    lastW = root.clientWidth;
    render(false);
  }).observe(root);
  window.fpRace = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
