// The block-scaling lab: quantize a vector with per-tensor FP8, MXFP8, MXFP4 and NVFP4 and see what each loses.
// Mounts into <div id="fp-blocks">. Hash keys: bp=<preset> or bv=<values>, bs=<scheme>, bx=0|1.
// The quantizers are blockscale.js (exact, tested against verify/blockscale.py); errors are summarised in doubles.

import { SCHEMES } from "./blockscale.js";
import { parseNumber, toNumber, toDecimal, sub } from "./fpcore.js";
import { floorLog2 } from "./fpbig.js";
import { el, sv, sup, hashParams, writeHash, captureMode, own, scrollToWidget } from "./common.js";

// Act 13's example: 64 draws from N(0, 1) rounded to 4 decimals (verify/act13.py, seed 2026), x[5] replaced by 40
const ACT13 = [0.8659, 0.8041, -1.9775, -0.1472, 0.5681, 40, -1.0269, -0.7563, 0.2623, -1.2323, -0.2085, -1.6969, 0.0086,
  -1.329, 0.0872, 1.3774, 1.414, 1.1809, -1.7453, 0.5713, -0.5975, -1.3849, 0.1883, 0.7375, 2.2837, -0.2898, 1.3561, -3.7565,
  -0.397, -0.034, -0.6869, 1.0764, -0.2893, 1.0674, 1.1135, -1.093, -0.1979, -0.1046, -0.2314, -0.922, -0.8907, 0.4736,
  -1.0085, -0.6026, 0.8841, 0.7939, 0.0012, -1.354, -2.2262, 0.1256, -0.0567, 1.0422, 1.0217, -0.5107, -0.2253, -0.0256,
  0.0838, 0.5189, 1.4887, 0.4722, 0.3525, -0.3037, -0.8102, -0.7833];
const NO_OUTLIER = ACT13.map((v, i) => (i === 5 ? 0.4273 : v)); // x[5]'s original draw
// the same, with the third block of 16 shrunk a millionfold: its largest value is then 3.6e-7 of the tensor's
const TINY = NO_OUTLIER.map((v, i) => (i >= 32 && i < 48 ? +(v * 1e-6).toPrecision(5) : v));
const PRESETS = {
  act13: { label: "Act 13: 64 normals, one 40", values: ACT13 },
  clean: { label: "the same, without the 40", values: NO_OUTLIER },
  tiny: { label: "one block a millionth the size", values: TINY },
};
const KEYS = ["bp", "bv", "bs", "bx"];
const state = { text: "", preset: "act13", scheme: "mx4", exclude: true };
const ui = {};

const splitVector = (text) => text.split(/[\s,;]+/).filter(Boolean);
const parseVector = (text) => {
  const parts = splitVector(text);
  if (!parts.length || parts.length > 512) return null;
  const xs = parts.map((p) => parseNumber(p));
  if (xs.some((x) => !x)) return null;
  return xs.map((x) => (x.neg ? { n: -x.mag.n, d: x.mag.d } : x.mag));
};
const textOf = (values) => values.map(String).join(", ");
const pct = (v) => (v >= 0.995 ? `${(v * 100).toFixed(0)}%` : `${(v * 100).toFixed(1)}%`);
const short = (r) => {
  if (r.n === 0n) return "0";
  const { text, exact } = toDecimal(r, 8);
  const s = exact && text.length <= 12 ? text : "≈ " + toNumber(r).toPrecision(8).replace(/\.?0+(e|$)/, "$1").replace(/e\+?/, "e");
  return s.replace(/^(≈ )?-/, "$1−");
};
const pow2Text = (r) => `2${sup(floorLog2(r))}`;

function analyse(xs, scheme) {
  const { out, scales } = SCHEMES[scheme].run(xs);
  const n = xs.length;
  let top = 0;
  for (let i = 1; i < n; i++) if (Math.abs(toNumber(xs[i])) > Math.abs(toNumber(xs[top]))) top = i;
  let se = 0;
  let sx = 0;
  let zeros = 0;
  // divide by the largest magnitude first, so squares of tiny inputs don't underflow to 0
  const sc = Math.abs(toNumber(xs[top])) || 1;
  for (let i = 0; i < n; i++) {
    if (xs[i].n !== 0n && out[i].n === 0n) zeros++;
    if (state.exclude && i === top && n > 1) continue;
    se += (toNumber(sub(xs[i], out[i])) / sc) ** 2;
    sx += (toNumber(xs[i]) / sc) ** 2;
  }
  return { out, scales, rel: sx > 0 ? Math.sqrt(se / sx) : 0, zeros, top };
}

// --- rendering ---------------------------------------------------------------------------------------------------

function renderSummary(xs) {
  const rows = Object.entries(SCHEMES).map(([k, s]) => {
    const a = analyse(xs, k);
    return el("tr", { class: k === state.scheme ? "fpw-sel" : "", style: "cursor:pointer", onclick: () => update({ scheme: k }) },
      el("td", { "data-label": "scheme" }, s.label),
      el("td", { class: "fpw-num", "data-label": "bits per value" }, s.bits),
      el("td", { class: "fpw-num", "data-label": state.exclude ? "RMS error of the others*" : "RMS error*" }, pct(a.rel)),
      el("td", { class: "fpw-num", "data-label": "inputs turned to 0" }, String(a.zeros)),
      el("td", { class: "fpw-num", "data-label": "the largest comes back as" }, `${short(xs[a.top])} → ${short(a.out[a.top])}`));
  });
  ui.summary.replaceChildren(el("table", { class: "fpw-table fpw-stack" },
    el("thead", {}, el("tr", {}, ["scheme", "bits per value", state.exclude ? "RMS error of the others*" : "RMS error*", "inputs turned to 0", "the largest comes back as"].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows)));
}

function renderChart(xs) {
  const a = analyse(xs, state.scheme);
  const s = SCHEMES[state.scheme];
  const n = xs.length;
  const W = Math.max(320, Math.round(ui.chart.clientWidth || 900));
  const H = 260;
  const L = 8;
  const R = W - 8;
  const mid = 130;
  const nums = xs.map(toNumber);
  const qs = a.out.map(toNumber);
  // a robust scale: if one value dwarfs the rest, clip it (and say so) rather than flatten everything else
  const mags = nums.map(Math.abs).sort((p, q) => q - p);
  const ymax = (mags.length > 1 && mags[0] > 3 * mags[1] ? mags[1] : mags[0]) * 1.15 || 1;
  const y = (v) => mid - (Math.max(-ymax, Math.min(ymax, v)) / ymax) * (mid - 26);
  const bw = (R - L) / n;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "fpw-svg fpw-bars", role: "img", "aria-label": `${n} values and their ${s.label} versions` });
  // block shading
  const k = s.block || n;
  for (let b = 0; b * k < n; b++) {
    if (b % 2) svg.append(sv("rect", { x: L + b * k * bw, y: 8, width: Math.min(k, n - b * k) * bw, height: H - 40, class: "blk" }));
  }
  svg.append(sv("line", { x1: L, x2: R, y1: mid, y2: mid, class: "axis" }));
  for (let i = 0; i < n; i++) {
    const x0 = L + i * bw + bw * 0.12;
    const w = Math.max(1, bw * 0.76);
    const zero = xs[i].n !== 0n && a.out[i].n === 0n;
    const q = qs[i];
    if (zero) svg.append(sv("rect", { x: x0, y: mid - 3, width: w, height: 6, class: "zero" }));
    else svg.append(sv("rect", { x: x0, y: Math.min(y(q), mid), width: w, height: Math.max(0.5, Math.abs(y(q) - mid)), class: "q" }));
    svg.append(sv("rect", { x: x0, y: Math.min(y(nums[i]), mid), width: w, height: Math.max(0.5, Math.abs(y(nums[i]) - mid)), class: "orig" }));
    if (Math.abs(nums[i]) > ymax) {
      const t = sv("text", { x: x0 + w / 2, y: nums[i] > 0 ? 20 : H - 36, "text-anchor": "middle", class: "fpw-strong" });
      t.textContent = `${short(xs[i])} → ${short(a.out[i])} ↕`;
      svg.append(t);
    }
  }
  // block scales along the bottom
  const lab = sv("text", { x: L, y: H - 8, class: "" });
  const sc0 = a.scales.length ? short(a.scales[0]) : null;
  lab.textContent = s.block ? "" : sc0 === null ? "every value is 0, so there's no scale to compute"
    : `one FP32 scale for the whole tensor, 448 / max|x| ${sc0.startsWith("≈") ? sc0 : "= " + sc0}`;
  svg.append(lab);
  if (s.block) {
    for (let b = 0; b < a.scales.length; b++) {
      const cx = L + (b * k + Math.min(k, n - b * k) / 2) * bw;
      const t = sv("text", { x: cx, y: H - 8, "text-anchor": "middle" });
      t.textContent = state.scheme === "nv4" ? `s = ${short(a.scales[b])}` : `X = ${pow2Text(a.scales[b])}`;
      svg.append(t);
    }
  }
  ui.chart.replaceChildren(svg);
  ui.chartNote.textContent = `Outlined: your values. Filled: what ${s.label} gives back. Pink: a nonzero value that came back as 0.${mags.length > 1 && mags[0] > 3 * mags[1] ? " The largest value is drawn clipped (↕) so the rest stay visible." : ""}${s.block ? ` Shading marks the blocks of ${s.block}.` : ""}`;
}

function render() {
  for (const [k, b] of Object.entries(ui.presetBtns)) b.setAttribute("aria-pressed", k === state.preset ? "true" : "false");
  for (const [k, b] of Object.entries(ui.schemeBtns)) b.setAttribute("aria-pressed", k === state.scheme ? "true" : "false");
  ui.excl.checked = state.exclude;
  const text = state.preset ? textOf(PRESETS[state.preset].values) : state.text;
  if (document.activeElement !== ui.ta) ui.ta.value = text;
  const xs = parseVector(text);
  ui.err.textContent = xs ? "" : "Type up to 512 numbers, separated by commas or spaces.";
  ui.body.hidden = !xs;
  if (!xs) return;
  try {
    renderSummary(xs);
    renderChart(xs);
  } catch (e) {
    if (!(e instanceof RangeError)) throw e; // only a range error means "doesn't fit in float32"
    ui.err.textContent = "Every value must fit in float32 (|x| below about 3.4 × 10³⁸).";
    ui.body.hidden = true;
  }
}

function update(partial) {
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { bp: state.preset || "", bv: state.preset ? "" : splitVector(state.text).join(","), bs: state.scheme, bx: state.exclude ? "" : "0" });
}

/** A fresh vector of 64 draws from N(0, 1), rounded to 4 decimals (Box–Muller, display randomness only). */
function randomVector() {
  const out = [];
  while (out.length < 64) {
    const u = 1 - Math.random();
    const v = Math.random();
    out.push(+(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)).toFixed(4));
  }
  return out;
}

function mount(root) {
  ui.presetBtns = Object.fromEntries(Object.entries(PRESETS).map(([k, p]) => [k, el("button", { type: "button", class: "fpw-chip", onclick: () => update({ preset: k }) }, p.label)]));
  const rnd = el("button", { type: "button", class: "fpw-chip", onclick: () => update({ preset: "", text: textOf(randomVector()) }) }, "64 new random normals");
  const outlier = el("button", { type: "button", class: "fpw-chip", onclick: () => {
    // works on the text as typed, so fractions like 1/3 and "−" minus signs survive
    const parts = state.preset ? PRESETS[state.preset].values.map(String) : splitVector(state.text);
    if (!parts.length || !parseVector(parts.join(","))) return;
    parts[Math.min(5, parts.length - 1)] = "40";
    update({ preset: "", text: parts.join(", ") });
  } }, "make the 6th value 40");
  ui.schemeBtns = Object.fromEntries(Object.entries(SCHEMES).map(([k, s]) => [k, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ scheme: k }) }, s.label)]));
  ui.ta = el("textarea", { class: "fpw-ta", spellcheck: "false", "aria-label": "Your vector", oninput: (e) => update({ preset: "", text: e.target.value }) });
  ui.excl = el("input", { type: "checkbox", onchange: (e) => update({ exclude: e.target.checked }) });
  ui.err = el("p", { class: "fpw-note", "aria-live": "polite" });
  ui.summary = el("div", {});
  ui.chart = el("div", {});
  ui.chartNote = el("p", { class: "fpw-note" });
  ui.body = el("div", { class: "fpw", style: "margin:0" },
    el("div", { class: "fpu-panel fpw-part-summary" }, el("div", { class: "fpu-panel-head" }, el("span", {}, "Every scheme on your vector"),
      el("label", { class: "fpu-knob fpu-small" }, ui.excl, " leave the largest |x| out of the error")), ui.summary,
      el("p", { class: "fpw-note" }, "* RMS of the errors divided by RMS of the values. Your numbers are first rounded to float32; every FP32 step of each recipe is then rounded exactly as float32 would round it. Click a row to draw it.")),
    el("div", { class: "fpu-panel fpw-part-chart" }, el("div", { class: "fpu-panel-head" }, "Element by element"), el("div", { class: "fpu-knobs" }, Object.values(ui.schemeBtns)), ui.chart, ui.chartNote));
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" },
      el("div", { class: "fpu-presets fpw-part-presets" }, el("span", { class: "fpu-small" }, "try:"), Object.values(ui.presetBtns), rnd, outlier),
      el("label", { class: "fpu-knob fpw-part-input", style: "display:block" }, el("span", { class: "fpu-small" }, "your vector (up to 512 numbers)"), ui.ta),
      ui.err),
    ui.body));
}

const root = document.getElementById("fp-blocks");
if (root) {
  const capturing = captureMode("blocks", root);
  const h = hashParams();
  if (own(PRESETS, h.get("bp"))) state.preset = h.get("bp");
  else if (h.has("bv")) {
    state.preset = "";
    state.text = h.get("bv").split(",").join(", ");
  }
  if (own(SCHEMES, h.get("bs"))) state.scheme = h.get("bs");
  if (h.has("bx")) state.exclude = h.get("bx") !== "0";
  mount(root);
  render();
  if (!capturing && KEYS.some((k) => h.has(k))) scrollToWidget(root);
  let lastW = root.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastW) < 2) return;
    lastW = root.clientWidth;
    render();
  }).observe(root);
  window.fpBlocks = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
