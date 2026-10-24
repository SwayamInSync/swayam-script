// The rounding lab: one real number, its two neighbours in a format, and where every rounding rule sends it.
// Mounts into <div id="fp-rounding">. Hash keys: rf=<format>, rx=<number>. window.fpRounding drives it for GIFs.
// All values are exact rationals (fpbig.js); JavaScript numbers are only used to place marks on the line.

import { FORMATS, bracket, roundTo, toDecimal, rat, cmp, sub, div, neg, rabs, mul, add, pow2 } from "./fpbig.js";
import { parseNumber, toNumber } from "./fpcore.js";
import { el, sv, sup, hashParams, writeHash, captureMode, own, scrollToWidget } from "./common.js";

const FMT_KEYS = ["toy", "e4m3", "e5m2", "binary16", "bfloat16", "binary32", "binary64", "binary128"];
const LABEL = {
  toy: "toy 1·4·3", e4m3: "FP8 E4M3", e5m2: "FP8 E5M2", binary16: "binary16", bfloat16: "bfloat16",
  binary32: "binary32", binary64: "binary64", binary128: "binary128",
};
const ROWS = [
  { mode: "RNE", name: "roundTiesToEven", note: "the default" },
  { mode: "RNA", name: "roundTiesToAway", note: "required for decimal formats only" },
  { mode: "RTZ", name: "roundTowardZero", note: "chop" },
  { mode: "RTP", name: "roundTowardPositive", note: "up" },
  { mode: "RTN", name: "roundTowardNegative", note: "down" },
  { mode: "RTO", name: "round to odd", note: "not an IEEE 754 mode (Act 10)" },
];
const PRESETS = [
  { label: "1/3 in binary64", f: "binary64", x: "1/3" },
  { label: "0.1 in binary64", f: "binary64", x: "0.1" },
  { label: "a tie: 1.0625 in the toy", f: "toy", x: "1.0625" },
  { label: "a tie: 1 + 2⁻⁸ in bfloat16", f: "bfloat16", x: "1.00390625" },
  { label: "464 in E4M3", f: "e4m3", x: "464" },
  { label: "overflow: 1.8e308", f: "binary64", x: "1.8e308" },
  { label: "underflow: 1e-325", f: "binary64", x: "1e-325" },
  { label: "−0.1 in the toy", f: "toy", x: "-0.1" },
];
const KEYS = ["rf", "rx"];

const state = { f: "binary64", x: "1/3" };
const ui = {};
// While the slider is held, the interval it spans stays fixed, even when x reaches one of its ends.
let anchor = null;

// --- exact numbers as text ------------------------------------------------------------------------

const ZERO = rat(0);
const isZero = (r) => r.n === 0n;

/** floor(log10 |r|) for r ≠ 0, exactly. */
function log10floor(r) {
  const a = rabs(r);
  let k = Math.floor(Math.log10(Math.abs(toNumber(a))));
  if (!Number.isFinite(k)) k = a.n.toString().length - a.d.toString().length;
  const ge = (j) => (j >= 0 ? a.n >= a.d * 10n ** BigInt(j) : a.n * 10n ** BigInt(-j) >= a.d);
  while (!ge(k)) k--;
  while (ge(k + 1)) k++;
  return k;
}

/** Exact decimal text, cut after `sig` significant digits with "…" when more digits follow. */
function exactText(r, sig = 40, plainMax = 21) {
  if (isZero(r)) return "0";
  const negative = r.n < 0n;
  const a = rabs(r);
  const k = log10floor(a);
  if (k >= -6 && k < plainMax) {
    const { text, exact } = toDecimal(a, sig);
    return (negative ? "−" : "") + text + (exact ? "" : "…");
  }
  // scientific: the first `sig` digits of a / 10^(k-sig+1), truncated
  const shift = k - sig + 1;
  const num = shift >= 0 ? a.n : a.n * 10n ** BigInt(-shift);
  const den = shift >= 0 ? a.d * 10n ** BigInt(shift) : a.d;
  const digits = (num / den).toString();
  const exact = num % den === 0n;
  const trimmed = exact ? digits.replace(/0+$/, "") || "0" : digits;
  const mant = trimmed.length > 1 ? trimmed[0] + "." + trimmed.slice(1) : trimmed;
  return (negative ? "−" : "") + mant + (exact ? "" : "…") + " × 10" + sup(k);
}

/** A short approximate label (≈, 6 significant digits) unless the value is short and exact. */
function approxText(r, digits = 6) {
  if (isZero(r)) return "0";
  const t = exactText(r, digits);
  return t.includes("…") ? "≈ " + exactText(roundSig(r, digits), digits, 9) : t;
}

/** r rounded to `digits` significant decimal digits (nearest, display only). */
function roundSig(r, digits) {
  const negative = r.n < 0n;
  const a = rabs(r);
  const k = log10floor(a);
  const shift = k - digits + 1;
  const unit = shift >= 0 ? rat(10n ** BigInt(shift)) : rat(1n, 10n ** BigInt(-shift));
  const q = div(a, unit);
  let c = q.n / q.d;
  if (2n * (q.n % q.d) >= q.d) c += 1n;
  const v = mul(rat(c), unit);
  return negative ? neg(v) : v;
}

/** Texts for two different values: in full when short, else with just enough digits to tell them apart. */
function distinctTexts(a, b, min = 6, negZero = false) {
  const full = (r) => {
    if (isZero(r)) return negZero ? "−0" : "0";
    const t = exactText(r, 20);
    return t.includes("…") ? null : t;
  };
  for (let sig = min; sig < 400; sig++) {
    const ta = full(a) ?? exactText(a, sig);
    const tb = full(b) ?? exactText(b, sig);
    if (ta !== tb) return [ta, tb];
  }
  return [exactText(a, 400), exactText(b, 400)];
}

const hexBits = (bits, width) => "0x" + BigInt(bits).toString(16).toUpperCase().padStart(Math.ceil(width / 4), "0");

// --- the computation -------------------------------------------------------------------------------

function compute() {
  const fmt = FORMATS[state.f];
  const x = parseNumber(state.x);
  if (!x) return { fmt, x: null };
  const b0 = bracket(fmt, x.mag);
  // the interval the slider spans: [lo, hi], or [x, next value up] when x is representable. Past the largest
  // finite value, lo is that value and hi is one more step of its binade, which exists only on paper.
  const over = cmp(x.mag, fmt.maxFinite) > 0;
  // a grid point past the largest finite value (2^1024 in binary64, 480 in E4M3) is not a value of the format
  const b = over && b0.exact ? { ...b0, exact: false } : b0;
  let lo = over ? fmt.maxFinite : b.lo;
  let hi = over ? add(fmt.maxFinite, pow2(fmt.emax - fmt.m)) : b.exact ? (isZero(b.lo) ? fmt.minSub : add(b.lo, b.q)) : b.hi;
  if (anchor && anchor.f === state.f && anchor.neg === x.neg) {
    lo = anchor.lo;
    hi = add(anchor.lo, anchor.gap);
  }
  const signed = (v) => (x.neg ? neg(v) : v);
  const xv = signed(x.mag);
  const gap = sub(hi, lo);
  const frac = div(sub(x.mag, lo), gap);
  const results = ROWS.map((row) => {
    const r = roundTo(fmt, x, row.mode);
    const err = r.value === null ? null : sub(r.value, xv);
    return { ...row, r, err, ulps: err === null ? null : div(err, gap) };
  });
  return { fmt, x, xv, b, lo, hi, gap, frac, signed, results, hiOverMax: cmp(hi, fmt.maxFinite) > 0, beyond: over && cmp(frac, rat(1)) >= 0 };
}

// --- rendering -------------------------------------------------------------------------------------

function renderControls() {
  for (const [k, btn] of Object.entries(ui.fmtBtns)) btn.setAttribute("aria-pressed", k === state.f ? "true" : "false");
  for (const [i, btn] of ui.presetBtns.entries()) {
    const p = PRESETS[i];
    btn.setAttribute("aria-pressed", p.f === state.f && p.x === state.x ? "true" : "false");
  }
  if (document.activeElement !== ui.xIn) ui.xIn.value = state.x;
}

function renderLine(c) {
  const W = Math.max(320, Math.round(ui.line.clientWidth || 900));
  const H = 196;
  const L = 24;
  const R = W - 24;
  const at = (f) => L + (0.12 + 0.76 * f) * (R - L);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "fpw-svg", role: "img" });
  const base = 92;
  // the left end is the lower value: lo for x ≥ 0, −hi for x < 0
  const left = c.x.neg ? c.signed(c.hi) : c.lo;
  const right = c.x.neg ? c.signed(c.lo) : c.hi;
  // x past the paper neighbour is drawn just beyond the right (or left) end
  const fx = Math.min(toNumber(c.frac), 1.1);
  const px = at(c.x.neg ? 1 - fx : fx);
  svg.setAttribute("aria-label", `x between ${distinctTexts(left, right, 6, c.x.neg).join(" and ")}`);
  svg.append(sv("rect", { x: at(0), y: base - 26, width: at(1) - at(0), height: 26, class: "fpw-band" }));
  svg.append(sv("line", { x1: L, x2: R, y1: base, y2: base, class: "fpw-line" }));
  svg.append(sv("line", { x1: at(0.5), x2: at(0.5), y1: base - 34, y2: base + 8, class: "fpw-mid" }));
  const t = (x, y, s, cls = "", anchor = "middle") => {
    const n = sv("text", { x, y, "text-anchor": anchor, class: cls });
    n.textContent = s;
    svg.append(n);
  };
  const nearMid = Math.abs(fx - 0.5) < 0.1;
  if (!nearMid) t(at(0.5), base - 40, "halfway");

  // the two neighbours, with just enough digits to tell them apart; an end past the largest finite value is dashed
  const [tl, tr] = distinctTexts(left, right, 4, c.x.neg);
  const labels = [tl, tr].map((s, i) => (((i === 1) !== c.x.neg) && c.hiOverMax ? (c.x.neg ? "past the most negative value" : "past the largest value") : s));
  // labels hang inward from their ticks; stagger them when they'd collide (≈ 8px per character)
  const stagger = (labels[0].length + labels[1].length) * 8 + 24 > at(1) - at(0) + 40;
  for (const [f, i] of [[0, 0], [1, 1]]) {
    const hollow = ((i === 1) !== c.x.neg) && c.hiOverMax;
    svg.append(sv("line", { x1: at(f), x2: at(f), y1: base - 26, y2: base + 10, class: "fpw-tick", ...(hollow ? { "stroke-dasharray": "3 3" } : {}) }));
    t(i ? at(1) + 20 : at(0) - 20, base + 28 + (stagger && i ? 20 : 0), labels[i], "fpw-strong", i ? "end" : "start");
  }
  const chipY = base + (stagger ? 70 : 50);

  // which rules land on which end
  const sideOf = (row) => {
    if (row.r.inf) return "inf";
    if (cmp(row.r.value, left) === 0) return 0;
    if (cmp(row.r.value, right) === 0) return 1;
    return null;
  };
  const groups = { 0: [], 1: [], inf: [] };
  for (const row of c.results) {
    const s = sideOf(row);
    if (s !== null) groups[s].push(row.mode);
  }
  if (!c.b.exact) {
    t(at(0) - 20, chipY, groups[0].join(" · "), "fpw-down", "start");
    t(at(1) + 20, chipY, groups[1].join(" · "), "fpw-up", "end");
    if (groups.inf.length) {
      t(c.x.neg ? at(0) - 20 : at(1) + 20, chipY + 22, `${groups.inf.join(" · ")} → ${c.x.neg ? "−∞" : "+∞"}`, "fpw-warn", c.x.neg ? "start" : "end");
    }
  }

  // x itself
  svg.append(sv("line", { x1: px, x2: px, y1: base - 50, y2: base, class: "fpw-x" }));
  t(px, base - 56, c.b.exact ? "x (exactly representable)" : c.beyond ? "x (further out)" : cmp(c.frac, rat(1, 2)) === 0 ? "x (exactly halfway)" : nearMid ? "x (near halfway)" : "x", "fpw-xt", c.beyond ? (c.x.neg ? "start" : "end") : "middle");
  ui.line.replaceChildren(svg);
}

function renderStats(c) {
  const items = [
    ["x", isZero(c.xv) && c.x.neg ? "−0" : exactText(c.xv, 30)],
    ["neighbours", distinctTexts(c.x.neg ? c.signed(c.hi) : c.lo, c.x.neg ? c.signed(c.lo) : c.hi, 6, c.x.neg).join("  and  ")],
    ["gap (1 ulp here)", exactText(c.gap, 16)],
    ["x sits", c.b.exact ? "on a value: nothing to round" : `${approxText(c.frac, 8)} of the way from the smaller-magnitude neighbour`],
  ];
  const extreme = c.x.neg ? "most negative" : "largest";
  if (c.hiOverMax) items[1] = ["neighbours", `${exactText(c.signed(c.lo), 17)} (the ${extreme} finite value) and nothing beyond`];
  if (c.beyond) {
    const n = approxText(div(sub(c.x.mag, c.lo), c.gap), 6);
    items[3][1] = `past the ${extreme} finite value by ${n} step${n === "1" ? "" : "s"} of its binade`;
  }
  if (!c.b.exact && cmp(c.frac, rat(1, 2)) === 0) items[3][1] = "exactly halfway: a tie";
  ui.stats.replaceChildren(...items.map(([k, v]) =>
    el("div", { class: "fpu-stat" }, el("span", { class: "fpu-k" }, k), el("span", { class: "fpu-v fpw-num" }, v))));
}

function renderTable(c) {
  const H = ["rule", "result", "bits", "error (result − x)", "error in ulps"];
  const head = el("tr", {}, H.map((h) => el("th", {}, h)));
  const rows = c.results.map((row) => {
    const value = row.r.inf ? (row.r.inf > 0 ? "+∞" : "−∞") : isZero(row.r.value) && c.x.neg ? "−0" : exactText(row.r.value, Math.max(24, Math.ceil(c.fmt.p * Math.log10(2)) + 3));
    let tag = "";
    if (row.r.cls === "overflow") tag = row.r.inf ? " (overflow)" : c.x.neg ? " (overflow: most negative finite)" : " (overflow: largest finite)";
    else if (row.r.cls === "underflow-zero") tag = " (underflow to zero)";
    const inf = row.r.inf > 0 ? "+∞" : "−∞";
    const dir = row.err === null ? (row.r.inf > 0 ? "fpw-up" : "fpw-down") : isZero(row.err) ? "" : row.err.n > 0n ? "fpw-up" : "fpw-down";
    return el("tr", { class: row.mode === "RTO" ? "fpw-dim" : "" },
      el("td", { "data-label": H[0] }, el("span", {}, el("strong", {}, row.mode), " ", el("span", { class: "fpu-small" }, `${row.name} · ${row.note}`))),
      el("td", { class: "fpw-num", "data-label": H[1] }, el("span", {}, value, tag ? el("span", { class: "fpw-warn" }, tag) : null)),
      el("td", { class: "fpw-num", "data-label": H[2] }, hexBits(row.r.bits, c.fmt.bits)),
      el("td", { class: `fpw-num ${dir}`, "data-label": H[3] }, row.err === null ? inf : approxText(row.err)),
      el("td", { class: `fpw-num ${dir}`, "data-label": H[4] }, row.ulps === null ? inf : approxText(row.ulps)));
  });
  ui.table.replaceChildren(el("table", { class: "fpw-table fpw-stack" }, el("thead", {}, head), el("tbody", {}, rows)));
}

function renderNotes(c) {
  const notes = [];
  if (c.b.exact) notes.push("x is already a value of this format, so every rule returns it unchanged (and no exception is raised).");
  else if (c.hiOverMax) {
    notes.push(c.x.neg
      ? `x is below the most negative finite value, ${exactText(c.signed(c.fmt.maxFinite), 20)}. The lower neighbour exists only on paper, as the next step of an unbounded exponent range.`
      : `x is above the largest finite value, ${exactText(c.fmt.maxFinite, 20)}. The upper neighbour exists only on paper, as the next step of an unbounded exponent range.`);
  }
  if (!c.b.exact && !c.hiOverMax) {
    const [tLo, tHi] = distinctTexts(c.signed(c.lo), c.signed(c.hi), 6, c.x.neg);
    notes.push(`Stochastic rounding (not an IEEE 754 mode; Act 12) would pick ${tHi} with probability ${approxText(c.frac, 8)} and ${tLo} otherwise: each neighbour with probability proportional to how close x is to it. On average, then, it returns x exactly.`);
  }
  if (c.fmt.specials === "e4m3") {
    notes.push("E4M3 has no ∞. Results that pass 448 saturate here (the OFP8 spec's saturating mode); in its non-saturating mode an overflow gives NaN instead. OFP8 requires only round-to-nearest-even for conversions (it allows other modes but doesn't define them): the other rows apply the IEEE rules to the E4M3 grid for comparison.");
  } else if (c.fmt === FORMATS.e5m2) {
    notes.push("OFP8 requires round-to-nearest-even for conversions, in a saturating and a non-saturating mode. The RNE row here is the non-saturating one, so an overflow gives ∞ (saturating mode would give ±57,344); the other rows apply the IEEE rules to the E5M2 grid for comparison.");
  }
  if (c.fmt === FORMATS.toy) notes.push("The toy uses the IEEE 754 rules on 8 bits: 4 exponent bits, 3 mantissa bits, bias 7.");
  ui.notes.replaceChildren(...notes.map((s) => el("p", { class: "fpw-note" }, s)));
}

function renderSlider(c) {
  ui.slider.disabled = !c.x;
  if (!c.x) return;
  // position of x between its neighbours, in thousandths, left to right on the line (mirrored for x < 0)
  const k = Math.min(1000, Math.round(toNumber(c.frac) * 1000));
  if (document.activeElement !== ui.slider) ui.slider.value = String(c.x.neg ? 1000 - k : k);
}

function render() {
  renderControls();
  const c = compute();
  ui.err.textContent = c.x ? "" : "Type a number like 0.1, -2.5e-3 or 1/3.";
  ui.body.hidden = !c.x;
  if (!c.x) return;
  renderStats(c);
  renderLine(c);
  renderTable(c);
  renderNotes(c);
  renderSlider(c);
}

function update(partial, fromSlider = false) {
  if (!fromSlider) anchor = null;
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { rf: state.f, rx: state.x });
}

/** Slider at k/1000 from the left end of the line: x = lo + (k/1000)(hi − lo) by magnitude, as an exact decimal. */
function onSlide(k) {
  if (!anchor) {
    const c = compute();
    if (!c.x) return;
    anchor = { f: state.f, lo: c.lo, gap: c.gap, neg: c.x.neg };
  }
  const mag = add(anchor.lo, mul(anchor.gap, rat(BigInt(anchor.neg ? 1000 - k : k), 1000n)));
  const { text } = toDecimal(mag, 100000);
  update({ x: (anchor.neg ? "-" : "") + text }, true);
}

// --- mount -------------------------------------------------------------------------------------------

function mount(root) {
  ui.fmtBtns = {};
  const fmts = el("div", { class: "fpu-presets", role: "group", "aria-label": "Format" },
    FMT_KEYS.map((k) => (ui.fmtBtns[k] = el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ f: k }) }, LABEL[k]))));
  ui.presetBtns = PRESETS.map((p) => el("button", { type: "button", class: "fpw-chip", onclick: () => update({ f: p.f, x: p.x }) }, p.label));
  ui.xIn = el("input", { type: "text", class: "fpu-xin", inputmode: "decimal", spellcheck: "false", "aria-label": "The real number to round",
    oninput: (ev) => update({ x: ev.target.value.trim() }) });
  ui.xIn.style.width = "min(28rem, 100%)";
  ui.slider = el("input", { type: "range", min: "0", max: "1000", step: "1", class: "fpw-range", "aria-label": "Move x between its two neighbours",
    oninput: (ev) => onSlide(parseInt(ev.target.value, 10)),
    onchange: () => { anchor = null; render(); } });
  ui.err = el("p", { class: "fpw-note", "aria-live": "polite" });
  ui.stats = el("div", { class: "fpu-stats" });
  ui.line = el("div", {});
  ui.table = el("div", {});
  ui.notes = el("div", {});
  ui.body = el("div", { class: "fpw", style: "margin:0" },
    ui.stats,
    el("div", { class: "fpu-panel fpw-p-line" },
      el("div", { class: "fpu-panel-head" }, el("span", {}, "x and its two neighbours"), el("span", { class: "fpu-small" }, "drag the slider to move x between them")),
      ui.line, ui.slider),
    el("div", { class: "fpu-panel fpw-p-table" },
      el("div", { class: "fpu-panel-head" }, "Where each rule sends x"), ui.table, ui.notes));
  root.replaceChildren(
    el("div", { class: "fpw" },
      el("div", { class: "fpu-controls fpw-controls" },
        fmts,
        el("label", { class: "fpu-knob" }, "x = ", ui.xIn),
        el("div", { class: "fpu-presets fpw-part-presets" }, el("span", { class: "fpu-small" }, "try:"), ui.presetBtns),
        ui.err),
      ui.body));
}

const root = document.getElementById("fp-rounding");
if (root) {
  const capturing = captureMode("rounding", root);
  const h = hashParams();
  if (own(FORMATS, h.get("rf")) && FMT_KEYS.includes(h.get("rf"))) state.f = h.get("rf");
  if (h.has("rx")) state.x = h.get("rx");
  mount(root);
  render();
  if (!capturing && (h.has("rf") || h.has("rx"))) scrollToWidget(root);
  let lastW = root.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - lastW) < 2) return;
    lastW = root.clientWidth;
    render();
  }).observe(root);
  // slide(k) moves x to k/1000 of the way along the line, as dragging would; release() ends the drag
  window.fpRounding = {
    set: (partial) => update(partial), slide: onSlide, release: () => { anchor = null; render(); },
    get state() { return { ...state }; },
  };
}
