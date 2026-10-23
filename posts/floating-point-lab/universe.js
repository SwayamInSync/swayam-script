// The 8-bit universe: every code of an 8-bit float on one screen.
// Mounts into <div id="fp-universe">. State lives in the URL hash so posts can deep-link
// to a preset, and window.fpUniverse lets the GIF recorder drive it.

import {
  makeFormat, roundRNE, parseNumber, toDecimal, shortLabel, toNumber, neighbours, sub, rabs, div, cmp,
} from "./fpcore.js";
import { captureMode, hashParams, writeHash as writeSharedHash } from "./common.js";

const PRESETS = {
  toy: { label: "Our toy 1·4·3", e: 4, specials: "ieee", subnormals: true },
  e4m3: { label: "FP8 E4M3 (OCP)", e: 4, specials: "e4m3", subnormals: true },
  e5m2: { label: "FP8 E5M2 (OCP)", e: 5, specials: "ieee", subnormals: true },
};

const SPECIALS = {
  ieee: "IEEE: top exponent = ∞ and NaN",
  e4m3: "E4M3 rule: no ∞, one NaN pattern",
  none: "None: every code is a number",
};

const CLS = {
  zero: "zero",
  sub: "subnormal",
  normal: "normal",
  inf: "infinity",
  nan: "NaN",
  flushed: "reads as 0 (no subnormals)",
};

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = (k) => String(k).replace(/[-0-9]/g, (c) => SUP[c]);
const bin = (v, w) => v.toString(2).padStart(w, "0");
const hex = (c) => "0x" + c.toString(16).toUpperCase().padStart(2, "0");
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k);
  return n;
};
const SVG = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

// ---------------------------------------------------------------------------

const state = { e: 4, specials: "ieee", subnormals: true, code: 0x38, x: "", view: "lin", zoom: "all" };
let fmt = null;
let landing = null; // result of rounding state.x
const ui = {};

const KEYS = ["preset", "e", "sp", "sub", "code", "x", "view", "zoom"];

function readHash() {
  const h = hashParams();
  if (h.has("preset") && PRESETS[h.get("preset")]) Object.assign(state, PRESETS[h.get("preset")]);
  if (h.has("e")) state.e = Math.min(6, Math.max(2, parseInt(h.get("e"), 10) || 4));
  if (h.has("sp") && SPECIALS[h.get("sp")]) state.specials = h.get("sp");
  if (h.has("sub")) state.subnormals = h.get("sub") !== "0";
  if (h.has("code")) state.code = (parseInt(h.get("code"), 16) || 0) & 0xff;
  if (h.has("x")) state.x = h.get("x");
  if (h.has("view")) state.view = h.get("view") === "log" ? "log" : "lin";
  if (h.has("zoom")) state.zoom = ["all", "one", "tiny"].includes(h.get("zoom")) ? h.get("zoom") : "all";
}

function writeHash() {
  writeSharedHash(KEYS, {
    e: state.e, sp: state.specials, sub: state.subnormals ? 1 : 0, code: state.code.toString(16),
    view: state.view, zoom: state.zoom, x: state.x,
  });
}

function rebuild() {
  fmt = makeFormat({ e: state.e, m: 7 - state.e, specials: state.specials, subnormals: state.subnormals });
  landing = null;
  const x = parseNumber(state.x);
  if (x) {
    landing = { x, ...roundRNE(fmt, x) };
    state.code = landing.code;
  }
}

function update(partial = {}) {
  Object.assign(state, partial);
  rebuild();
  render();
  writeHash();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  renderControls();
  renderStats();
  renderLine();
  renderDetail();
  renderMap();
  renderLanding();
  renderTable();
}

function renderControls() {
  for (const [k, b] of Object.entries(ui.presetBtns)) {
    const p = PRESETS[k];
    const on = p.e === state.e && p.specials === state.specials && p.subnormals === state.subnormals;
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
  ui.eRange.value = state.e;
  ui.eOut.textContent = `${state.e} exponent · ${7 - state.e} mantissa`;
  ui.spSel.value = state.specials;
  ui.subChk.checked = state.subnormals;
  for (const b of ui.viewBtns) b.setAttribute("aria-pressed", b.dataset.v === state.view ? "true" : "false");
  for (const b of ui.zoomBtns) {
    b.setAttribute("aria-pressed", b.dataset.z === state.zoom ? "true" : "false");
    b.disabled = state.view === "log";
  }
  if (ui.xIn.value !== state.x) ui.xIn.value = state.x;
}

function renderStats() {
  const below1 = fmt.positives.filter((p) => cmp(p.value, { n: 1n, d: 1n }) < 0).length;
  const items = [
    ["max finite", toDecimal(fmt.max.value).text],
    ["smallest normal", toDecimal(fmt.minNormal).text],
    ["smallest subnormal", fmt.minSub ? toDecimal(fmt.minSub).text : "none"],
    ["finite codes", String(fmt.counts.zero + fmt.counts.sub + fmt.counts.normal + fmt.counts.flushed)],
    ["infinities", String(fmt.counts.inf)],
    ["NaN codes", String(fmt.counts.nan)],
    ["below 1", `${below1} of ${fmt.positives.length} non-negative values`],
  ];
  ui.stats.replaceChildren(
    ...items.map(([k, v]) => el("div", { class: "fpu-stat" }, el("span", { class: "fpu-k" }, k), el("span", { class: "fpu-v" }, v))),
  );
}

// --- number line --------------------------------------------------------------

function lineGeometry() {
  // draw in real pixels so labels stay readable on phones
  const W = Math.max(320, Math.round(ui.line.clientWidth || 1000));
  const L = 16;
  const R = W - 16;
  const P = fmt.positives;
  if (state.view === "log") {
    const minPos = toNumber(P[1].value);
    const lo = Math.log2(minPos) - 0.6;
    const hi = Math.log2(toNumber(fmt.max.value)) + 0.6;
    return { W, L, R, x: (v) => (v <= 0 ? null : L + ((Math.log2(v) - lo) / (hi - lo)) * (R - L)), lo, hi };
  }
  const D =
    state.zoom === "tiny" ? 4 * toNumber(fmt.minNormal)
      : state.zoom === "one" ? Math.min(2, toNumber(fmt.max.value))
        : toNumber(fmt.max.value);
  return { W, L, R, D, x: (v) => (v > D * (1 + 1e-12) ? null : L + (v / D) * (R - L)) };
}

function renderLine() {
  const g = lineGeometry();
  const svg = sv("svg", { viewBox: `0 0 ${g.W} 150`, class: "fpu-svg", role: "img" });
  svg.setAttribute("aria-label", `Number line of the ${fmt.positives.length} non-negative values`);
  const base = 104;

  if (state.view === "log") {
    // one shaded band per binade [2^k, 2^(k+1)), labelled at its left edge
    let lastLabel = -Infinity;
    for (let k = Math.floor(g.lo); k <= Math.ceil(g.hi); k++) {
      const x0 = g.x(2 ** k);
      const x1 = g.x(2 ** (k + 1));
      if (x0 == null || x1 == null) continue;
      const a = Math.max(g.L, x0);
      const b = Math.min(g.R, x1);
      if (b <= a) continue;
      const isSub = k < fmt.emin;
      svg.append(sv("rect", { x: a, y: 40, width: b - a, height: base - 40, class: isSub ? "fpu-band-sub" : k % 2 ? "fpu-band-a" : "fpu-band-b" }));
      if (x0 >= g.L && x0 <= g.R) {
        svg.append(sv("line", { x1: x0, x2: x0, y1: base, y2: base + 6, class: "fpu-axis" }));
        if (x0 - lastLabel >= 38 || k === fmt.emin) {
          const t = sv("text", { x: x0, y: base + 20, class: `fpu-axis-label${k === fmt.emin ? " fpu-axis-strong" : ""}`, "text-anchor": "middle" });
          t.textContent = `2${sup(k)}`;
          svg.append(t);
          lastLabel = x0;
        }
      }
    }
    svg.append(sv("line", { x1: g.L, x2: g.R, y1: base, y2: base, class: "fpu-axis" }));
    const xs = g.x(2 ** fmt.emin);
    if (fmt.subnormals && xs != null && xs - g.L > 70) {
      const t = sv("text", { x: (g.L + xs) / 2, y: 56, class: "fpu-band-label", "text-anchor": "middle" });
      t.textContent = "subnormals";
      svg.append(t);
    }
    const note = sv("text", { x: g.L, y: 24, class: "fpu-note" });
    note.textContent = g.W < 560 ? "log₂ scale · one band per binade" : "log₂ scale: each band is one binade (zero sits at −∞, off the left edge)";
    svg.append(note);
  } else {
    svg.append(sv("line", { x1: g.L, x2: g.R, y1: base, y2: base, class: "fpu-axis" }));
    const labels = [0];
    for (let k = Math.floor(Math.log2(g.D)); labels.length < 5; k--) labels.push(2 ** k);
    for (const v of labels) {
      const x = g.x(v);
      if (x == null) continue;
      svg.append(sv("line", { x1: x, x2: x, y1: base, y2: base + 6, class: "fpu-axis" }));
      const t = sv("text", { x, y: base + 22, class: "fpu-axis-label", "text-anchor": "middle" });
      t.textContent = v === 0 ? "0" : v >= 1 ? String(v) : `2${sup(Math.round(Math.log2(v)))}`;
      svg.append(t);
    }
    const off = fmt.positives.filter((p) => g.x(toNumber(p.value)) == null).length;
    const note = sv("text", { x: g.L, y: 24, class: "fpu-note" });
    note.textContent = off
      ? g.W < 560 ? `linear · ${off} more off-screen →` : `linear scale · ${off} larger values off to the right →`
      : "linear scale · every non-negative value";
    svg.append(note);
  }

  const sel = fmt.all[state.code];
  const selPos = sel.value ? fmt.positives.find((p) => cmp(p.value, rabs(sel.value)) === 0) : null;
  for (const p of fmt.positives) {
    const x = g.x(toNumber(p.value));
    if (x == null) continue;
    const on = selPos && p.code === selPos.code;
    const tick = sv("line", { x1: x, x2: x, y1: on ? 48 : 62, y2: base, class: `fpu-tick fpu-c-${p.cls}${on ? " fpu-tick-on" : ""}` });
    tick.dataset.code = p.code;
    svg.append(tick);
  }
  if (selPos) {
    const x = g.x(toNumber(selPos.value));
    if (x != null) {
      const t = sv("text", { x: Math.min(g.R - 4, Math.max(g.L + 4, x)), y: 42, class: "fpu-sel-label", "text-anchor": x > g.R - 60 ? "end" : x < g.L + 60 ? "start" : "middle" });
      t.textContent = (sel.s ? "−" : "") + toDecimal(selPos.value, 12).text + (sel.s ? " (mirrored)" : "");
      svg.append(t);
    }
  }
  if (landing && landing.x.mag.n !== 0n) {
    const xv = g.x(toNumber(landing.x.mag));
    if (xv != null) {
      svg.append(sv("line", { x1: xv, x2: xv, y1: 54, y2: base + 2, class: "fpu-x-mark" }));
      const t = sv("text", { x: xv, y: 140, class: "fpu-x-label", "text-anchor": "middle" });
      t.textContent = `x = ${state.x.trim()}`;
      svg.append(t);
    }
  }

  // click the nearest tick
  svg.addEventListener("click", (ev) => {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    let best = null;
    for (const p of fmt.positives) {
      const x = g.x(toNumber(p.value));
      if (x == null) continue;
      const d = Math.abs(x - loc.x);
      if (!best || d < best.d) best = { d, code: p.code };
    }
    if (best && best.d < 12) update({ code: best.code | (sel.s ? fmt.signBit : 0), x: "" });
  });
  ui.line.replaceChildren(svg);
}

// --- detail panel ---------------------------------------------------------------

function renderDetail() {
  const d = fmt.all[state.code];
  const e = fmt.e;
  const m = fmt.m;
  const bits = [];
  for (let i = 7; i >= 0; i--) {
    const role = i === 7 ? "s" : i >= m ? "e" : "m";
    const on = (state.code >> i) & 1;
    bits.push(
      el("button", {
        class: `fpu-bit fpu-bit-${role}`, type: "button", "aria-pressed": on ? "true" : "false",
        "aria-label": `${role === "s" ? "sign" : role === "e" ? "exponent" : "mantissa"} bit ${i}, ${on}. Click to flip.`,
        onclick: () => update({ code: state.code ^ (1 << i), x: "" }),
      }, String(on)),
    );
    if (i === 7 || i === m) bits.push(el("span", { class: "fpu-bit-gap" }));
  }

  const sgn = d.s ? "−" : "+";
  let formula;
  if (d.cls === "normal") {
    formula = [
      el("div", {}, `(−1)${sup(d.s)} × 2^(E − bias) × (1 + M/2${sup(m)})`),
      el("div", {}, `= ${sgn}2^(${d.E} − ${fmt.bias}) × (1 + ${d.M}/${1 << m}) = ${sgn}2${sup(d.E - fmt.bias)} × ${toDecimal({ n: BigInt((1 << m) + d.M), d: BigInt(1 << m) }).text}`),
    ];
  } else if (d.cls === "sub") {
    formula = [
      el("div", {}, `(−1)${sup(d.s)} × 2^(1 − bias) × (0 + M/2${sup(m)})`),
      el("div", {}, `= ${sgn}2${sup(fmt.emin)} × ${d.M}/${1 << m}   (E = 0: no hidden 1, same spacing as the first normal binade)`),
    ];
  } else if (d.cls === "zero") {
    formula = [el("div", {}, d.s ? "E = 0 and M = 0: negative zero. Equal to +0, but 1/−0 = −∞." : "E = 0 and M = 0: zero.")];
  } else if (d.cls === "flushed") {
    formula = [el("div", {}, "E = 0 with M ≠ 0. Without subnormals this code is just another zero: a wasted pattern, and a hole between 0 and the smallest normal.")];
  } else if (d.cls === "inf") {
    formula = [el("div", {}, `All-ones exponent with M = 0: ${sgn}∞. Overflow lands here instead of wrapping around.`)];
  } else {
    formula = [el("div", {}, fmt.specials === "e4m3"
      ? "All-ones exponent and all-ones mantissa: the only NaN pattern in E4M3. Every other top-exponent code is a normal number."
      : "All-ones exponent with M ≠ 0: NaN (0/0, ∞ − ∞, √−1). The spare mantissa bits can carry a payload.")];
  }

  const rows = [
    el("div", { class: "fpu-bits" }, bits),
    el("div", { class: "fpu-legend-bits" },
      el("span", { class: "fpu-bit-s-t" }, "sign"), el("span", { class: "fpu-bit-e-t" }, `exponent (${e} bits, bias ${fmt.bias})`), el("span", { class: "fpu-bit-m-t" }, `mantissa (${m} bits)`)),
    el("div", { class: "fpu-row" },
      el("span", { class: `fpu-badge fpu-c-${d.cls}` }, CLS[d.cls]),
      el("span", { class: "fpu-mono" }, `${hex(state.code)} · ${bin(d.s, 1)} ${bin(d.E, e)} ${bin(d.M, m)} · as an unsigned integer: ${state.code}`)),
    el("div", { class: "fpu-formula fpu-mono" }, formula),
  ];
  if (d.value !== null) {
    const exact = toDecimal(d.value);
    rows.push(el("div", { class: "fpu-value" }, el("span", { class: "fpu-k" }, "exact value"), el("span", { class: "fpu-v fpu-mono" }, exact.text)));
    if (d.cls !== "zero" && d.cls !== "flushed") {
      const { down, up } = neighbours(fmt, d.code & (fmt.signBit - 1));
      const nb = [];
      if (down) nb.push(`next toward 0: ${toDecimal(down.value).text}`);
      if (up) nb.push(`next away from 0: ${toDecimal(up.value).text} (gap ${toDecimal(sub(up.value, rabs(d.value))).text})`);
      else nb.push("this is the largest finite value");
      rows.push(el("div", { class: "fpu-neigh" }, nb.join(" · ")));
    }
  }
  ui.detail.replaceChildren(...rows);
}

// --- all 256 codes ------------------------------------------------------------------

function renderMap() {
  const cells = [el("div", { class: "fpu-map-corner" })];
  for (let c = 0; c < 16; c++) cells.push(el("div", { class: "fpu-map-col" }, c.toString(16).toUpperCase()));
  for (let r = 0; r < 16; r++) {
    cells.push(el("div", { class: "fpu-map-row" }, r.toString(16).toUpperCase() + "_"));
    for (let c = 0; c < 16; c++) {
      const code = (r << 4) | c;
      const d = fmt.all[code];
      const label = d.value !== null ? toDecimal(d.value, 14).text : d.cls === "inf" ? (d.s ? "−∞" : "+∞") : "NaN";
      cells.push(el("button", {
        type: "button",
        class: `fpu-cell fpu-c-${d.cls}${code === state.code ? " fpu-cell-on" : ""}`,
        title: `${hex(code)} · ${bin(d.s, 1)} ${bin(d.E, fmt.e)} ${bin(d.M, fmt.m)} · ${label}`,
        "aria-label": `code ${hex(code)}, ${CLS[d.cls]}, ${label}`,
        onclick: () => update({ code, x: "" }),
      }));
    }
  }
  const leg = Object.keys(CLS)
    .filter((k) => fmt.counts[k])
    .map((k) => el("span", { class: "fpu-leg" }, el("i", { class: `fpu-sw fpu-c-${k}` }), `${CLS[k]} · ${fmt.counts[k]}`));
  ui.map.replaceChildren(el("div", { class: "fpu-map-grid" }, cells), el("div", { class: "fpu-legend" }, leg));
}

// --- where does x land? -------------------------------------------------------------

function renderLanding() {
  if (!state.x.trim()) {
    ui.land.replaceChildren(el("p", { class: "fpu-hint" }, "Type any real number (0.1, 1/3, 300, -2.5e-3) to see which code it rounds to."));
    return;
  }
  if (!landing) {
    ui.land.replaceChildren(el("p", { class: "fpu-hint" }, "That isn't a number this box understands. Try 0.1, -2.75, 1/3 or 1e-3."));
    return;
  }
  const x = landing.x;
  const xs = (x.neg ? "-" : "") + toDecimal(x.mag, 30).text + (toDecimal(x.mag, 30).exact ? "" : "…");
  const got = fmt.all[landing.code];
  const lines = [];
  const val = got.value !== null ? toDecimal(got.value).text : got.cls === "inf" ? (got.s ? "−∞" : "+∞") : "NaN";
  lines.push(el("p", {}, el("b", {}, "x = "), el("span", { class: "fpu-mono" }, xs), " lands on ", el("b", { class: "fpu-mono" }, val), ` (code ${hex(landing.code)}).`));

  const why = {
    exact: "It's exactly representable. No rounding happened.",
    rounded: landing.tie
      ? "x sits exactly halfway between two neighbours, so the tie goes to the one with an even mantissa (last bit 0)."
      : "Not representable, so it rounds to the nearest neighbour.",
    "underflow-zero": "x is less than half the smallest subnormal, so it rounds all the way to zero (keeping its sign).",
    flushed: "Without subnormals, anything below the smallest normal is flushed to zero.",
    "overflow-inf": "Too big: past the largest finite value by at least half a gap, so it overflows to infinity.",
    "overflow-sat": fmt.specials === "e4m3"
      ? "Too big for E4M3, which has no ∞. OCP conversions must offer SAT (clamp to ±448, shown here) and OVF (give NaN)."
      : "Too big, and this format has no ∞, so the value saturates to the largest finite number.",
  }[landing.kind];
  lines.push(el("p", {}, why));

  if (landing.lo && landing.hi && landing.kind !== "exact") {
    lines.push(el("p", { class: "fpu-mono fpu-small" }, `neighbours: ${toDecimal(landing.lo.value).text}  ≤  |x|  ≤  ${toDecimal(landing.hi.value).text}`));
  }
  if (got.value !== null && landing.kind !== "exact") {
    const xSigned = x.neg ? { n: -x.mag.n, d: x.mag.d } : x.mag;
    const err = sub(got.value, xSigned);
    const rel = div(rabs(err), x.mag);
    const errT = toDecimal(err, 30);
    lines.push(el("p", { class: "fpu-mono fpu-small" },
      `stored − x = ${err.n > 0n ? "+" : ""}${errT.text}${errT.exact ? " exactly" : "…"} · relative error ${(toNumber(rel) * 100).toPrecision(3)}%`));
  }
  ui.land.replaceChildren(...lines);
}

// --- binade table -----------------------------------------------------------------------

function renderTable() {
  const m = fmt.m;
  const rows = [];
  const head = [el("th", {}, "E field")];
  for (let M = 0; M <= fmt.mMask; M++) head.push(el("th", { class: "fpu-mono" }, bin(M, m)));
  rows.push(el("tr", {}, head));
  for (let E = 0; E <= fmt.top; E++) {
    let label;
    if (E === 0) label = fmt.subnormals ? `0 · subnormals, 2${sup(fmt.emin)} × 0.M` : "0 · zero only";
    else if (E === fmt.top && fmt.specials === "ieee") label = `${E} · ∞ / NaN`;
    else label = `${E} · 2${sup(E - fmt.bias)} × 1.M`;
    const tr = [el("th", { class: "fpu-mono" }, `${bin(E, fmt.e)} `, el("span", { class: "fpu-small" }, label))];
    for (let M = 0; M <= fmt.mMask; M++) {
      const code = (E << m) | M;
      const d = fmt.all[code];
      const text = d.value !== null ? shortLabel(d.value) : d.cls === "inf" ? "∞" : "NaN";
      tr.push(el("td", {},
        el("button", {
          type: "button",
          class: `fpu-tcell fpu-c-${d.cls}${(state.code & (fmt.signBit - 1)) === code ? " fpu-cell-on" : ""}`,
          onclick: () => update({ code: code | (state.code & fmt.signBit), x: "" }),
        }, text)));
    }
    rows.push(el("tr", {}, tr));
  }
  ui.table.replaceChildren(el("table", { class: "fpu-table" }, rows));
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

function mount(root) {
  ui.presetBtns = {};
  const presets = el("div", { class: "fpu-presets", role: "group", "aria-label": "Presets" },
    Object.entries(PRESETS).map(([k, p]) => (ui.presetBtns[k] = el("button", { type: "button", class: "fpu-btn", onclick: () => update({ ...p, x: state.x }) }, p.label))));

  ui.eRange = el("input", { type: "range", min: "2", max: "6", step: "1", "aria-label": "Exponent bits",
    oninput: (ev) => update({ e: parseInt(ev.target.value, 10) }) });
  ui.eOut = el("output", { class: "fpu-mono" });
  ui.spSel = el("select", { "aria-label": "Special values", onchange: (ev) => update({ specials: ev.target.value }) },
    Object.entries(SPECIALS).map(([k, v]) => el("option", { value: k }, v)));
  ui.subChk = el("input", { type: "checkbox", onchange: (ev) => update({ subnormals: ev.target.checked }) });

  const controls = el("div", { class: "fpu-controls" },
    presets,
    el("div", { class: "fpu-knobs" },
      el("label", { class: "fpu-knob" }, "Exponent bits ", ui.eRange, " ", ui.eOut),
      el("label", { class: "fpu-knob" }, ui.spSel),
      el("label", { class: "fpu-knob" }, ui.subChk, " subnormals (gradual underflow)")));

  ui.viewBtns = ["lin", "log"].map((v) => el("button", { type: "button", class: "fpu-btn fpu-btn-sm", "data-v": v, onclick: () => update({ view: v }) }, v === "lin" ? "linear" : "log₂"));
  ui.zoomBtns = [["all", "0 → max"], ["one", "0 → 2"], ["tiny", "near zero"]].map(([z, t]) =>
    el("button", { type: "button", class: "fpu-btn fpu-btn-sm", "data-z": z, onclick: () => update({ zoom: z }) }, t));

  ui.stats = el("div", { class: "fpu-stats" });
  ui.line = el("div", { class: "fpu-line" });
  ui.detail = el("div", { class: "fpu-detail", "aria-live": "polite" });
  ui.map = el("div", { class: "fpu-map" });
  ui.xIn = el("input", { type: "text", class: "fpu-xin", inputmode: "decimal", placeholder: "0.1", "aria-label": "A real number to round",
    oninput: (ev) => update({ x: ev.target.value }) });
  ui.land = el("div", { class: "fpu-land-out", "aria-live": "polite" });
  ui.table = el("div", { class: "fpu-table-wrap" });

  root.replaceChildren(
    el("div", { class: "fpu", tabindex: "-1" },
      controls,
      ui.stats,
      el("div", { class: "fpu-panel fpu-p-line" },
        el("div", { class: "fpu-panel-head" }, el("span", {}, "Every non-negative value"), el("span", { class: "fpu-toggles" }, ui.viewBtns, el("span", { class: "fpu-sep" }), ui.zoomBtns)),
        ui.line),
      el("div", { class: "fpu-cols" },
        el("div", { class: "fpu-panel fpu-p-detail" }, el("div", { class: "fpu-panel-head" }, "The selected code (click any bit to flip it)"), ui.detail,
          el("div", { class: "fpu-land" }, el("label", { class: "fpu-panel-head" }, "Where does x land? ", ui.xIn), ui.land)),
        el("div", { class: "fpu-panel fpu-p-map" }, el("div", { class: "fpu-panel-head" }, "All 256 codes (row = high hex digit)"), ui.map)),
      el("details", { class: "fpu-panel fpu-p-table", open: true },
        el("summary", { class: "fpu-panel-head" }, "One binade per row: the positive half, by exponent field"),
        ui.table),
    ),
  );

  root.addEventListener("keydown", (ev) => {
    if (ev.target === ui.xIn || ev.target.tagName === "SELECT" || ev.target.type === "range") return;
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
      ev.preventDefault();
      const c = Math.min(255, Math.max(0, state.code + (ev.key === "ArrowRight" ? 1 : -1)));
      update({ code: c, x: "" });
    }
  });
}

const root = document.getElementById("fp-universe");

captureMode("universe", root);

if (root) {
  readHash();
  mount(root);
  rebuild();
  render();
  let lastW = ui.line.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(ui.line.clientWidth - lastW) < 2) return;
    lastW = ui.line.clientWidth;
    renderLine();
  }).observe(ui.line);
  window.fpUniverse = {
    set: (partial) => update(partial),
    preset: (k) => update({ ...PRESETS[k] }),
    get state() {
      return { ...state };
    },
  };
}
