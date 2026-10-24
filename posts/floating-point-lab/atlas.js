// The format atlas: every scalar float format from FP4 to binary256 on one range-vs-precision chart.
// Mounts into <div id="fp-atlas">. Data comes from formats.json, which verify/formats.py generates
// from the same checked table as the post. Hash keys: atlas=<format id>, adec=1 (show decimal formats).

import { el, sv, sup, hashParams, writeHash, captureMode, scrollToWidget } from "./common.js";

const GROUPS = {
  ieee: "IEEE 754 binary",
  ocp: "OCP FP8 / FP6 / FP4",
  ml: "bfloat16 & TF32",
  x87: "x87 extended",
  decimal: "IEEE 754 decimal",
};

// label placement per format so names don't collide: [dx, dy, anchor]
const LABEL = {
  "fp4-e2m1": [10, 4, "start"], "fp6-e2m3": [-10, -8, "end"], "fp6-e3m2": [0, 20, "middle"],
  "fp8-e4m3": [10, -8, "start"], "fp8-e5m2": [10, 16, "start"], binary16: [10, 4, "start"],
  bfloat16: [10, 14, "start"], tf32: [10, -2, "start"], binary32: [-10, -8, "end"],
  binary64: [-10, 4, "end"], "x87-extended": [10, 4, "start"], binary128: [-10, 4, "end"],
  binary256: [-10, 4, "end"], decimal32: [10, 4, "start"], decimal64: [10, 4, "start"],
  decimal128: [10, 4, "start"],
};
const NARROW_LABELS = new Set(["binary32", "binary64", "binary128", "binary256"]);
const SHORT = { "x87-extended": "x87 80-bit", "fp4-e2m1": "FP4 E2M1", "fp6-e2m3": "FP6 E2M3", "fp6-e3m2": "FP6 E3M2",
  "fp8-e4m3": "FP8 E4M3", "fp8-e5m2": "FP8 E5M2", tf32: "TF32" };

const KEYS = ["atlas", "adec"];
const state = { sel: "binary64", dec: false };
let data = null;
const ui = {};

const label = (f) => SHORT[f.id] || f.name;
const precisionBits = (f) => (f.group === "decimal" ? f.p_bits_equiv : f.p);
const visible = () => [...data.binary, ...(state.dec ? data.decimal : [])];
const byId = (id) => [...data.binary, ...data.decimal].find((f) => f.id === id);
const fmtBinades = (b) => (b >= 100 ? Math.round(b).toLocaleString("en-US") : b.toFixed(b >= 10 ? 1 : 2));

function readState() {
  const h = hashParams();
  if (h.has("atlas") && byId(h.get("atlas"))) state.sel = h.get("atlas");
  if (h.has("adec")) state.dec = h.get("adec") === "1";
  if (byId(state.sel).group === "decimal") state.dec = true;
}

function update(partial) {
  // re-rendering replaces the chart, so put keyboard focus back on the point that had it
  const fk = document.activeElement?.dataset?.fk;
  Object.assign(state, partial);
  if (!state.dec && byId(state.sel).group === "decimal") state.sel = "binary64";
  render();
  if (fk) root.querySelector(`[data-fk="${fk}"]`)?.focus();
  writeHash(KEYS, { atlas: state.sel, adec: state.dec ? "1" : "" });
}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

function renderChart() {
  const W = Math.max(320, Math.round(ui.chart.clientWidth || 900));
  const H = Math.round(Math.min(460, Math.max(300, W * 0.5)));
  const m = { l: W < 560 ? 44 : 58, r: 18, t: 18, b: W < 560 ? 54 : 50 };
  const X0 = Math.log10(2);
  const X1 = 6.1;
  const Y0 = Math.log2(1.7);
  const Y1 = Math.log2(300);
  const x = (binades) => m.l + ((Math.log10(binades) - X0) / (X1 - X0)) * (W - m.l - m.r);
  const y = (p) => H - m.b - ((Math.log2(p) - Y0) / (Y1 - Y0)) * (H - m.t - m.b);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "fpa-svg", role: "img" });
  svg.setAttribute("aria-label", "Chart of floating-point formats: range in binades against precision in bits, both on log scales");

  // grid + axes
  for (const [v, t] of [[2, "2"], [10, "10"], [100, "100"], [1000, "1,000"], [1e4, "10⁴"], [1e5, "10⁵"], [1e6, "10⁶"]]) {
    const gx = x(v);
    svg.append(sv("line", { x1: gx, x2: gx, y1: m.t, y2: H - m.b, class: "fpa-grid" }));
    const tx = sv("text", { x: gx, y: H - m.b + 16, class: "fpa-tick", "text-anchor": "middle" });
    tx.textContent = t;
    svg.append(tx);
  }
  for (const p of [2, 4, 8, 16, 32, 64, 128, 256]) {
    const gy = y(p);
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: gy, y2: gy, class: "fpa-grid" }));
    const ty = sv("text", { x: m.l - 8, y: gy + 4, class: "fpa-tick", "text-anchor": "end" });
    ty.textContent = String(p);
    svg.append(ty);
  }
  const xt = sv("text", { x: m.l + (W - m.l - m.r) / 2, y: H - 8, class: "fpa-axis-title", "text-anchor": "middle" });
  xt.textContent = W < 560 ? "range in binades (log) →" : "range: binades from the smallest positive value to the largest (log scale) →";
  svg.append(xt);
  const yt = sv("text", { x: 14, y: m.t + (H - m.t - m.b) / 2, class: "fpa-axis-title", "text-anchor": "middle",
    transform: `rotate(-90 14 ${m.t + (H - m.t - m.b) / 2})` });
  yt.textContent = W < 560 ? "precision (bits) →" : "precision p in bits (log scale) →";
  svg.append(yt);

  // the IEEE binary family as a dashed path
  const fam = data.binary.filter((f) => f.group === "ieee");
  svg.append(sv("polyline", { points: fam.map((f) => `${x(f.range_binades)},${y(f.p)}`).join(" "), class: "fpa-family" }));

  const sel = byId(state.sel);
  const pts = visible().slice().sort((a, b) => (a.id === state.sel) - (b.id === state.sel));
  for (const f of pts) {
    const cx = x(f.range_binades);
    const cy = y(precisionBits(f));
    const on = f.id === sel.id;
    const g = sv("g", { class: `fpa-pt fpa-g-${f.group}${on ? " fpa-pt-on" : ""}`, tabindex: "0", role: "button", "data-fk": `pt-${f.id}` });
    g.setAttribute("aria-label", f.group === "decimal"
      ? `${f.name}: ${f.p_digits} decimal digits (about ${precisionBits(f).toFixed(0)} bits) of precision, ${fmtBinades(f.range_binades)} binades of range`
      : `${f.name}: ${precisionBits(f).toFixed(0)} bits of precision, ${fmtBinades(f.range_binades)} binades of range`);
    if (f.group === "decimal") {
      const r = on ? 8 : 6;
      g.append(sv("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, class: "fpa-mark fpa-mark-hollow" }));
    } else {
      g.append(sv("circle", { cx, cy, r: on ? 9 : 6.5, class: "fpa-mark" }));
    }
    // narrow screens: label only the selection and a few anchors, so names don't pile up
    if (W >= 560 || on || NARROW_LABELS.has(f.id)) {
      const [dx, dy, anchor] = LABEL[f.id] || [10, 4, "start"];
      const t = sv("text", { x: cx + dx, y: cy + dy, class: "fpa-label", "text-anchor": anchor });
      t.textContent = label(f);
      g.append(t);
    }
    g.addEventListener("click", () => update({ sel: f.id }));
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        update({ sel: f.id });
      }
    });
    svg.append(g);
  }
  ui.chart.replaceChildren(svg);
}

// ---------------------------------------------------------------------------
// Bit layout bar
// ---------------------------------------------------------------------------

function segments(f) {
  if (f.group === "decimal") {
    return [["s", 1, "sign"], ["c", f.combination_bits, "combination"], ["t", f.trailing_bits, "trailing significand"]];
  }
  const seg = [];
  if (f.sign) seg.push(["s", 1, "sign"]);
  seg.push(["e", f.exp_bits, "exponent"]);
  if (f.explicit_int_bit) seg.push(["i", 1, "explicit integer bit"]);
  seg.push(["m", f.man_bits, "mantissa"]);
  return seg;
}

/** One format's bit layout as an SVG bar; `scaleBits` fixes the width of a bit across rows. */
function layoutBar(f, width, scaleBits, height = 22) {
  const bitW = width / scaleBits;
  const svg = sv("svg", { viewBox: `0 0 ${width} ${height}`, class: "fpa-bar", preserveAspectRatio: "xMinYMid meet" });
  let at = 0;
  for (const [role, n] of segments(f)) {
    const w = n * bitW;
    svg.append(sv("rect", { x: at, y: 0, width: Math.max(w - 0.5, 0.5), height, class: `fpa-seg fpa-seg-${role}` }));
    if (bitW >= 6) {
      for (let i = 1; i < n; i++) svg.append(sv("line", { x1: at + i * bitW, x2: at + i * bitW, y1: 3, y2: height - 3, class: "fpa-bitline" }));
    }
    at += w;
  }
  return svg;
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

function renderDetail() {
  const f = byId(state.sel);
  const W = Math.max(280, Math.round(ui.detail.clientWidth || 420) - 4);
  const segs = segments(f);
  const segText = segs.map(([, n, name]) => `${n} ${name}`).join(" · ");
  const hidden = f.group !== "decimal" && !f.explicit_int_bit ? " (+1 hidden bit)" : "";
  const rows = [];
  const row = (k, v) => rows.push(el("dt", {}, k), el("dd", {}, v));

  if (f.group === "decimal") {
    row("precision", `${f.p_digits} decimal digits (≈ ${f.p_bits_equiv.toFixed(1)} bits)`);
    row("exponent range", `emin = ${String(f.emin).replace("-", "−")}, emax = ${f.emax} for d.ddd… × 10ᵉ; the stored exponent is q + ${f.bias}, where q = e − ${f.p_digits - 1} is the exponent of the integer significand`);
  } else {
    row("precision", `p = ${f.p} bits`);
    row("exponent range", `emin = ${String(f.emin).replace("-", "−")}, emax = ${f.emax} (bias ${f.bias})`);
  }
  row("largest", f.max);
  row("smallest normal", f.min_normal + (Number.isInteger(f.log2_min_normal) && f.group !== "decimal" ? `  (2${sup(f.log2_min_normal)})` : ""));
  row("smallest subnormal", f.min_subnormal + (f.group !== "decimal" && Number.isInteger(f.log2_min_pos) ? `  (2${sup(f.log2_min_pos)})` : ""));
  if (f.group !== "decimal") row("machine epsilon", `2${sup(f.eps_exp)} ${f.eps.startsWith("≈") ? "" : "= "}${f.eps}`);
  // the digit formulas assume a wide exponent range; for the tiny formats they'd mislead, so skip them
  if (f.group !== "decimal" && f.p >= 8) row("decimal digits", `${f.digits[0]} always survive · ${f.digits[1]} to round-trip`);
  row("range", `${fmtBinades(f.range_binades)} binades`);
  if (f.specials) row("special values", f.specials);
  if (f.container_bits) row("storage", `${f.bits} bits of format, stored in a ${f.container_bits}-bit container`);
  if (f.notes) row("note", f.notes);

  ui.detail.replaceChildren(
    el("div", { class: "fpa-title" }, el("span", { class: "fpa-name" }, f.name), el("span", { class: `fpa-badge fpa-g-${f.group}` }, GROUPS[f.group])),
    el("div", { class: "fpa-detail-bar" }, layoutBar(f, W, Math.max(f.bits, 16), 26)),
    el("div", { class: "fpa-segtext" }, `${f.bits} bits: ${segText}${hidden}`),
    el("dl", { class: "fpa-facts" }, rows),
  );
}

// ---------------------------------------------------------------------------
// Every layout, to scale
// ---------------------------------------------------------------------------

function renderLayouts() {
  const W = Math.max(260, Math.round(ui.layouts.clientWidth || 420) - 120);
  const list = visible().slice().sort((a, b) => a.bits - b.bits || a.p - b.p);
  ui.layouts.replaceChildren(
    ...list.map((f) => el("button", {
      type: "button", class: `fpa-lrow${f.id === state.sel ? " fpa-lrow-on" : ""}`, onclick: () => update({ sel: f.id }),
      "aria-label": `${f.name}, ${f.bits} bits`,
    }, el("span", { class: "fpa-lname" }, label(f)), el("span", { class: "fpa-lbar" }, layoutBar(f, W, 256, 14)),
    el("span", { class: "fpa-lbits" }, String(f.bits)))),
  );
}

function renderControls() {
  ui.decBtn.setAttribute("aria-pressed", state.dec ? "true" : "false");
  ui.decBtn.textContent = state.dec ? "hide decimal formats" : "show decimal formats";
}

function render() {
  renderControls();
  renderChart();
  renderDetail();
  renderLayouts();
}

// ---------------------------------------------------------------------------

function mount(root) {
  ui.decBtn = el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ dec: !state.dec }) }, "show decimal formats");
  const legend = el("div", { class: "fpa-legend" },
    Object.entries(GROUPS).filter(([k]) => k !== "decimal").map(([k, v]) => el("span", { class: "fpa-leg" }, el("i", { class: `fpa-sw fpa-g-${k}` }), v)),
    el("span", { class: "fpa-leg" }, el("i", { class: "fpa-sw fpa-sw-hollow fpa-g-decimal" }), GROUPS.decimal),
    ui.decBtn);
  ui.chart = el("div", { class: "fpa-chart" });
  ui.detail = el("div", { class: "fpa-detail", "aria-live": "polite" });
  ui.layouts = el("div", { class: "fpa-layouts" });
  root.replaceChildren(
    el("div", { class: "fpa" },
      el("div", { class: "fpu-panel fpa-p-chart" },
        el("div", { class: "fpu-panel-head" }, el("span", {}, "Range vs precision, every format"), legend),
        ui.chart,
        el("div", { class: "fpa-hint" }, "Right = more range, up = more precision. Click (or tap) any format.")),
      el("div", { class: "fpa-cols" },
        el("div", { class: "fpu-panel fpa-p-detail" }, ui.detail),
        el("div", { class: "fpu-panel fpa-p-layouts" },
          el("div", { class: "fpu-panel-head" }, "Every layout, to scale (one bit = one sliver)"), ui.layouts,
          el("div", { class: "fpa-keys" },
            el("span", {}, el("i", { class: "fpa-sw fpa-seg-s" }), "sign"), el("span", {}, el("i", { class: "fpa-sw fpa-seg-e" }), "exponent"),
            el("span", {}, el("i", { class: "fpa-sw fpa-seg-i" }), "explicit integer bit"), el("span", {}, el("i", { class: "fpa-sw fpa-seg-m" }), "mantissa"),
            el("span", {}, el("i", { class: "fpa-sw fpa-seg-c" }), "decimal combination field"), el("span", {}, el("i", { class: "fpa-sw fpa-seg-t" }), "decimal trailing significand"))))));
}

const root = document.getElementById("fp-atlas");
if (root) {
  const capturing = captureMode("atlas", root);
  fetch(new URL("./formats.json", import.meta.url))
    .then((r) => r.json())
    .then((json) => {
      data = json;
      readState();
      mount(root);
      render();
      // links from the post (…/#atlas=binary64) should land on this widget, not the top of the page
      if (!capturing && hashParams().has("atlas")) scrollToWidget(root);
      let lastW = root.clientWidth;
      new ResizeObserver(() => {
        if (Math.abs(root.clientWidth - lastW) < 2) return;
        lastW = root.clientWidth;
        render();
      }).observe(root);
      window.fpAtlas = { set: (partial) => update(partial), get state() { return { ...state }; } };
    })
    .catch(() => {
      root.textContent = "Couldn't load the format data.";
    });
}
