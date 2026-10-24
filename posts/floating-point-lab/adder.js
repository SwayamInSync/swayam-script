// The adder: how hardware adds two floats with only three extra bits (guard, round, sticky) and still gets the
// correctly rounded answer. Mounts into <div id="fp-adder">. Hash keys: ga, gb, gop, gp, gm, gc.
// The arithmetic is grs.js (a port of verify/grs.py, tested against it); this file only draws it.

import { addHW, isCorrect, roundExact, value, MODES, CONFIGS } from "./grs.js";
import { parseNumber, toDecimal, rat } from "./fpcore.js";
import { el, sup, hashParams, writeHash, captureMode, own, scrollToWidget } from "./common.js";

const MODE_NAMES = { RNE: "nearest, ties to even", RTZ: "toward zero", RTP: "toward +∞", RTN: "toward −∞" };
const PRESETS = [
  { label: "1 + 0.0703125: the sticky bit's save", a: "1", b: "0.0703125", op: "+", p: 4 },
  { label: "8 − 1.375: only all three bits get it", a: "8", b: "1.375", op: "−", p: 4 },
  { label: "4 − 1.375: the guard bit isn't enough", a: "4", b: "1.375", op: "−", p: 4 },
  { label: "7.5 + 1.125: a carry", a: "7.5", b: "1.125", op: "+", p: 4 },
  { label: "0.1 + 0.2 in 24 bits", a: "0.1", b: "0.2", op: "+", p: 24 },
];
const KEYS = ["ga", "gb", "gop", "gp", "gm", "gc"];
const state = { a: "1", b: "0.0703125", op: "+", p: 4, mode: "RNE", cfg: "GRS" };
const ui = {};

const dec = (r) => {
  const t = toDecimal(r, 60);
  return t.text.replace("-", "−") + (t.exact ? "" : "…");
};
const binSig = ({ M }, p) => {
  const s = M.toString(2).padStart(p, "0");
  return s[0] + (p > 1 ? "." + s.slice(1) : "");
};
const show = (x, p) => `${x.s ? "−" : ""}${binSig(x, p)} × 2${sup(x.e)}`;

/** Parse a typed number and round it to p significant bits (nearest, ties to even). */
function operand(text, p, negate) {
  const x = parseNumber(text);
  if (!x || x.mag.n === 0n) return null;
  const exact = x.neg !== negate ? { n: -x.mag.n, d: x.mag.d } : x.mag;
  const r = roundExact(exact, p, "RNE");
  const v = value(r, p);
  return { ...r, rounded: v.n * exact.d !== exact.n * v.d, typed: exact };
}

// --- registers --------------------------------------------------------------------------------------------

/** Labels for the extra-bit positions, left to right. */
function extraLabels(cfg) {
  const { extra, sticky } = CONFIGS[cfg];
  const names = ["G", "R", "S"];
  if (!extra) return [];
  if (sticky) return [...names.slice(0, extra - 1), "S"];
  return names.slice(0, extra);
}

const colWidth = (p) => Math.max(p + 3, 11);
let labelWidth = 13; // widened in render() when the shift label ("smaller >> 100") needs it

/** One register line: the significand (with its binary point), then the extra bits, coloured by role. */
function regLine(label, R, p, cfg, { intWidth = p, signText = " ", lost = null, lostWidth = 0, note = "" } = {}) {
  const { extra } = CONFIGS[cfg];
  const X = BigInt(extra);
  const ip = (R >> X).toString(2).padStart(intWidth, "0");
  const intText = ip.length > p ? ip.slice(0, ip.length - p + 1) + "." + ip.slice(ip.length - p + 1) : ip[0] + "." + ip.slice(1);
  const ex = extra ? (R & ((1n << X) - 1n)).toString(2).padStart(extra, "0") : "";
  const labs = extraLabels(cfg);
  const kids = [el("span", { class: "fpw-reg-l" }, label.padEnd(labelWidth)), signText, intText.padStart(colWidth(p))];
  if (extra) {
    kids.push(" │ ");
    [...ex].forEach((bit, i) => kids.push(el("span", { class: labs[i].toLowerCase(), title: { G: "guard", R: "round", S: "sticky" }[labs[i]] }, bit), " "));
  }
  if (lost !== null && lostWidth > 0) {
    // the bits shifted past the register, most significant first
    const all = lost.toString(2).padStart(lostWidth, "0");
    const bits = lostWidth <= 40 ? all : `${all.slice(0, 12)}… (${lostWidth} bits)`;
    kids.push(extra ? " " : "  ", "fell off: ", el("span", { class: "lost", title: "shifted past the register" }, bits));
  }
  if (note) kids.push("   ", el("span", { class: "fpu-small" }, note));
  return el("div", {}, kids);
}

function headerLine(p, cfg) {
  const labs = extraLabels(cfg);
  const kids = [" ".repeat(labelWidth), " ", "significand".padStart(colWidth(p))];
  if (labs.length) kids.push(" │ ", labs.join(" "));
  return el("div", { class: "fpu-small" }, kids);
}

// --- rendering ----------------------------------------------------------------------------------------------

function render() {
  for (const [k, b] of Object.entries(ui.cfgBtns)) b.setAttribute("aria-pressed", k === state.cfg ? "true" : "false");
  for (const [k, b] of Object.entries(ui.modeBtns)) b.setAttribute("aria-pressed", k === state.mode ? "true" : "false");
  for (const [k, b] of Object.entries(ui.opBtns)) b.setAttribute("aria-pressed", k === state.op ? "true" : "false");
  for (const [i, b] of ui.presetBtns.entries()) {
    const q = PRESETS[i];
    b.setAttribute("aria-pressed", q.a === state.a && q.b === state.b && q.op === state.op && q.p === state.p ? "true" : "false");
  }
  if (document.activeElement !== ui.aIn) ui.aIn.value = state.a;
  if (document.activeElement !== ui.bIn) ui.bIn.value = state.b;
  ui.pIn.value = String(state.p);
  ui.pOut.textContent = `${state.p} bits`;

  const p = state.p;
  const a = operand(state.a, p, false);
  const b = operand(state.b, p, state.op === "−");
  ui.err.textContent = a && b ? "" : "Type two nonzero numbers, like 1 and 0.0703125.";
  ui.body.hidden = !(a && b);
  if (!(a && b)) return;

  const cfg = CONFIGS[state.cfg];
  const { result, trace: t } = addHW(a, b, p, state.mode, cfg.extra, cfg.sticky);
  const good = isCorrect(a, b, p, state.mode, result);

  // the operands
  const opText = (x, name, typed) => el("p", { class: "fpw-num", style: "margin:0.15rem 0" },
    `${name} = ${show(x, p)} = ${dec(value(x, p))}`,
    x.rounded ? el("span", { class: "fpu-small" }, `  (${typed} rounded to ${p} bits first)`) : null);
  ui.ops.replaceChildren(opText(a, "a", state.a), opText(b, state.op === "−" ? "−b" : "b", state.b));

  // the registers
  labelWidth = Math.max(13, `smaller >> ${t.d}`.length + 1);
  const lines = [headerLine(p, state.cfg)];
  const sgn = (s) => (s ? "−" : "+");
  lines.push(regLine(`larger`, t.A, p, state.cfg, { signText: sgn(t.big.s) }));
  lines.push(regLine(t.d ? `smaller >> ${t.d}` : "smaller", t.B, p, state.cfg, {
    signText: sgn(t.small.s), lost: t.lost >> BigInt(cfg.extra), lostWidth: Math.max(0, t.d - cfg.extra),
    note: t.d && t.lost ? (cfg.sticky ? "(the sticky bit records that a 1 fell off)" : "(forgotten)") : "",
  }));
  const intWidth = Math.max(p, (t.S0 >> BigInt(cfg.extra)).toString(2).length);
  lines.push(regLine(t.subtract ? "difference" : "sum", t.S0, p, state.cfg, { intWidth, signText: sgn(t.big.s) }));
  if (!t.zero) {
    const rs = t.shifts.filter((x) => x.dir === "right").length;
    const ls = t.shifts.filter((x) => x.dir === "left").length;
    if (rs || ls) {
      lines.push(regLine("normalised", t.S, p, state.cfg, {
        signText: sgn(t.big.s), note: rs ? "carry: shifted right one place" : `cancellation: shifted left ${ls} place${ls > 1 ? "s" : ""}, zeros came in`,
      }));
    }
  }
  ui.regs.replaceChildren(...lines);

  // the rounding decision
  const steps = [];
  if (t.zero) {
    steps.push(state.mode === "RTN"
      ? "The two cancel exactly. An exact zero sum is −0 when rounding toward −∞, so the result is −0."
      : "The two cancel exactly. An exact zero sum is +0 in every rounding mode except toward −∞, so the result is +0.");
  } else {
    const remBits = cfg.extra ? t.rem.toString(2).padStart(cfg.extra, "0") : "";
    if (!cfg.extra) steps.push("With no extra bits, the hardware can't see anything past the significand, so it keeps what it has (it chops).");
    else if (state.mode === "RNE") {
      const halfBits = "1".padEnd(cfg.extra, "0");
      if (t.rem === 0n) steps.push(`The extra bits are ${remBits}, so as far as the hardware can tell the result is exact: nothing to round.`);
      else if (t.tie) steps.push(`The extra bits are ${remBits}, which looks exactly halfway (${halfBits}), so it's treated as a tie and goes to the even neighbour: ${t.up ? "up" : "down"}.`);
      else steps.push(`The extra bits are ${remBits}: ${t.up ? "more" : "less"} than halfway (${halfBits}), so it rounds ${t.up ? "up" : "down"}.`);
    } else {
      const dir = { RTZ: "toward zero", RTP: "toward +∞", RTN: "toward −∞" }[state.mode];
      steps.push(t.rem === 0n ? `The extra bits are ${remBits}, so as far as the hardware can tell the result is exact: nothing to round.` : `The extra bits are ${remBits}, not zero, so it rounds ${dir}: ${t.up ? "the significand goes up by one" : "it keeps the significand"}.`);
    }
    if (t.roundCarry) steps.push("Rounding up carried out of the top bit, so the exponent goes up by one.");
  }
  const exactSum = (() => {
    const va = value(a, p);
    const vb = value(b, p);
    return rat(va.n * vb.d + vb.n * va.d, va.d * vb.d);
  })();
  const correct = exactSum.n === 0n ? null : roundExact(exactSum, p, state.mode);
  const zero = state.mode === "RTN" ? "−0" : "+0"; // an exact zero sum (x + (−x))
  const resText = result ? `${show(result, p)} = ${dec(value(result, p))}` : zero;
  const corrText = correct ? `${show(correct, p)} = ${dec(value(correct, p))}` : zero;
  ui.round.replaceChildren(
    ...steps.map((s) => el("p", { style: "margin:0.2rem 0" }, s)),
    el("p", { class: "fpw-num", style: "margin:0.4rem 0 0" }, el("strong", {}, "hardware result: "), resText),
    el("p", { class: "fpw-num", style: "margin:0.1rem 0" }, el("strong", {}, "exact sum: "), dec(exactSum), "  →  correctly rounded: ", corrText),
    el("p", { style: "margin:0.3rem 0 0" }, good ? el("span", { class: "fpw-yes" }, "✓ correctly rounded") : el("span", { class: "fpw-no" }, "✗ wrong: these extra bits weren't enough here")),
  );

  // every configuration, side by side
  const rows = Object.entries(CONFIGS).map(([k, c]) => {
    const r = addHW(a, b, p, state.mode, c.extra, c.sticky).result;
    const ok = isCorrect(a, b, p, state.mode, r);
    return el("tr", { class: k === state.cfg ? "fpw-sel" : "" },
      el("td", { "data-label": "extra bits" }, c.label),
      el("td", { class: "fpw-num", "data-label": "result" }, r ? dec(value(r, p)) : zero),
      el("td", { "data-label": "correct?" }, ok ? el("span", { class: "fpw-yes" }, "✓") : el("span", { class: "fpw-no" }, "✗")));
  });
  ui.table.replaceChildren(el("table", { class: "fpw-table fpw-stack" },
    el("thead", {}, el("tr", {}, ["extra bits", "result", "correct?"].map((h) => el("th", {}, h)))), el("tbody", {}, rows)));
}

function update(partial) {
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { ga: state.a, gb: state.b, gop: state.op === "−" ? "-" : "+", gp: state.p, gm: state.mode, gc: state.cfg });
}

function mount(root) {
  ui.aIn = el("input", { type: "text", class: "fpu-xin", spellcheck: "false", "aria-label": "a", oninput: (e) => update({ a: e.target.value.trim() }) });
  ui.bIn = el("input", { type: "text", class: "fpu-xin", spellcheck: "false", "aria-label": "b", oninput: (e) => update({ b: e.target.value.trim() }) });
  ui.aIn.style.width = ui.bIn.style.width = "9rem";
  ui.opBtns = Object.fromEntries(["+", "−"].map((o) => [o, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ op: o }) }, o)]));
  ui.pIn = el("input", { type: "range", min: "2", max: "24", step: "1", "aria-label": "Significand bits", oninput: (e) => update({ p: parseInt(e.target.value, 10) }) });
  ui.pOut = el("output", { class: "fpu-mono" });
  ui.modeBtns = Object.fromEntries(MODES.map((m) => [m, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", title: MODE_NAMES[m], onclick: () => update({ mode: m }) }, m)]));
  ui.cfgBtns = Object.fromEntries(Object.entries(CONFIGS).map(([k, c]) => [k, el("button", { type: "button", class: "fpu-btn fpu-btn-sm", onclick: () => update({ cfg: k }) }, c.label)]));
  ui.presetBtns = PRESETS.map((q) => el("button", { type: "button", class: "fpw-chip", onclick: () => update({ a: q.a, b: q.b, op: q.op, p: q.p }) }, q.label));
  ui.err = el("p", { class: "fpw-note", "aria-live": "polite" });
  ui.ops = el("div", {});
  ui.regs = el("div", { class: "fpw-reg" });
  ui.round = el("div", {});
  ui.table = el("div", {});
  ui.body = el("div", { class: "fpw", style: "margin:0" },
    el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "Inside the adder"), ui.ops, ui.regs,
      el("p", { class: "fpw-note" }, "The larger operand stays put; the smaller one's significand shifts right until the binary points line up. Bits past the register's end are lost, except that a sticky bit remembers whether any of them was a 1.")),
    el("div", { class: "fpw-grid2" },
      el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "Rounding"), ui.round),
      el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "Every choice of extra bits, same inputs"), ui.table,
        el("p", { class: "fpw-note" }, "Exponents are unbounded here: the adder is about alignment and rounding, not overflow or underflow."))));
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" },
      el("div", { class: "fpu-knobs" },
        el("label", { class: "fpu-knob" }, "a = ", ui.aIn), el("span", { class: "fpu-toggles" }, Object.values(ui.opBtns)),
        el("label", { class: "fpu-knob" }, "b = ", ui.bIn),
        el("label", { class: "fpu-knob" }, "significand ", ui.pIn, " ", ui.pOut)),
      el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small" }, "rounding:"), Object.values(ui.modeBtns)),
      el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small" }, "extra bits:"), Object.values(ui.cfgBtns)),
      el("div", { class: "fpu-presets fpw-part-presets" }, el("span", { class: "fpu-small" }, "try:"), ui.presetBtns),
      ui.err),
    ui.body));
}

const root = document.getElementById("fp-adder");
if (root) {
  const capturing = captureMode("adder", root);
  const h = hashParams();
  if (h.has("ga")) state.a = h.get("ga");
  if (h.has("gb")) state.b = h.get("gb");
  if (h.has("gop")) state.op = h.get("gop") === "-" ? "−" : "+";
  if (h.has("gp")) state.p = Math.min(24, Math.max(2, parseInt(h.get("gp"), 10) || 4));
  if (h.has("gm") && MODES.includes(h.get("gm"))) state.mode = h.get("gm");
  if (own(CONFIGS, h.get("gc"))) state.cfg = h.get("gc");
  mount(root);
  render();
  if (!capturing && KEYS.some((k) => h.has(k))) scrollToWidget(root);
  window.fpAdder = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
