// The comparison playground: one pair of doubles, every common "are they close?" test, and why they disagree.
// Mounts into <div id="fp-compare">. Hash keys: ca, cb, crel, cabs. window.fpCompare drives it.
// The tests run in binary64, the way the libraries themselves run them (closeness.js, tested against them).

import { epsEqual, pyIsclose, npIsclose, npAssertAllclose, pytestApprox, juliaIsapprox, gtestDoubleEq, ulpDistance } from "./closeness.js";
import { evaluate, toDouble } from "./doubles.js";
import { jsString, pyRepr } from "./printers.js";
import { el, hashParams, writeHash, captureMode, scrollToWidget } from "./common.js";

const PRESETS = [
  { label: "0.1 + 0.2 vs 0.3", a: "0.1 + 0.2", b: "0.3" },
  { label: "1e16 vs the next double", a: "1e16", b: "10000000000000002" },
  { label: "sin(π) vs 0", a: "1.2246467991473532e-16", b: "0" },
  { label: "1e-10 vs 2e-10", a: "1e-10", b: "2e-10" },
  { label: "NumPy's asymmetry", a: "1.0", b: "1.0000100101" },
  { label: "±5e-324", a: "5e-324", b: "-5e-324" },
  { label: "largest double vs ∞", a: "1.7976931348623157e308", b: "inf" },
  { label: "NaN vs NaN", a: "nan", b: "nan" },
];
const KEYS = ["ca", "cb", "crel", "cabs"];
const state = { a: "0.1 + 0.2", b: "0.3", rel: "1e-9", abs: "0" };
const ui = {};

const yes = (t = "True") => el("span", { class: "fpw-yes" }, t);
const no = (t = "False") => el("span", { class: "fpw-no" }, t);
const tf = (v) => (v ? yes() : no());
const pf = (v, p = "passes", f = "fails") => (v ? yes(p) : no(f));
const code = (s) => el("code", {}, s);

function render() {
  if (document.activeElement !== ui.aIn) ui.aIn.value = state.a;
  if (document.activeElement !== ui.bIn) ui.bIn.value = state.b;
  if (document.activeElement !== ui.relIn) ui.relIn.value = state.rel;
  if (document.activeElement !== ui.absIn) ui.absIn.value = state.abs;
  for (const [i, btn] of ui.presetBtns.entries()) btn.setAttribute("aria-pressed", PRESETS[i].a === state.a && PRESETS[i].b === state.b ? "true" : "false");
  const ea = evaluate(state.a);
  const eb = evaluate(state.b);
  const rel = toDouble(state.rel);
  const abs = toDouble(state.abs);
  const okTol = rel !== null && abs !== null && rel >= 0 && abs >= 0 && Number.isFinite(rel) && Number.isFinite(abs);
  ui.err.textContent = !ea || !eb ? "Type two numbers (inf and nan work too), or an expression like 0.1 + 0.2 (with spaces around the operator)."
    : !okTol ? "Tolerances must be finite and ≥ 0 here (math.isclose itself raises ValueError only for negative ones)." : "";
  ui.body.hidden = !(ea && eb && okTol);
  if (!(ea && eb && okTol)) return;
  const a = ea.x;
  const b = eb.x;

  const diff = Math.abs(a - b);
  const relDiff = diff / Math.max(Math.abs(a), Math.abs(b));
  const ud = ulpDistance(a, b);
  const stats = [
    ["a", pyRepr(a)], ["b", pyRepr(b)], ["|a − b|", pyRepr(diff)],
    ["|a − b| / max(|a|, |b|)", Number.isNaN(relDiff) ? "undefined (nan)" : pyRepr(relDiff)],
    ["ULP distance", ud === null ? "none: NaN isn't on the line" : ud.toLocaleString("en-US")],
  ];
  ui.stats.replaceChildren(...stats.map(([k, v]) => el("div", { class: "fpu-stat" }, el("span", { class: "fpu-k", style: "text-transform:none" }, k), el("span", { class: "fpu-v fpw-num" }, v))));

  const R = pyRepr;
  const rows = [
    ["exact equality", code("a == b"), "the same number (NaN equals nothing)", tf(a === b)],
    ["the folk test", code("abs(a - b) < sys.float_info.epsilon"), "|a − b| < 2⁻⁵² ≈ 2.2 × 10⁻¹⁶", tf(epsEqual(a, b))],
    ["Python", code("math.isclose(a, b)"), "|a − b| ≤ 10⁻⁹ · max(|a|, |b|); ±∞ only close to itself", tf(pyIsclose(a, b))],
    ["Python, your tolerances", code(`math.isclose(a, b, rel_tol=${R(rel)}, abs_tol=${R(abs)})`), "|a − b| ≤ max(rel_tol · max(|a|, |b|), abs_tol); ±∞ only close to itself", tf(pyIsclose(a, b, rel, abs))],
    ["NumPy", code("np.isclose(a, b)"), "|a − b| ≤ 10⁻⁸ + 10⁻⁵ · |b|: measured against b; ±∞ only close to itself", tf(npIsclose(a, b))],
    ["NumPy, swapped", code("np.isclose(b, a)"), "|b − a| ≤ 10⁻⁸ + 10⁻⁵ · |a|: measured against a; ±∞ only close to itself", tf(npIsclose(b, a))],
    ["NumPy's tests", code("np.testing.assert_allclose(a, b)"), "|a − b| ≤ 10⁻⁷ · |b|, no absolute tolerance; ±∞ only close to itself; NaN equals NaN", pf(npAssertAllclose(a, b), "passes", "raises AssertionError")],
    ["pytest", code("a == pytest.approx(b)"), "|a − b| ≤ max(10⁻⁶ · |b|, 10⁻¹²); ±∞ only close to itself", tf(pytestApprox(a, b))],
    ["Julia", code("isapprox(a, b)"), "|a − b| ≤ √ε · max(|a|, |b|), √ε = 2⁻²⁶ ≈ 1.5 × 10⁻⁸; ±∞ only close to itself", tf(juliaIsapprox(a, b))],
    ["GoogleTest", code("EXPECT_DOUBLE_EQ(a, b)"), "at most 4 ulps apart, neither NaN", pf(gtestDoubleEq(a, b))],
  ];
  ui.table.replaceChildren(el("table", { class: "fpw-table fpw-stack" },
    el("thead", {}, el("tr", {}, ["", "the call", "what it checks", "says"].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.map(([who, c, what, says]) => el("tr", {},
      el("td", { "data-label": "" }, el("strong", {}, who)), el("td", { "data-label": "the call", class: "fpw-num" }, c),
      el("td", { "data-label": "what it checks" }, what), el("td", { "data-label": "says" }, says))))));

  const notes = [];
  if (npIsclose(a, b) !== npIsclose(b, a)) notes.push("NumPy's isclose gives different answers depending on the order: it scales its tolerance by |b| only.");
  if (a !== 0 && b !== 0 && Number.isFinite(a) && Number.isFinite(b) && Math.max(Math.abs(a), Math.abs(b)) < 5e-9 && npIsclose(a, b)) notes.push("Both numbers are smaller than 5 × 10⁻⁹, so their difference is below NumPy's default atol of 10⁻⁸, and np.isclose calls them close, whatever they are.");
  if ((a === 0) !== (b === 0) && Number.isFinite(a) && Number.isFinite(b) && !pyIsclose(a, b)) notes.push("Against exactly zero, a relative tolerance below 1 can never pass (it scales with the numbers): you need an absolute one, sized for your problem.");
  if (ea.op || eb.op) notes.push(`Expressions like ${ea.op ? state.a : state.b} are one binary64 operation on the two doubles nearest what you typed.`);
  ui.notes.replaceChildren(...notes.map((s) => el("p", { class: "fpw-note" }, s)));
  ui.printed.textContent = `JavaScript prints them as ${jsString(a)} and ${jsString(b)}.`;
}

function update(partial) {
  Object.assign(state, partial);
  render();
  writeHash(KEYS, { ca: state.a, cb: state.b, crel: state.rel === "1e-9" ? "" : state.rel, cabs: state.abs === "0" ? "" : state.abs });
}

function mount(root) {
  const input = (key, label, width) => {
    const n = el("input", { type: "text", class: "fpu-xin", spellcheck: "false", "aria-label": label, oninput: (e) => update({ [key]: e.target.value }) });
    n.style.width = width;
    return n;
  };
  ui.aIn = input("a", "a", "17rem");
  ui.bIn = input("b", "b", "17rem");
  ui.relIn = input("rel", "rel_tol", "6rem");
  ui.absIn = input("abs", "abs_tol", "6rem");
  ui.presetBtns = PRESETS.map((q) => el("button", { type: "button", class: "fpw-chip", onclick: () => update({ a: q.a, b: q.b }) }, q.label));
  ui.err = el("p", { class: "fpw-note", "aria-live": "polite" });
  ui.stats = el("div", { class: "fpu-stats" });
  ui.table = el("div", {});
  ui.notes = el("div", {});
  ui.printed = el("p", { class: "fpw-note" });
  ui.body = el("div", { class: "fpw", style: "margin:0" }, ui.stats,
    el("div", { class: "fpu-panel" }, el("div", { class: "fpu-panel-head" }, "Ten ways to ask “equal?”"), ui.table, ui.notes,
      el("p", { class: "fpw-note" }, "Every test runs in binary64 exactly as the library's own code runs it (same operations, same order), so True and False here are what the library returns. Inputs are rounded to the nearest double first, as Python's float() would."),
      ui.printed));
  root.replaceChildren(el("div", { class: "fpw" },
    el("div", { class: "fpu-controls" },
      el("div", { class: "fpu-knobs" }, el("label", { class: "fpu-knob" }, "a = ", ui.aIn), el("label", { class: "fpu-knob" }, "b = ", ui.bIn)),
      el("div", { class: "fpu-knobs" }, el("span", { class: "fpu-small" }, "your math.isclose tolerances:"),
        el("label", { class: "fpu-knob" }, "rel_tol ", ui.relIn), el("label", { class: "fpu-knob" }, "abs_tol ", ui.absIn)),
      el("div", { class: "fpu-presets fpw-part-presets" }, el("span", { class: "fpu-small" }, "try:"), ui.presetBtns),
      ui.err),
    ui.body));
}

const root = document.getElementById("fp-compare");
if (root) {
  const capturing = captureMode("compare", root);
  const h = hashParams();
  if (h.has("ca")) state.a = h.get("ca");
  if (h.has("cb")) state.b = h.get("cb");
  if (h.has("crel")) state.rel = h.get("crel");
  if (h.has("cabs")) state.abs = h.get("cabs");
  mount(root);
  render();
  if (!capturing && KEYS.some((k) => h.has(k))) scrollToWidget(root);
  window.fpCompare = { set: (partial) => update(partial), get state() { return { ...state }; } };
}
