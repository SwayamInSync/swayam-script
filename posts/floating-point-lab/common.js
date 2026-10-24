// Shared helpers for the Floating Point Lab widgets.

export const el = (tag, attrs = {}, ...kids) => {
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
export const sv = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
export const sup = (k) => String(k).replace(/[-0-9]/g, (c) => SUP[c]);

// Widgets write exponents as Unicode superscripts (2⁻⁵³), but fonts draw ¹²³ (Latin-1) and ⁰⁴…⁹ (U+2070 block) at
// different sizes, so 10¹⁶ looks uneven. Every run of them on the page becomes a real superscript instead:
// - in HTML, a <sup> (hidden from screen readers) next to a visually hidden copy of the original characters;
// - in SVG, a smaller <tspan> raised with dy, and the text after it lowered back (Firefox ignores baseline-shift).
// Copying text puts the Unicode characters back.
const PLAIN = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "−", "ᵉ": "e" };
const RUN = /[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵉ]+/g;
const HAS_RUN = /[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵉ]/;
const NO_LIFT = new Set(["SCRIPT", "STYLE", "TEXTAREA", "OPTION", "TITLE", "SUP"]);
const UP = 0.5; // the raise, in the superscript's own em (its font-size is 72%), so 0.36em of the text around it

function liftText(t) {
  const parent = t.parentNode;
  if (!parent || NO_LIFT.has(parent.nodeName.toUpperCase())) return;
  if (parent.classList?.contains("fpu-sup") || parent.classList?.contains("fpu-sr")) return;
  const s = t.nodeValue;
  if (!HAS_RUN.test(s)) return; // (not RUN.test: a /g regex's lastIndex would make matchAll skip the first run)
  const inSvg = parent.namespaceURI === SVG;
  // in HTML, one <span> holds the pieces, so a flex or grid parent still sees a single item
  const frag = inSvg ? document.createDocumentFragment() : document.createElement("span");
  let raised = false;
  const plain = (text) => {
    if (!raised) return frag.append(text);
    const back = sv("tspan", { dy: `${UP * 0.72}em` });
    back.textContent = text;
    frag.append(back);
    raised = false;
  };
  let pos = 0;
  for (const m of s.matchAll(RUN)) {
    if (m.index > pos) plain(s.slice(pos, m.index));
    const digits = [...m[0]].map((c) => PLAIN[c]).join("");
    if (inSvg) {
      const n = sv("tspan", { dy: `-${UP}em`, "font-size": "72%", class: "fpu-sup", "data-raw": m[0] });
      n.textContent = digits;
      frag.append(n);
      raised = true;
    } else {
      frag.append(el("span", { class: "fpu-sr" }, m[0]), el("sup", { class: "fpu-sup", "aria-hidden": "true", "data-raw": m[0] }, digits));
    }
    pos = m.index + m[0].length;
  }
  if (pos < s.length) plain(s.slice(pos));
  else if (raised && t.nextSibling) plain("​"); // lower whatever follows in the same <text>
  parent.replaceChild(frag, t);
}

function liftAll(node) {
  if (node.nodeType === Node.TEXT_NODE) return liftText(node);
  if (node.nodeType !== Node.ELEMENT_NODE || NO_LIFT.has(node.nodeName.toUpperCase())) return;
  const w = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  const texts = [];
  for (let t = w.nextNode(); t; t = w.nextNode()) texts.push(t);
  texts.forEach(liftText);
}

if (typeof document !== "undefined" && !window.fpuSupLifter) {
  window.fpuSupLifter = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "characterData") liftText(m.target);
      else m.addedNodes.forEach(liftAll);
    }
  });
  window.fpuSupLifter.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  liftAll(document.body);
  document.addEventListener("copy", (e) => {
    const sel = document.getSelection();
    if (!sel || sel.isCollapsed) return;
    const box = el("div", { "aria-hidden": "true", style: "position:fixed;left:-100000px;top:0;width:900px" });
    for (let i = 0; i < sel.rangeCount; i++) box.append(sel.getRangeAt(i).cloneContents());
    if (!box.querySelector(".fpu-sup")) return;
    box.querySelectorAll(".fpu-sr").forEach((n) => n.remove());
    box.querySelectorAll(".fpu-sup").forEach((n) => n.replaceWith(n.dataset.raw ?? n.textContent));
    // innerText needs the box laid out, and keeps the line and table-cell breaks (and leaves out <script> text);
    // it's read before the lifter's next run, which happens after this handler returns
    document.body.append(box);
    const text = box.innerText.replace(/​/g, "");
    const htmlText = box.innerHTML;
    box.remove();
    e.clipboardData.setData("text/plain", text);
    e.clipboardData.setData("text/html", htmlText);
    e.preventDefault();
  });
}

// Every widget on the page shares the URL hash; each reads and writes only its own keys.
export const hashParams = () => new URLSearchParams(location.hash.slice(1));

/** obj[key] if key is obj's own property (so "constructor" or "__proto__" in a URL can't match). */
export const own = (obj, key) => (key != null && Object.hasOwn(obj, key) ? obj[key] : undefined);

// Writes are batched: browsers rate-limit history.replaceState (Chrome drops calls past about 200 in 10 s), and a
// slider drag writes on every input event. Pending keys from every widget are kept until the next flush.
const pendingHash = new Map();
let hashTimer = 0;

function flushHash() {
  hashTimer = 0;
  const h = hashParams();
  for (const [k, v] of pendingHash) {
    h.delete(k);
    if (v !== undefined && v !== null && v !== "") h.set(k, v);
  }
  try {
    history.replaceState(null, "", "#" + h.toString());
    pendingHash.clear();
  } catch {
    // over the browser's limit: the keys stay pending, and the next write flushes them too
  }
}

export function writeHash(ownKeys, values) {
  for (const k of new Set([...ownKeys, ...Object.keys(values)])) pendingHash.set(k, values[k]);
  clearTimeout(hashTimer);
  hashTimer = setTimeout(flushHash, 150);
}

/** A link into the lab scrolls to the widget its hash is for; if it holds several widgets' keys, to the first. */
export function scrollToWidget(root) {
  if (window.fpuScrolled) return;
  window.fpuScrolled = true;
  root.scrollIntoView({ block: "start" });
}

// ?capture=<name> shows only that widget at a fixed width, for recording GIFs.
// &w=<px> sets the width; &hide=a,b,c is handed to the widget as data-hide.
export function captureMode(name, root) {
  const q = new URLSearchParams(location.search);
  if (!root || q.get("capture") !== name) return false;
  const wrap = el("div", { class: "fpu-capture-wrap" });
  wrap.style.width = `${parseInt(q.get("w"), 10) || 960}px`;
  wrap.append(root);
  // the page's <link> to lab.css lives in the body; keep it before clearing the body
  for (const l of document.body.querySelectorAll('link[rel="stylesheet"]')) document.head.append(l);
  document.body.replaceChildren(wrap);
  document.body.classList.add("fpu-capture");
  root.dataset.hide = (q.get("hide") || "").split(",").join(" ");
  return true;
}
