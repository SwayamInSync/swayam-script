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

// Every widget on the page shares the URL hash; each reads and writes only its own keys.
export const hashParams = () => new URLSearchParams(location.hash.slice(1));

export function writeHash(ownKeys, values) {
  const h = hashParams();
  for (const k of ownKeys) h.delete(k);
  for (const [k, v] of Object.entries(values)) if (v !== undefined && v !== null && v !== "") h.set(k, v);
  history.replaceState(null, "", "#" + h.toString());
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
