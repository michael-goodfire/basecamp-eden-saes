// data.js — fetching, binary decode, shared state, name resolution, palettes.
//
// This is the foundation module: every other module imports the shared `state`
// object and the palette/format/name helpers from here. All fetches use
// RELATIVE URLs so the app is fully relocatable.

// ---------------------------------------------------------------------------
// Shared mutable state. Exported by reference; mutate fields in place so all
// modules observe updates without re-importing.
// ---------------------------------------------------------------------------
export const state = {
  DICTS: [],          // [{id,label,model,headline}]
  DICT: null,         // active dictionary id
  META: {},           // <dict>/index.json
  INDEX: [],          // <dict>/index_rows.json
  RATES: {},          // <dict>/rates.json — {feat:{ann:[pos_rate,bg_rate]}} (position enrichment)
  SPAN: {},           // <dict>/span_rates.json — {feat:{ann:[precision_fold,cover_rate,bg_rate]}} (span enrichment)
  NAMES: { pfam: {}, go: {}, pfam2go: {} },  // shared names.json
  CUR: null,          // active feature detail object (with decoded exemplars)
  EXI: 0,             // active exemplar index
  VIEW: null,         // [v0, v1] window-relative nt view range
};

// ---------------------------------------------------------------------------
// Palettes / constants shared by logos, tracks and the structure panel.
// ---------------------------------------------------------------------------
export const NT_COL = { A: "var(--a)", C: "var(--c)", G: "var(--g)", T: "var(--t)", N: "#999" };
export const AA_GRP = {
  G: "np", A: "np", V: "np", L: "np", I: "np", P: "np", F: "np", M: "np", W: "np",
  S: "pol", T: "pol", C: "pol", Y: "pol", N: "pol", Q: "pol",
  K: "pos", R: "pos", H: "pos", D: "neg", E: "neg",
};
export const GRP_COL = { np: "#555", pol: "#3f8a5a", pos: "#4E728A", neg: "#a4553b" };
export const CODCOL = { 1: "#4E728A", 2: "#C4650D", 3: "#4a6b52" }; // codon pos 1/2/3

// viridis (perceptually-uniform) control points -> used by the structure panel.
export const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];
// distinct hues for structural-domain shading of the AlphaFold cartoon.
export const DOMPAL = [
  0x4E728A, 0xC4650D, 0x4a6b52, 0x7b5ea7, 0xb0563a, 0x2f8f6a, 0xb08a2b,
  0x9a5ea0, 0x3a8a8a, 0xa4553b, 0x6b8f2f, 0x8a5a9a,
];
// human-readable labels for #38 v2 regulatory/operon/replication ann_types.
// motif ann_types are the consensus sequence itself, shown verbatim.
export const V2NAMES = {
  rbs: "ribosome binding site", terminator: "transcription terminator",
  "promoter_-10": "promoter −10 box", "promoter_-35": "promoter −35 box",
  operon: "operon", operon_internal: "operon (internal gene)",
  oriC: "replication origin (oriC)", ter: "replication terminus (ter)",
  dnaA_box: "DnaA box",
};
export function v2name(t) { return V2NAMES[t] || t; }

// ---------------------------------------------------------------------------
// Formatting helpers.
// ---------------------------------------------------------------------------
export const $ = (s) => document.querySelector(s);

export function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
export function fmt(n) { return Number(n).toLocaleString(); }
export function clip(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s;
}
// scientific notation for tiny q-values; plain decimals for readable ranges.
export function sci(x) {
  x = +x;
  if (!isFinite(x)) return "—";
  if (x === 0) return "0";
  if (x >= 0.01) return x.toFixed(3);
  const e = x.toExponential(1);
  return e.replace("e", "×10^").replace("^-", "⁻")
    .replace(/\^(\d)/, (m, d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
}
// enrichment fold: already Haldane-bounded/finite server-side — just display it
// cleanly (large values as rounded integers, otherwise one decimal).
export function fmtFold(x) {
  x = +x;
  if (!isFinite(x)) return "—";
  if (x >= 100) return Math.round(x).toLocaleString() + "×";
  return x.toFixed(1) + "×";
}

// ---------------------------------------------------------------------------
// Binary decode: base64 -> typed arrays, IEEE half-float -> float32.
// ---------------------------------------------------------------------------
export function b64u8(b) {
  const s = atob(b), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u;
}
export function f16(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 6.103515625e-5 * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
export function b64f16(b) {
  const s = atob(b), n = s.length >> 1, o = new Float32Array(n);
  for (let i = 0; i < n; i++)
    o[i] = f16(s.charCodeAt(2 * i) | (s.charCodeAt(2 * i + 1) << 8));
  return o;
}

// ---------------------------------------------------------------------------
// Fetch helpers (relative URLs).
// ---------------------------------------------------------------------------
export async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
// Decode the per-nt binary arrays on each exemplar once, up front.
export function decodeExemplars(det) {
  for (const e of det.exemplars || []) {
    e._gc = e.gc ? b64u8(e.gc) : new Uint8Array(0);
    if (e.dense) e._dense = b64f16(e.dense);
  }
  return det;
}

// ---------------------------------------------------------------------------
// Name resolution. Accession keys in names.json are UNVERSIONED — strip `.NN`.
// ---------------------------------------------------------------------------
const bareAcc = (acc) => String(acc).split(".")[0];
export function pfName(acc) { return state.NAMES.pfam[bareAcc(acc)] || acc; }
export function goName(id) { return state.NAMES.go[id] || id; }
export function pfGO(acc) { return state.NAMES.pfam2go[bareAcc(acc)] || []; }

// raw index-row label -> display name (list column / search).
export function labelName(lab, cls) {
  if (cls === "pfam") return pfName(lab);
  if (cls === "go") return goName(lab);
  return String(lab).replace(/_/g, " ");
}

// A `detected` annotation row -> {name, sub, go[]} for the annotation table.
export function detName(a) {
  if (a.class === "pfam") {
    const acc = a.id && a.id.includes("|") ? a.id.split("|")[1] : a.pretty;
    const go = pfGO(acc).slice(0, 2).map(goName);
    return { name: pfName(acc), sub: acc + (go.length ? " · " + go.join(", ") : ""), go };
  }
  if (a.class === "go") return { name: goName(a.pretty), sub: a.id, go: [] };
  return { name: String(a.pretty || a.id || "").replace(/_/g, " "), sub: a.id, go: [] };
}
