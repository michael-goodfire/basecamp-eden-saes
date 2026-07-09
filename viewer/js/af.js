// af.js — the 3D protein-structure panel (NGL), lazy-loaded.
//
// Structures are precomputed and served offline (issue-49): a per-protein
// struct_index.json maps each coding exemplar's protein id to a cached mmCIF in
// af_cache/ tagged by source (AlphaFold DB or ESMFold2) and mean pLDDT. The panel
// reads that cache first and only falls back to a live lookup for a protein the
// index doesn't cover. Non-coding features (no protein) show an explicit
// "no structure expected" state rather than an empty panel.
//
// Two correctness guards are essential (both were bugs in earlier versions):
//   (a) zero-height WebGL canvas — the container starts at height:0; before
//       constructing the NGL.Stage we set an explicit height and await a
//       double-rAF so the layout is real, plus a ResizeObserver to catch the
//       first non-zero layout, then handleResize()/autoView() once settled.
//   (b) double-load / stale-load on rapid paging — a debounce plus a
//       monotonically-increasing token; in-flight loads whose token is stale
//       are dropped, and the previous stage/observer is disposed before a new
//       build.

import { $, esc, VIRIDIS, DOMPAL } from "./data.js";

// ---- module-private state ------------------------------------------------
let NGLP = null;          // memoised NGL script-load promise
let AFSEQ = 0;            // monotonic exemplar token
let AFSTAGE = null;       // current NGL.Stage
let AFRO = null;          // current ResizeObserver
let AFREP = null;         // current cartoon representation
let AFSCHEMES = null;     // {act:{scheme,legend}, dom:{scheme,legend}}
let STRUCTIDXP = null;    // memoised struct_index.json load promise
const AFCACHE = {};
const INACTIVE_GRAY = 0xBFBFBF;

const SOURCE_LABEL = { afdb: "AlphaFold DB", esmfold2: "ESMFold2" };
const SOURCE_COLOR = { afdb: "#4E728A", esmfold2: "#2E6E4E", live: "#988453" };

const raf2 = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

// Called by main on every exemplar change to invalidate in-flight loads.
export function nextAFToken() { return ++AFSEQ; }

function disposeAF() {
  if (AFRO) { try { AFRO.disconnect(); } catch (_) {} AFRO = null; }
  if (AFSTAGE) { try { AFSTAGE.dispose(); } catch (_) {} AFSTAGE = null; }
  AFREP = null; AFSCHEMES = null;
}

// Lazy-load NGL (vendored locally for fully-offline serving) only when a
// structure actually renders.
function ensureNGL() {
  return NGLP || (NGLP = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "js/vendor/ngl.js";
    s.onload = () => res(window.NGL);
    s.onerror = () => rej(new Error("NGL load failed"));
    document.head.appendChild(s);
  }));
}

// Load the precomputed per-protein structure index once (offline, co-located).
function ensureStructIndex() {
  return STRUCTIDXP || (STRUCTIDXP = fetch("struct_index.json")
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({})));
}

// offline-only read from the co-located cache, validated by the CIF magic prefix.
async function fetchCachedCif(key) {
  if (key in AFCACHE) return AFCACHE[key];
  try {
    const c = await fetch(`af_cache/${key}.cif`);
    if (c.ok) { const t = await c.text(); if (t.startsWith("data_")) { AFCACHE[key] = t; return t; } }
  } catch (e) {}
  AFCACHE[key] = null;
  return null;
}

// residue (1-based) -> max feature activation over the codon's 3 nt.
function residueActivation(e) {
  const p = e.prot, act = {};
  if (!p) return act;
  const resOf = (abs) => (p.st === "+") ? Math.floor((abs - p.gs) / 3) + 1 : Math.floor((p.ge - 1 - abs) / 3) + 1;
  if (e._dense) {
    const D = e._dense;
    for (let rel = 0; rel < D.length; rel++) {
      const v = D[rel];
      if (v <= 0) continue;
      const abs = e.start + rel;
      if (abs < p.gs || abs >= p.ge) continue;
      const res = resOf(abs);
      if (act[res] === undefined || v > act[res]) act[res] = v;
    }
    return act;
  }
  for (let k = 0; k < e.track_rel.length; k++) {
    const abs = e.start + e.track_rel[k];
    if (abs < p.gs || abs >= p.ge) continue;
    const res = resOf(abs), v = e.track_act[k];
    if (act[res] === undefined || v > act[res]) act[res] = v;
  }
  return act;
}

// viridis lerp: t in [0,1] -> 0xRRGGBB.
function actColor(t) {
  t = Math.max(0, Math.min(1, t));
  const s = t * (VIRIDIS.length - 1), i = Math.floor(s), u = s - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * u);
  const g = Math.round(a[1] + (b[1] - a[1]) * u);
  const bl = Math.round(a[2] + (b[2] - a[2]) * u);
  return (r << 16) | (g << 8) | bl;
}

// residue -> domain colour + ordered domain list (for scheme + legend).
function afDomainColors(e) {
  const doms = (e.prot && e.prot.domains) || [], rc = {};
  const items = doms.map((d, i) => {
    const col = DOMPAL[i % DOMPAL.length];
    for (let r = d.res_lo; r <= d.res_hi; r++) rc[r] = col;
    return Object.assign({ col }, d);
  });
  return { rc, items };
}

// Build both colour schemes (activation + fold-domain) and their legends.
function buildAFSchemes(e, NGL) {
  const act = residueActivation(e);
  const vals = Object.values(act).filter((v) => v > 0).sort((a, b) => a - b);
  const lo = vals.length ? vals[0] : 0;
  const hi = vals.length ? vals[Math.min(vals.length - 1, Math.floor((vals.length - 1) * 0.95))] : 1;
  const span = Math.max(1e-6, hi - lo);

  const actScheme = NGL.ColormakerRegistry.addScheme(function () {
    this.atomColor = function (atom) {
      const val = act[atom.resno];
      if (val === undefined || val <= 0) return INACTIVE_GRAY;
      return actColor((val - lo) / span);
    };
  });
  const actLegend =
    `residues colored by dense activation (viridis: <span style="color:#440154">■</span> low → ` +
    `<span style="color:#21918c">■</span> mid → <span style="color:#fde725;background:#555;padding:0 2px">■</span> high), ` +
    `codon-max per residue, scaled to p95` + (vals.length ? ` (range ${lo.toFixed(2)}–${hi.toFixed(2)})` : ``) +
    `; <span style="color:#8f8f8f">■</span> = inactive.`;

  const { rc, items } = afDomainColors(e);
  const domScheme = NGL.ColormakerRegistry.addScheme(function () {
    this.atomColor = function (atom) {
      return rc[atom.resno] !== undefined ? rc[atom.resno] : INACTIVE_GRAY;
    };
  });
  const hex = (c) => "#" + (c >>> 0).toString(16).padStart(6, "0");
  const domLegend = items.length
    ? `residues shaded by structural fold domain (CATH + TED): ` + items.map((d) =>
        `<span style="color:${hex(d.col)}">■</span> ${esc(d.name)} <span style="color:var(--muted)">(${d.src}, res ${d.res_lo}–${d.res_hi})</span>`).join(" · ") +
      `; <span style="color:#8f8f8f">■</span> = no domain.`
    : `no CATH/TED fold domains mapped for this protein.`;

  return { act: { scheme: actScheme, legend: actLegend }, dom: { scheme: domScheme, legend: domLegend } };
}

function applyAFColor(mode) {
  if (!AFREP || !AFSCHEMES) return;
  const m = AFSCHEMES[mode] ? mode : "act";
  try { AFREP.setColor(AFSCHEMES[m].scheme); } catch (_) {}
  const leg = $("#aflegend");
  if (leg) { leg.style.display = "block"; leg.innerHTML = AFSCHEMES[m].legend; }
  const sel = $("#afcolor");
  if (sel) sel.value = m;
}

// A small coloured provenance badge: structure source + mean pLDDT.
function sourceBadge(source, plddt, truncated) {
  const col = SOURCE_COLOR[source] || SOURCE_COLOR.live;
  const lab = SOURCE_LABEL[source] || "AlphaFold DB (live)";
  const pl = (plddt != null && isFinite(+plddt))
    ? ` <span style="color:var(--muted)">· mean pLDDT ${(+plddt).toFixed(1)}</span>` : "";
  const tr = truncated ? ` <span style="color:#a4553b">· truncated to model context</span>` : "";
  return `<span style="display:inline-block;font-size:10.5px;font-weight:600;color:#fff;background:${col};` +
    `border-radius:3px;padding:1px 6px;letter-spacing:.02em">${esc(lab)}</span>${pl}${tr}`;
}

// Render the panel shell and schedule a debounced, token-guarded load.
export function renderAFPanel(e, token) {
  const el = $("#afpanel");
  disposeAF(); // tear down the previous exemplar's stage/observer
  if (!e.prot || !e.prot.id) {
    // non-coding feature: there is no protein to fold — say so explicitly.
    el.innerHTML = `<div style="margin-top:12px;padding:10px 12px;border:1px dashed var(--rule);border-radius:6px;background:#fafafa">
      <b style="font-size:13px">3D structure</b>
      <span id="afstatus" style="font-size:11.5px;color:var(--muted);margin-left:8px">no structure expected — this window is non-coding (no protein).</span></div>`;
    return;
  }
  const nres = Object.keys(residueActivation(e)).length;
  const ndom = ((e.prot.domains) || []).length;
  const domToggle = ndom
    ? `<label style="font-size:11.5px;color:var(--muted);margin-left:auto">color: ` +
      `<select id="afcolor" style="font-size:11.5px"><option value="act">activation</option>` +
      `<option value="dom">domains (${ndom})</option></select></label>`
    : ``;
  el.innerHTML = `<div style="margin-top:12px;padding:10px 12px;border:1px solid var(--rule);border-radius:6px;background:#fff">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <b style="font-size:13px">3D structure</b>
      <span style="font-size:11.5px;color:var(--muted)">${esc(e.prot.product || e.prot.id)} · ${nres} activated residue(s) in this window</span>
      <span id="afbadge"></span>
      <span id="afstatus" style="font-size:11.5px;color:var(--muted)">loading…</span>${domToggle}</div>
    <div id="afview" style="width:100%;height:0;margin-top:8px;border-radius:5px;overflow:hidden;transition:height .1s"></div>
    <div id="aflegend" style="display:none;font-size:11px;color:var(--muted);margin-top:6px"></div></div>`;
  // debounce so rapid paging doesn't storm the APIs; token guard drops loads
  // whose exemplar is no longer on screen.
  setTimeout(() => { if (token === AFSEQ) loadAF(e, token); }, 180);
}

async function loadAF(e, token) {
  const cur = () => token === AFSEQ;
  const st = () => $("#afstatus"), view = () => $("#afview");
  const set = (h) => { const s = st(); if (cur() && s) s.innerHTML = h; };
  const badge = (h) => { const b = $("#afbadge"); if (cur() && b) b.innerHTML = h; };
  const retry = `<a href="#" id="afretry">retry</a>`;

  // Consult the precomputed structure index (offline). A coding protein with no
  // index entry is a known gap (no AlphaFold DB model) — shown as
  // "no structure available", distinct from a non-coding "no structure expected".
  set("resolving structure…");
  const idx = await ensureStructIndex();
  if (!cur()) return;
  const rec = idx[e.prot.id];
  if (!rec) {
    set(`no structure available — no AlphaFold DB model for this protein (can be folded later).`);
    return;
  }

  const source = rec.source, plddt = rec.plddt, truncated = !!rec.truncated;
  set(`loading ${SOURCE_LABEL[source] || source} model…`);
  const cif = await fetchCachedCif(rec.key);
  if (!cur()) return;
  if (!cif) { set(`structure file missing from cache for ${esc(e.prot.id)}. ${retry}`); wireRetry(e); return; }
  badge(sourceBadge(source, plddt, truncated));

  let NGL;
  try { NGL = await ensureNGL(); }
  catch (err) { if (cur()) { set(`3D viewer failed to load. ${retry}`); wireRetry(e); } return; }
  if (!cur()) return;

  set(`rendering…`);
  disposeAF(); // tear down any prior stage before building a new one

  // GUARD (a): size the container synchronously and wait for real layout so the
  // WebGL canvas initializes with non-zero dimensions.
  const v = view();
  const leg = $("#aflegend"); if (leg) leg.style.display = "block";
  v.style.transition = "none";
  v.style.height = "340px";
  v.innerHTML = "";
  await raf2();
  if (!cur()) { v.style.height = "0"; return; }

  AFSCHEMES = buildAFSchemes(e, NGL);
  const stage = new NGL.Stage(v, { backgroundColor: "white" });
  AFSTAGE = stage;
  // repaint on the first non-zero layout (auto-load can precede layout).
  AFRO = new ResizeObserver(() => { try { stage.handleResize(); } catch (_) {} });
  AFRO.observe(v);

  try {
    const comp = await stage.loadFile(new Blob([cif], { type: "text/plain" }), { ext: "cif" });
    if (!cur() || AFSTAGE !== stage) { try { stage.dispose(); } catch (_) {} return; }
    AFREP = comp.addRepresentation("cartoon", { color: AFSCHEMES.act.scheme });
    const csel = $("#afcolor");
    if (csel) csel.onchange = () => applyAFColor(csel.value);
    applyAFColor((csel && csel.value) || "act");
    await raf2(); // let the canvas settle at final size
    if (!cur() || AFSTAGE !== stage) { try { stage.dispose(); } catch (_) {} return; }
    stage.handleResize();
    comp.autoView();
    set(`${SOURCE_LABEL[source] || "AlphaFold"} model rendered`);
  } catch (err) {
    if (cur()) { set("failed to render structure."); v.style.height = "0"; }
    disposeAF();
  }
}

function wireRetry(e) {
  const a = $("#afretry");
  if (a) a.onclick = (ev) => { ev.preventDefault(); loadAF(e, AFSEQ); };
}
