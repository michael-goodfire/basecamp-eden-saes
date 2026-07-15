// main.js — app wiring: dictionary switcher, feature list/search/sort/filter,
// feature detail view (metrics, annotation table, context + correlation bars,
// logos, exemplar pager, tracks, protein + AlphaFold panels).

import {
  state, $, esc, fmt, sci, fmtFold,
  fetchJSON, decodeExemplars, labelName, detName, goName,
} from "./data.js";
import { drawLogo, ntColor, aaColor } from "./logo.js";
import { drawTracks, wireTrackInteractions, initTracks, zoomBy, resetView } from "./tracks.js";
import { renderAFPanel, nextAFToken } from "./af.js";

const ANNOTATED_CLASSES = ["pfam", "go", "keyword", "structural", "cath", "ted", "regulatory", "operon", "replication", "motif"];

// ==========================================================================
// Boot
// ==========================================================================
async function init() {
  try { state.NAMES = await fetchJSON("names.json"); } catch (e) {}
  try { state.DICTS = (await fetchJSON("dicts.json")).dicts; }
  catch (e) { state.DICTS = [{ id: ".", label: "OG2" }]; }

  const sel = $("#dictsel");
  state.DICTS.forEach((d) => {
    const o = document.createElement("option");
    o.value = d.id; o.textContent = d.label;
    sel.appendChild(o);
  });
  sel.onchange = () => loadDict(sel.value);

  ["#search", "#classfilter", "#sortby", "#hidedead", "#onlyannot"].forEach((s) => {
    $(s).addEventListener("input", renderList);
    $(s).addEventListener("change", renderList);
  });

  const ab = $("#about");
  $("#aboutlink").onclick = (e) => { e.preventDefault(); ab.style.display = "flex"; };
  $("#aboutclose").onclick = () => (ab.style.display = "none");
  ab.onclick = (e) => { if (e.target === ab) ab.style.display = "none"; };

  initTracks(); // one-time wheel/drag listeners (target persists across renders)

  document.addEventListener("keydown", (e) => {
    if (!state.CUR || !state.CUR.exemplars) return;
    if (e.key === "ArrowRight" && state.EXI < state.CUR.exemplars.length - 1) { state.EXI++; setExemplar(); }
    if (e.key === "ArrowLeft" && state.EXI > 0) { state.EXI--; setExemplar(); }
  });
  window.addEventListener("resize", () => { if (state.CUR && state.CUR.exemplars && state.VIEW) drawTracks(); });

  await loadDict(state.DICTS[0].id);
}

// ==========================================================================
// Dictionary loading
// ==========================================================================
async function loadDict(id) {
  state.DICT = id;
  $("#dictsel").value = id;
  try {
    state.META = await fetchJSON(`${id}/index.json`);
    state.INDEX = await fetchJSON(`${id}/index_rows.json`);
    // per-(feature,annotation) firing rates behind the enrichment fold (optional sidecar)
    try { state.RATES = await fetchJSON(`${id}/rates.json`); } catch (e) { state.RATES = {}; }
    // per-(feature,annotation) span-level enrichment (precision_fold + rates) sidecar
    try { state.SPAN = await fetchJSON(`${id}/span_rates.json`); } catch (e) { state.SPAN = {}; }
    // precomputed 3D-structure coverage per dictionary (optional sidecar)
    try { state.STRUCTCOV = (await fetchJSON(`struct_coverage.json`))[id] || null; } catch (e) { state.STRUCTCOV = null; }
  } catch (err) {
    state.INDEX = [];
    $("#count").textContent = "load error";
    $("#rows").innerHTML =
      `<div style="padding:14px;color:#a4553b;font-size:12px">Failed to load ` +
      `<code>${esc(id)}</code> feature index: ${esc(String(err.message || err))}. ` +
      `The index file may be missing or malformed.</div>`;
    return;
  }
  const M = state.META;
  const hl = (M.coverage_headline != null) ? ` · <b>${M.coverage_headline}%</b> biological coverage` : "";
  const SC = state.STRUCTCOV;
  const scTitle = SC
    ? `${SC.afdb_ex.toLocaleString()} of ${SC.coding_ex.toLocaleString()} coding exemplars have a precomputed AlphaFold DB structure`
      + (SC.esmfold_ex ? ` (+${SC.esmfold_ex.toLocaleString()} ESMFold2)` : ``)
      + `; ${SC.uncovered_ex.toLocaleString()} coding exemplars have no model yet (foldable later)`
    : "";
  const sc = SC ? ` · <b title="${scTitle}">${SC.coverage_pct}%</b> 3D-structure coverage` : "";
  $("#hsub").innerHTML =
    `<b>${esc(M.dict_label || M.dictionary)}</b> (${esc(M.model || "")}) · ` +
    `${Number(M.F).toLocaleString()} features · layer ${M.layer} · window ${M.window} nt${hl}${sc}`;
  $("#aboutprov").innerHTML =
    `SAE <code>${esc(M.sae || "")}</code> · layer ${M.layer} · ` +
    `${(M.n_tokens || 0).toLocaleString()} tokens · ${(M.res_n || 0).toLocaleString()} retained positions.`;

  const cf = $("#classfilter");
  cf.innerHTML = '<option value="">all classes</option>';
  [...new Set(state.INDEX.map((r) => r.top_class))].sort().forEach((c) => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    cf.appendChild(o);
  });

  state.CUR = null; state.EXI = 0; state.VIEW = null;
  $("#main").innerHTML = '<div id="empty">Select a feature from the index to browse its top-activating windows.</div>';
  renderList();
}

// ==========================================================================
// Feature list (left column)
// ==========================================================================
function renderList() {
  const q = $("#search").value.trim().toLowerCase();
  const cf = $("#classfilter").value, sb = $("#sortby").value;
  const hd = $("#hidedead").checked, oa = $("#onlyannot").checked;

  let rows = state.INDEX.filter((r) => {
    if (hd && r.dead) return false;
    if (oa && !ANNOTATED_CLASSES.includes(r.top_class)) return false;
    if (cf && r.top_class !== cf) return false;
    if (q) {
      const s = ("f" + r.feature_id + " " + labelName(r.top_annotation, r.top_class) + " " +
        r.top_annotation + " " + r.top_class).toLowerCase();
      if (!s.includes(q)) return false;
    }
    return true;
  });
  // guard against non-finite sort keys (never parse/emit Infinity).
  const key = (r) => { const v = +r[sb]; return isFinite(v) ? v : 0; };
  rows.sort((a, b) => (sb === "feature_id" ? a.feature_id - b.feature_id : key(b) - key(a)));

  $("#count").textContent = `${rows.length.toLocaleString()} features` + (rows.length > 600 ? " (showing top 600)" : "");
  const rc = $("#rows");
  rc.innerHTML = "";
  for (const r of rows.slice(0, 600)) {
    const d = document.createElement("div");
    d.className = "frow" + (state.CUR && state.CUR.feature_id === r.feature_id ? " sel" : "");
    const nm = labelName(r.top_annotation, r.top_class);
    const fold = +r.top_fold;
    d.innerHTML =
      `<div class="fid">feature ${r.feature_id}${r.dead ? " · dead" : ""}</div>` +
      `<div class="fann">${esc(nm)}</div>` +
      `<div class="fmeta"><span class="pill ${esc(r.top_class)}">${esc(r.top_class)}</span>` +
      (isFinite(fold) && fold > 1 ? `<span>${fmtFold(fold)} enr.</span>` : "") +
      `<span>fires ${fmt(r.fire_count)}</span></div>`;
    d.onclick = () => loadFeature(r.feature_id);
    rc.appendChild(d);
  }
}

// ==========================================================================
// Feature detail (right column)
// ==========================================================================
async function loadFeature(id) {
  state.CUR = { feature_id: id };
  renderList(); // reflect selection immediately
  const det = decodeExemplars(await fetchJSON(`${state.DICT}/feature/latent_${String(id).padStart(5, "0")}.json`));
  // merge in the firing-rate sidecar so the enrichment ratio is transparent
  const fr = (state.RATES || {})[String(id)] || {};
  const sp = (state.SPAN || {})[String(id)] || {};
  for (const e of (det.detected || [])) {
    if (fr[e.id]) e.rates = fr[e.id];
    if (sp[e.id]) e.span = sp[e.id];
  }
  state.CUR = det;
  state.EXI = 0;
  renderMain();
}

// build the "annotation context at firing sites" bars (continuous vs background).
function contextBars(c, bg) {
  const items = [
    ["CDS", "sense_CDS"], ["codon1", "sense_codon1"], ["codon2", "sense_codon2"],
    ["codon3", "sense_codon3"], ["intergenic", "intergenic"], ["tRNA", "sense_tRNA"],
    ["rRNA", "sense_rRNA"], ["GC", "gc"],
  ];
  return items.map(([lab, k]) => {
    const v = +c[k] || 0, b = +bg[k] || 0, w = Math.min(100, v * 100);
    return `<div style="display:flex;align-items:center;gap:6px;margin:1px 0">
      <span style="width:66px;color:var(--muted);font-size:11.5px">${lab}</span>
      <div style="flex:1;background:#eee;height:9px;border-radius:2px;position:relative">
        <div style="width:${w}%;background:var(--blue);height:9px;border-radius:2px"></div>
        <div style="position:absolute;left:${Math.min(100, b * 100)}%;top:-1px;width:1px;height:11px;background:#7a3a24"></div></div>
      <span style="width:74px;font-size:11px;text-align:right">${v.toFixed(3)}<span style="color:var(--muted)"> / ${b.toFixed(3)}</span></span></div>`;
  }).join("");
}

// build the (optional) signed annotation-context correlation bars.
function correlationBars(corr) {
  const LABELS = {
    codon1: "codon 1", codon2: "codon 2", codon3: "codon 3",
    gc: "GC", gc_skew: "GC skew", strand: "strand",
  };
  const entries = Object.entries(corr).filter(([, v]) => isFinite(+v));
  if (!entries.length) return "";
  const bars = entries.map(([k, v]) => {
    const r = Math.max(-1, Math.min(1, +v));
    const w = Math.abs(r) * 50; // half-width; center is the zero line
    const side = r >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
    const col = r >= 0 ? "var(--blue)" : "#a4553b";
    return `<div style="display:flex;align-items:center;gap:6px;margin:1px 0">
      <span style="width:66px;color:var(--muted);font-size:11.5px">${esc(LABELS[k] || k)}</span>
      <div style="flex:1;background:#eee;height:9px;border-radius:2px;position:relative">
        <div style="position:absolute;left:50%;top:-1px;width:1px;height:11px;background:#999"></div>
        <div style="position:absolute;${side};background:${col};height:9px;border-radius:2px"></div></div>
      <span style="width:74px;font-size:11px;text-align:right">${r >= 0 ? "+" : ""}${r.toFixed(2)}</span></div>`;
  }).join("");
  return `<div style="margin-top:10px">
    <div style="font-size:11px;color:var(--muted);margin-bottom:2px">annotation-context correlations (Pearson r)</div>${bars}</div>`;
}

// build the detected-annotation table rows.
function annotationTable(d) {
  const det = d.detected || [];
  // fraction rate: 3 decimals when >= 0.001, else 1-sig-fig scientific.
  const fmtRate = (v) => (!isFinite(+v)) ? "–" : (+v >= 0.001 ? (+v).toFixed(3) : (+v).toExponential(1));
  if (det.length) {
    // Two enrichment metrics per annotation, shown side by side so they are
    // never conflated (they can differ — that is the point):
    //   span     = fraction of the fold's spans the feature covers ÷ matched-
    //              negative-span rate (#38/#40 precision_fold, the gate number),
    //              sub-line: cover-rate / matched-bg-rate.
    //   position = per-position in-fold firing rate ÷ background (the atlas
    //              Haldane circular-shift fold), sub-line: pos-rate / bg-rate.
    const head = `<tr><th>source</th><th>annotation</th>` +
      `<th class="num">span enrichment<div class="sub" style="font-weight:400">cover / bg-span rate</div></th>` +
      `<th class="num">lift<div class="sub" style="font-weight:400">PPV / prior</div></th>` +
      `<th class="num">recall</th><th class="num">q</th></tr>`;
    const rows = det.slice(0, 16).map((a) => {
      const { name, sub } = detName(a);
      const recall = Math.round((+a.recall) * 100);
      const ci = (isFinite(+a.recall_lo) && isFinite(+a.recall_hi))
        ? `<div class="sub">CI ${Math.round(+a.recall_lo * 100)}–${Math.round(+a.recall_hi * 100)}%</div>` : "";
      // span enrichment (gate) — embedded fields, with the sidecar as fallback.
      const sf = (a.span_fold !== undefined) ? a.span_fold
        : (Array.isArray(a.span) ? a.span[0] : undefined);
      const cr = (a.cover_rate !== undefined) ? a.cover_rate
        : (Array.isArray(a.span) ? a.span[1] : undefined);
      const mbg = (a.matched_bg_rate !== undefined) ? a.matched_bg_rate
        : (Array.isArray(a.span) ? a.span[2] : undefined);
      const spanCell = (sf !== undefined)
        ? `<span title="span-level: fraction-of-spans-covered ÷ matched-negative-span rate (the gate metric)">${fmtFold(sf)}</span>` +
          `<div class="sub">${fmtRate(cr)} / ${fmtRate(mbg)}</div>`
        : `<span class="sub" title="no span-level metric for this annotation type (e.g. a per-position track like codon or GC)">n/a</span>`;
      // position-level effect size is now LIFT = PPV / prior (unbounded, not
      // capped). PPV = P(in annotation | feature fires) in [0,1]; prior =
      // P(in annotation). Significance (q) still comes from the circular-shift null.
      // lift is served via the rates sidecar (rates.json -> e.rates = [pos_rate, bg_rate, ppv, prior, lift]),
      // falling back to embedded a.lift if present.
      const rt = Array.isArray(a.rates) ? a.rates : null;
      const liftV = (a.lift !== undefined) ? a.lift : (rt ? rt[4] : undefined);
      const ppvV = (a.ppv !== undefined) ? a.ppv : (rt ? rt[2] : undefined);
      const priorV = (a.prior !== undefined) ? a.prior : (rt ? rt[3] : undefined);
      const liftCell = (liftV !== undefined)
        ? `<span title="lift = PPV ÷ prior = P(in annotation | feature fires) ÷ P(in annotation)">${fmtFold(liftV)}</span>` +
          `<div class="sub">${fmtRate(ppvV)} / ${fmtRate(priorV)}</div>`
        : `<span class="sub" title="lift not available for this annotation type (no per-position firing rate, e.g. GO/keyword)">n/a</span>`;
      return `<tr><td><span class="pill ${esc(a.class)}">${esc(a.class)}</span></td>` +
        `<td>${esc(name)}<div class="sub">${esc(sub)}</div></td>` +
        `<td class="num">${spanCell}</td>` +
        `<td class="num">${liftCell}</td>` +
        `<td class="num">${recall}%${ci}</td>` +
        `<td class="num">${sci(a.fdr)}</td></tr>`;
    }).join("");
    return { head, rows: rows || emptyAnnRow(), usingDet: true };
  }
  // legacy hypergeometric fallback.
  const head = `<tr><th>class</th><th>annotation</th><th>fold</th><th>overlap/support</th><th>FDR</th></tr>`;
  const rows = (d.top_annotations || []).slice(0, 10).map((a) => {
    const nm = a.class === "pfam" ? detName({ class: "pfam", id: "pfam|" + a.pretty, pretty: a.pretty }).name
      : a.class === "go" ? goName(a.pretty) : String(a.pretty || "").replace(/_/g, " ");
    return `<tr><td><span class="pill ${esc(a.class)}">${esc(a.class)}</span></td>` +
      `<td>${esc(nm)}<div class="sub">${esc(a.pretty)}</div></td>` +
      `<td class="num">${fmtFold(a.fold)}</td><td class="num">${a.overlap}/${fmt(a.support)}</td>` +
      `<td class="num">${sci(a.fdr)}</td></tr>`;
  }).join("");
  return { head, rows: rows || emptyAnnRow(), usingDet: false };
}
const emptyAnnRow = () => `<tr><td colspan="6" style="color:var(--muted)">no significant annotations</td></tr>`;

function renderMain() {
  const d = state.CUR, m = $("#main"), exs = d.exemplars || [];
  const { head: annHead, rows: annRows, usingDet } = annotationTable(d);

  // sense/antisense firing fraction — the logos are gene-strand oriented but the
  // model reads the firing strand.
  const coding = exs.filter((e) => e.prot && e.prot.st);
  const senseN = coding.filter((e) => e.strand === e.prot.st).length;
  const sensePct = coding.length ? Math.round((100 * senseN) / coding.length) : null;
  const logoCaveat = coding.length
    ? `logos are <b>gene-strand</b> oriented · ${sensePct}% of coding exemplars fire <b>sense</b> (same strand as the gene)` +
      (sensePct < 50 ? ` — mostly <b>antisense</b>: the model's actual input is the reverse complement of the motif shown` : ``)
    : `logos are gene-strand oriented`;

  const ctxBars = contextBars(d.continuous || {}, d.continuous_bg || {});
  const corrBars = d.correlations ? correlationBars(d.correlations) : "";

  m.innerHTML = `
    <div class="summary">
      <div>
        <h2>Feature ${d.feature_id}${d.confound ? `<span class="confound-tag">confound: ${esc(d.confound)}</span>` : ""}</h2>
        <div class="stat">fires <b>${fmt(d.fire_count)}</b> · mean act <b>${(+d.mean_act).toFixed(3)}</b>
          · density <b>${(+d.density).toExponential(2)}</b> · <b>${exs.length}</b> exemplars</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:2px">${usingDet
          ? "detected annotations (genomic + CATH/TED folds): span enrichment = cover-rate ÷ matched-negative-span rate (the gate); lift = PPV ÷ prior = P(in annotation | fires) ÷ P(in annotation); recall (Wilson CI); FDR-q from the circular-shift null"
          : "enriched annotations (legacy hypergeometric)"}</div>
        <table class="ann"><thead>${annHead}</thead><tbody>${annRows}</tbody></table>
        <div style="margin-top:10px">
          <div style="font-size:11px;color:var(--muted);margin-bottom:2px">annotation context at firing sites
            <span style="color:#7a3a24">|</span> = genome background</div>${ctxBars}</div>
        ${corrBars}
      </div>
      <div class="logos">
        <div class="logo-box"><div class="lt">nucleotide logo (±${(d.nt_logo.freq.length - 1) / 2} nt, n=${d.nt_logo.n})</div><div id="ntlogo"></div></div>
        <div class="logo-box"><div class="lt">amino-acid logo (±${(d.aa_logo.freq.length - 1) / 2} codons, n=${d.aa_logo.n})</div><div id="aalogo"></div></div>
        <div class="lt" style="margin-top:2px;line-height:1.35">${logoCaveat}</div>
      </div>
    </div>
    <div class="exbar">
      <div class="pager"><button id="prev">◄ prev</button><span class="idx" id="exidx"></span><button id="next">next ►</button></div>
      <div class="zoom"><button id="zout">– zoom</button><button id="zin">+ zoom</button><button id="zreset">reset</button>
        <span class="coord" id="zspan"></span></div>
      <div class="coord" id="excoord"></div>
    </div>
    <div id="tracks"></div>
    <div class="legend">
      <span><i style="background:var(--ember)"></i>activation</span>
      <span><i style="background:#d9c3a8"></i>CDS +</span><span><i style="background:#b9c9d4"></i>CDS −</span>
      <span><i style="background:var(--blue)"></i>Pfam</span><span><i style="background:#7b5ea7"></i>CATH fold</span><span><i style="background:#9a8fc0"></i>TED fold</span><span><i style="background:var(--forest)"></i>RNA</span>
      <span><i style="background:#c98a2b"></i>regulatory</span><span><i style="background:#4f8f6a"></i>operon</span><span><i style="background:#b0563a"></i>replication</span><span><i style="background:#8a8a4f"></i>motif</span>
      <span><i style="background:#4E728A"></i>codon 1</span><span><i style="background:#C4650D"></i>codon 2</span><span><i style="background:#4a6b52"></i>codon 3</span>
      <span><i style="background:#7a3a24;height:2px"></i>GC%</span><span><i style="background:#b98"></i>low-cplx</span></div>
    <div class="hint">Zoom with the buttons or mouse-wheel, pan by dragging. Sequence + AA letters appear when zoomed in.
      Dashed line = max-activating position. The activation track is the feature's dense per-nucleotide
      activation across the window (a span-clustered feature reads as a filled span; where a dense profile
      isn't available it falls back to the harvest's retained firing sites, graded by strength).</div>
    <div id="protpanel"></div>
    <div id="afpanel"></div>`;

  drawLogo("#ntlogo", d.nt_logo, 2, ntColor);
  drawLogo("#aalogo", d.aa_logo, Math.log2(20), aaColor);

  $("#prev").onclick = () => { if (state.EXI > 0) { state.EXI--; setExemplar(); } };
  $("#next").onclick = () => { if (state.EXI < exs.length - 1) { state.EXI++; setExemplar(); } };
  $("#zin").onclick = () => zoomBy(1 / 1.6);
  $("#zout").onclick = () => zoomBy(1.6);
  $("#zreset").onclick = () => resetView();

  setExemplar();
}

function setExemplar() {
  const exs = state.CUR.exemplars, e = exs[state.EXI];
  $("#exidx").textContent = `${state.EXI + 1} / ${exs.length}`;
  $("#prev").disabled = state.EXI === 0;
  $("#next").disabled = state.EXI === exs.length - 1;
  $("#excoord").innerHTML =
    `${esc(e.acc)} · ${esc(e.contig)}:${fmt(e.start + 1)}–${fmt(e.end)} (${e.strand}) · ` +
    `max act <b style="color:var(--ember)">${(+e.center_act).toFixed(2)}</b>`;
  state.VIEW = [0, e.seq.length];
  drawTracks();
  wireTrackInteractions();
  renderProtPanel(e);
  renderAFPanel(e, nextAFToken()); // new token invalidates any in-flight load
}

function renderProtPanel(e) {
  const el = $("#protpanel");
  if (!e.prot) { el.innerHTML = ""; return; }
  const gos = e.prot.go || [];
  const goHtml = gos.length
    ? gos.map((id) => `<span class="pill go" title="${esc(id)}">${esc(goName(id))} <span style="opacity:.6">${esc(id)}</span></span>`).join(" ")
    : `<span style="color:var(--muted)">no GO terms mapped for this protein</span>`;
  el.innerHTML = `<div style="margin-top:12px;padding:10px 12px;border:1px solid var(--rule);border-radius:6px;background:#fff">
    <div style="font-size:12.5px;color:var(--muted);margin-bottom:4px">exemplar protein · <b style="color:var(--text)">${esc(e.prot.product || e.prot.id)}</b>
      <span style="color:var(--muted)">(${esc(e.prot.id)}, ${e.prot.st} strand)</span></div>
    <div style="display:flex;flex-wrap:wrap;gap:5px;font-size:11.5px">${goHtml}</div></div>`;
}

init();
