// tracks.js — the SVG genome-track renderer + zoom/pan/tooltip interactions.
//
// Renders, top to bottom: the dense per-nt activation track, gene/CDS (+/−),
// Pfam domains (IGV-expanded, one per row), CATH/TED folds, RNA, the #38 v2
// layers (regulatory/operon/replication/motif), codon position, GC%,
// low-complexity, and the nucleotide + amino-acid rows. A row only appears when
// it has ≥1 span in the current window. All positions are window-relative;
// absolute = exemplar.start + rel.

import { state, $, esc, clip, NT_COL, CODCOL, GRP_COL, AA_GRP, pfName, pfGO, goName, v2name } from "./data.js";

// ---- tooltip -------------------------------------------------------------
const tt = $("#tt");
function showtt(e, html) {
  tt.innerHTML = html;
  tt.style.opacity = 1;
  let x = e.clientX + 12, y = e.clientY + 12;
  if (x > innerWidth - 330) x = e.clientX - tt.offsetWidth - 12;
  tt.style.left = x + "px";
  tt.style.top = y + "px";
}
function hidett() { tt.style.opacity = 0; }

// ---- greedy lane packing for overlapping spans ---------------------------
function packLanes(spans) {
  const s = spans.map((x, i) => ({ ...x, i })).sort((a, b) => a.start - b.start);
  const lanes = [];
  for (const x of s) {
    let placed = false;
    for (const ln of lanes) {
      if (ln[ln.length - 1].end <= x.start) { ln.push(x); placed = true; break; }
    }
    if (!placed) lanes.push([x]);
  }
  return lanes;
}

// ---- zoom / pan ----------------------------------------------------------
const SPAN_MIN = 24; // never zoom tighter than this many nt

export function zoomBy(f) {
  const e = state.CUR.exemplars[state.EXI], L = e.seq.length;
  let [v0, v1] = state.VIEW;
  const c = (v0 + v1) / 2;
  const span = Math.min(L, Math.max(SPAN_MIN, (v1 - v0) * f));
  v0 = Math.max(0, Math.round(c - span / 2));
  v1 = Math.min(L, v0 + Math.round(span));
  v0 = Math.max(0, v1 - Math.round(span));
  state.VIEW = [v0, v1];
  drawTracks();
}
// cursor-anchored zoom (mouse wheel): keep the nt under the cursor fixed.
export function zoomAt(f, fracPos) {
  const e = state.CUR.exemplars[state.EXI], L = e.seq.length;
  let [v0, v1] = state.VIEW;
  const anchor = v0 + fracPos * (v1 - v0);
  const span = Math.min(L, Math.max(SPAN_MIN, (v1 - v0) * f));
  v0 = Math.round(anchor - fracPos * span);
  v1 = v0 + Math.round(span);
  if (v0 < 0) { v0 = 0; v1 = Math.round(span); }
  if (v1 > L) { v1 = L; v0 = Math.max(0, L - Math.round(span)); }
  state.VIEW = [v0, v1];
  drawTracks();
}
export function resetView() {
  const e = state.CUR.exemplars[state.EXI];
  state.VIEW = [0, e.seq.length];
  drawTracks();
}

// ---- main render ---------------------------------------------------------
export function drawTracks() {
  const e = state.CUR.exemplars[state.EXI];
  const [v0, v1] = state.VIEW;
  const span = v1 - v0, L = e.seq.length;
  const zspan = $("#zspan"); if (zspan) zspan.textContent = `${span} nt view`;

  const cont = $("#tracks");
  const W = Math.max(680, cont.clientWidth || 900);
  const PADL = 70, PADR = 12, iw = W - PADL - PADR, px = iw / span;
  const X = (p) => PADL + (p - v0) * px;
  const showSeq = px >= 7;

  const genesP = (e.genes || []).filter((g) => g.strand === "+");
  const genesM = (e.genes || []).filter((g) => g.strand === "-");
  // Pfam domains: IGV-expanded — one domain per row so overlapping domains are
  // all visible at once, each on its own labeled row.
  const domLanes = (e.domains || []).slice()
    .sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start))
    .map((d) => [d]);
  const rnaLanes = packLanes(e.rnas || []);
  const rfamLanes = packLanes(e.rfam || []);
  const fold = e.folds || {}, v2 = e.v2 || {};
  const cathLanes = packLanes(fold.cath || []), tedLanes = packLanes(fold.ted || []);
  const regLanes = packLanes(v2.regulatory || []), opLanes = packLanes(v2.operon || []);
  const repLanes = packLanes(v2.replication || []), motLanes = packLanes(v2.motif || []);

  // absolute opacity reference (peak of the top exemplar) so a strong-but-not-max
  // window still reads as active.
  const yMax = state.CUR.exemplars[0].center_act || 1;
  const cen = e.center;

  // ---- assemble the row layout -----------------------------------------
  let y = 6;
  const R = [];
  const row = (h, label, draw) => { R.push({ y, h, label, draw }); y += h + 7; };

  row(52, e._dense ? "activation" : "activation (sparse)", aRow);
  row(16, "gene / CDS +", (r) => geneRow(r, genesP, "#d9c3a8", "+"));
  row(16, "gene / CDS −", (r) => geneRow(r, genesM, "#b9c9d4", "−"));
  const domH = Math.max(1, domLanes.length) * 15;
  row(domH, "Pfam", (r) => laneRow(r, domLanes, "var(--blue)", "dom"));
  if (cathLanes.length) row(cathLanes.length * 15, "CATH fold", (r) => laneRow(r, cathLanes, "#7b5ea7", "fold"));
  if (tedLanes.length) row(tedLanes.length * 15, "TED fold", (r) => laneRow(r, tedLanes, "#9a8fc0", "fold"));
  const rnaH = Math.max(1, rnaLanes.length) * 15;
  row(rnaH, "RNA", (r) => laneRow(r, rnaLanes, "var(--forest)", "rna"));
  if (rfamLanes.length) row(rfamLanes.length * 15, "Rfam RNA", (r) => laneRow(r, rfamLanes, "#2f8f8f", "rfam"));
  if (regLanes.length) row(regLanes.length * 15, "regulatory", (r) => laneRow(r, regLanes, "#c98a2b", "v2"));
  if (opLanes.length) row(opLanes.length * 15, "operon", (r) => laneRow(r, opLanes, "#4f8f6a", "v2"));
  if (repLanes.length) row(repLanes.length * 15, "replication", (r) => laneRow(r, repLanes, "#b0563a", "v2"));
  if (motLanes.length) row(motLanes.length * 13, "motif", (r) => laneRow(r, motLanes, "#8a8a4f", "v2"));
  row(13, "codon 1/2/3", cRow);
  row(24, "GC%", gcRow);
  row(9, "low-cplx", lcRow);
  row(showSeq ? 16 : 7, "seq", seqRow);
  row(showSeq ? 15 : 7, "AA", aaRow);

  const H = y + 2;
  let s = `<svg width="100%" viewBox="0 0 ${W} ${H}" font-family="Inter" font-size="10">`;
  for (const r of R) s += `<text x="6" y="${r.y + Math.min(12, r.h / 2 + 4)}" fill="#555">${r.label}</text>`;
  if (cen >= v0 && cen < v1) {
    const cx = X(cen);
    s += `<line x1="${cx}" y1="2" x2="${cx}" y2="${H - 2}" stroke="var(--ember)" stroke-dasharray="3,3" stroke-width="1" opacity="0.55"/>`;
  }
  for (const r of R) s += r.draw(r);
  s += `</svg>`;
  cont.innerHTML = s;

  // ---- per-row draw functions (closures over view geometry) ------------
  function aRow(r) {
    const base = r.y + r.h;
    let o = `<rect x="${PADL}" y="${r.y}" width="${iw}" height="${r.h}" fill="#faf9f5" stroke="#eee"/>`;
    if (e._dense) {
      // dense per-nt re-harvested profile: a continuous filled path scaled to a
      // WINDOW-LOCAL max so a span-shaped feature fills its span.
      const D = e._dense, Ld = D.length;
      let wMax = 1e-6;
      for (let i = v0; i < Math.min(v1, Ld); i++) if (D[i] > wMax) wMax = D[i];
      const Hd = (v) => Math.min(1, v / wMax) * (r.h - 3);
      let path = `M ${X(v0 + 0.5).toFixed(1)} ${base}`;
      for (let i = v0; i < Math.min(v1, Ld); i++) path += ` L ${X(i + 0.5).toFixed(1)} ${(base - Hd(D[i])).toFixed(1)}`;
      path += ` L ${X(Math.min(v1, Ld) - 1 + 0.5).toFixed(1)} ${base} Z`;
      o += `<path d="${path}" fill="var(--ember)" fill-opacity="0.22" stroke="var(--ember)" stroke-width="0.8" stroke-opacity="0.85"/>`;
      if (cen >= v0 && cen < v1 && cen < Ld) {
        const x = X(cen + 0.5);
        o += `<circle cx="${x.toFixed(1)}" cy="${(base - Hd(D[cen])).toFixed(1)}" r="2.6" fill="var(--ember)"/>`;
      }
      // invisible wide hitmarks give per-nt hover tooltips.
      for (let i = v0; i < Math.min(v1, Ld); i++) {
        if (D[i] > 0.1) {
          const x = X(i + 0.5);
          o += `<line x1="${x.toFixed(1)}" y1="${base}" x2="${x.toFixed(1)}" y2="${(base - Hd(D[i])).toFixed(1)}" stroke="var(--ember)" stroke-width="${Math.max(0.5, px).toFixed(2)}" stroke-opacity="0" class="hz" data-tip="activation ${D[i].toFixed(2)} @ +${i} nt"/>`;
        }
      }
      return o;
    }
    // fallback: sparse retained-firing profile (filled envelope + graded stems).
    const pts = [];
    for (let k = 0; k < e.track_rel.length; k++) {
      const p = e.track_rel[k];
      if (p >= v0 && p < v1) pts.push([p, e.track_act[k]]);
    }
    pts.sort((a, b) => a[0] - b[0]);
    if (pts.length) {
      const wMax = Math.max(...pts.map((p) => p[1]));
      const Hh = (a) => Math.max(0.12, a / wMax) * (r.h - 3); // floor so moderate positions stay visible
      const OP = (a) => 0.4 + 0.6 * Math.min(1, a / yMax);    // colour by absolute activation
      let path = `M ${X(pts[0][0] + 0.5).toFixed(1)} ${base}`;
      for (const [p, a] of pts) path += ` L ${X(p + 0.5).toFixed(1)} ${(base - Hh(a)).toFixed(1)}`;
      path += ` L ${X(pts[pts.length - 1][0] + 0.5).toFixed(1)} ${base} Z`;
      o += `<path d="${path}" fill="var(--ember)" fill-opacity="0.16" stroke="none"/>`;
      for (const [p, a] of pts) {
        const x = X(p + 0.5), h = Hh(a), isc = p === cen, op = isc ? 1 : OP(a);
        o += `<line x1="${x.toFixed(1)}" y1="${base}" x2="${x.toFixed(1)}" y2="${(base - h).toFixed(1)}" stroke="var(--ember)" stroke-width="${isc ? 2.6 : 1.4}" stroke-opacity="${op}" class="hz" data-tip="activation ${a.toFixed(2)} @ +${p} nt"/>`;
        o += `<circle cx="${x.toFixed(1)}" cy="${(base - h).toFixed(1)}" r="${isc ? 2.8 : 1.6}" fill="var(--ember)" fill-opacity="${op}"/>`;
      }
    }
    return o;
  }

  function geneRow(r, list, col, sy) {
    let o = "";
    for (const g of list) {
      const x0 = X(Math.max(v0, g.start)), x1 = X(Math.min(v1, g.end));
      if (x1 <= x0) continue;
      o += `<rect x="${x0.toFixed(1)}" y="${r.y + 1}" width="${(x1 - x0).toFixed(1)}" height="${r.h - 2}" fill="${col}" class="hz" data-tip="${esc(g.label)} (${g.strand})"/>`;
      const ar = sy === "+"
        ? `M${x1 - 5} ${r.y + 1} L${x1} ${r.y + r.h / 2} L${x1 - 5} ${r.y + r.h - 1}`
        : `M${x0 + 5} ${r.y + 1} L${x0} ${r.y + r.h / 2} L${x0 + 5} ${r.y + r.h - 1}`;
      if (x1 - x0 > 10) o += `<path d="${ar}" fill="#8a6a3a" opacity="0.8"/>`;
      if (x1 - x0 > 60) o += `<text x="${((x0 + x1) / 2).toFixed(1)}" y="${r.y + r.h - 3}" text-anchor="middle" font-size="9" fill="#4a3a15">${esc(clip(g.label, Math.floor((x1 - x0) / 6)))}</text>`;
    }
    return o;
  }

  function laneRow(r, lanes, col, kind) {
    let o = "";
    lanes.forEach((ln, li) => {
      const ly = r.y + li * 15;
      for (const d of ln) {
        const x0 = X(Math.max(v0, d.start)), x1 = X(Math.min(v1, d.end));
        if (x1 <= x0) continue;
        let nm = d.label, tip = nm;
        if (kind === "dom") {
          nm = pfName(d.pfam_acc);
          const gos = pfGO(d.pfam_acc).map(goName);
          tip = `${nm} (${d.pfam_acc})` + (gos.length ? "<br>GO: " + esc(gos.slice(0, 4).join(", ")) : "");
        } else if (kind === "fold") {
          nm = d.name || d.code;
          tip = `${esc(nm)} (${esc(d.code)})`;
        } else if (kind === "rfam") {
          nm = d.label;
          tip = `${esc(d.label)} (${esc(d.rf)})` + (d.strand && d.strand !== "." ? ` [${d.strand}]` : "");
        } else if (kind === "v2") {
          nm = v2name(d.t);
          tip = esc(nm) + (d.l && d.l !== d.t ? `<br>${esc(d.l)}` : "") + (d.strand && d.strand !== "." ? ` (${d.strand})` : "");
        } else {
          tip = `${esc(d.label)} (${d.rtype || "RNA"})`;
        }
        o += `<rect x="${x0.toFixed(1)}" y="${ly + 1}" width="${Math.max(2, x1 - x0).toFixed(1)}" height="12" rx="2" fill="${col}" class="hz" data-tip="${tip.replace(/"/g, "&quot;")}"/>`;
        if (x1 - x0 > 50) {
          o += `<text x="${((x0 + x1) / 2).toFixed(1)}" y="${ly + 10}" text-anchor="middle" font-size="8.5" fill="#fff">${esc(clip(nm, Math.floor((x1 - x0) / 6)))}</text>`;
        } else if (kind === "dom") {
          // narrow domain: label to the right so every Pfam row is named.
          const avail = (W - PADR) - x1 - 4;
          if (avail > 18) o += `<text x="${(x1 + 3).toFixed(1)}" y="${ly + 10}" font-size="8.5" fill="#31506a">${esc(clip(nm, Math.floor(avail / 6)))}</text>`;
        }
      }
    });
    return o;
  }

  function cRow(r) {
    let o = "";
    const lab = px >= 9;
    for (let i = Math.floor(v0); i < v1; i++) {
      const cp = e.codon.charCodeAt(i) - 48;
      if (CODCOL[cp]) {
        o += `<rect x="${X(i).toFixed(1)}" y="${r.y}" width="${Math.max(0.4, px).toFixed(2)}" height="${r.h}" fill="${CODCOL[cp]}"/>`;
        if (lab) o += `<text x="${X(i + 0.5).toFixed(1)}" y="${r.y + r.h - 2.5}" text-anchor="middle" font-size="8.5" fill="#fff">${cp}</text>`;
      }
    }
    return o;
  }

  function aaRow(r) {
    let o = "";
    const showAA = 3 * px >= 9;
    if (!showAA) {
      // compact: colour block per codon by AA property group.
      for (let i = v0; i < v1; i++) {
        const ch = e.aa ? e.aa[i] : ".";
        if (ch !== ".") {
          const col = ch === "*" ? "#999" : (GRP_COL[AA_GRP[ch]] || "#888");
          o += `<rect x="${X(i - 1).toFixed(1)}" y="${r.y}" width="${Math.max(0.6, 3 * px).toFixed(2)}" height="${r.h}" fill="${col}" opacity="0.6"/>`;
        }
      }
      return o;
    }
    for (let i = v0; i < v1; i++) {
      const ch = e.aa ? e.aa[i] : ".";
      if (ch === ".") continue;
      const col = ch === "*" ? "#999" : (GRP_COL[AA_GRP[ch]] || "#555");
      o += `<text x="${X(i + 0.5).toFixed(1)}" y="${r.y + 12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="${Math.min(13, 3 * px * 0.7).toFixed(1)}" fill="${col}" class="hz" data-tip="residue ${ch}">${ch}</text>`;
    }
    return o;
  }

  function gcRow(r) {
    let o = `<rect x="${PADL}" y="${r.y}" width="${iw}" height="${r.h}" fill="#faf9f5" stroke="#eee"/>`;
    let path = "";
    for (let i = v0; i < v1; i++) {
      const g = e._gc[i] / 100, yy = r.y + r.h - g * r.h;
      path += (i === v0 ? "M" : "L") + X(i + 0.5).toFixed(1) + " " + yy.toFixed(1) + " ";
    }
    o += `<path d="${path}" fill="none" stroke="#7a3a24" stroke-width="1"/>`;
    o += `<line x1="${PADL}" y1="${r.y + r.h / 2}" x2="${W - PADR}" y2="${r.y + r.h / 2}" stroke="#ddd" stroke-dasharray="2,2"/>`;
    return o;
  }

  function lcRow(r) {
    let o = "";
    for (let i = v0; i < v1; i++)
      if (e.lc.charCodeAt(i) === 49)
        o += `<rect x="${X(i).toFixed(1)}" y="${r.y}" width="${Math.max(0.4, px).toFixed(2)}" height="${r.h}" fill="#bb9988"/>`;
    return o;
  }

  function seqRow(r) {
    let o = "";
    if (showSeq) {
      for (let i = v0; i < v1; i++) {
        const ch = e.seq[i];
        o += `<text x="${X(i + 0.5).toFixed(1)}" y="${r.y + 12}" text-anchor="middle" font-family="IBM Plex Mono" font-size="${Math.min(13, px).toFixed(1)}" fill="${NT_COL[ch] || "#999"}">${ch}</text>`;
      }
    } else {
      for (let i = v0; i < v1; i++) {
        const ch = e.seq[i];
        o += `<rect x="${X(i).toFixed(1)}" y="${r.y}" width="${Math.max(0.4, px).toFixed(2)}" height="${r.h}" fill="${NT_COL[ch] || "#ccc"}" opacity="0.7"/>`;
      }
    }
    return o;
  }
}

// ---- interactions --------------------------------------------------------
// Global drag state. Window-level listeners are attached ONCE at boot; the
// #tracks container is recreated per feature, so its wheel/mousedown handlers
// are (re)assigned (not addEventListener'd) on each render — see wireTracks.
let dragging = false, lastX = 0;

// One-time window-level drag listeners. Safe to call before #tracks exists:
// the handlers query the container lazily and no-op unless a drag is active.
export function initTracks() {
  window.addEventListener("mousemove", (ev) => {
    if (!dragging || !state.CUR || !state.VIEW) return;
    const cont = $("#tracks");
    if (!cont) return;
    const e = state.CUR.exemplars[state.EXI], L = e.seq.length;
    let [v0, v1] = state.VIEW;
    const span = v1 - v0, rect = cont.getBoundingClientRect(), px = (rect.width - 82) / span;
    const dnt = (ev.clientX - lastX) / px;
    lastX = ev.clientX;
    let nv0 = v0 - dnt;
    if (nv0 < 0) nv0 = 0;
    if (nv0 + span > L) nv0 = L - span;
    state.VIEW = [Math.round(nv0), Math.round(nv0) + span];
    drawTracks();
  });
  window.addEventListener("mouseup", () => {
    if (dragging) { dragging = false; const c = $("#tracks"); if (c) c.classList.remove("drag"); }
  });
}

// Re-wire the container (wheel + drag start) and per-span hover tooltips after
// each SVG rebuild. Container handlers use assignment so they never stack.
export function wireTrackInteractions() {
  const cont = $("#tracks");
  if (!cont) return;
  cont.onwheel = (ev) => {
    if (!state.CUR || !state.VIEW) return;
    ev.preventDefault();
    const rect = cont.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left - 70) / (rect.width - 82)));
    zoomAt(ev.deltaY < 0 ? 1 / 1.2 : 1.2, frac);
  };
  cont.onmousedown = (ev) => {
    if (!state.CUR || !state.VIEW) return;
    dragging = true; lastX = ev.clientX; cont.classList.add("drag"); hidett();
  };
  cont.querySelectorAll(".hz").forEach((el) => {
    el.addEventListener("mousemove", (ev) => { ev.stopPropagation(); showtt(ev, el.dataset.tip); });
    el.addEventListener("mouseleave", hidett);
  });
}
