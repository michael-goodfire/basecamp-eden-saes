// logo.js — information-content sequence logos (nucleotide + amino acid).
//
// Per-column information content = maxbits − Shannon entropy; each glyph's
// height is proportional to p·IC and glyphs are stacked largest-on-bottom.
// nt: maxbits = 2 (ACGT). aa: maxbits = log2(20).
//
// The server delivers `freq` ALREADY oriented to the gene reading strand, so we
// consume it verbatim — no client-side re-orientation.

import { $, NT_COL, GRP_COL, AA_GRP } from "./data.js";

// colorFn: alphabet-letter -> CSS color.
export function drawLogo(sel, logo, maxbits, colorFn) {
  const off = logo.offsets, freq = logo.freq, alpha = logo.alphabet;
  const n = off.length;
  // larger cells so per-position letters are easily readable (was 50px tall, 12-20px cols)
  const cw = Math.max(15, Math.min(30, Math.floor(360 / n)));
  const H = 110, pad = 17, L0 = 28, W = L0 + n * cw + 6, CAP = 18, GLYPH = 13;

  let svg = `<svg width="100%" viewBox="0 0 ${W} ${H + pad}">`;
  svg += `<line x1="${L0 - 2}" y1="${H}" x2="${W}" y2="${H}" stroke="#ccc"/>`;
  svg += `<text x="0" y="12" font-size="9" fill="#999" font-family="Inter">${maxbits.toFixed(1)}b</text>`;
  svg += `<text x="0" y="${H}" font-size="9" fill="#999" font-family="Inter">0</text>`;

  for (let i = 0; i < n; i++) {
    const col = freq[i];
    // Shannon entropy of the column, then information content (bits).
    let ent = 0;
    for (const p of col) if (p > 0) ent -= p * Math.log2(p);
    const ic = Math.max(0, maxbits - ent);
    // stack: draw smallest glyph on top, largest on the baseline.
    const stack = col.map((p, j) => [p, j]).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]);
    let yb = H;
    for (let s = stack.length - 1; s >= 0; s--) {
      const [p, j] = stack[s];
      const h = (p * ic / maxbits) * H;
      if (h < 0.8) continue;
      const ch = alpha[j], cx = L0 + i * cw + cw / 2, sy = h / GLYPH;
      svg += `<g transform="translate(${cx.toFixed(1)} ${yb.toFixed(1)}) scale(${(cw / GLYPH * 0.85).toFixed(3)} ${sy.toFixed(3)})">` +
        `<text x="0" y="0" font-size="${CAP}" text-anchor="middle" font-family="IBM Plex Mono,monospace" ` +
        `font-weight="500" fill="${colorFn(ch)}">${ch}</text></g>`;
      yb -= h;
    }
    if (i % 3 === 0 || n <= 13)
      svg += `<text x="${L0 + i * cw + cw / 2}" y="${H + pad - 1}" font-size="8.5" text-anchor="middle" fill="#999" font-family="Inter">${off[i]}</text>`;
  }
  svg += `</svg>`;
  $(sel).innerHTML = svg;
}

// Convenience colour functions for the two logo alphabets.
export const ntColor = (ch) => NT_COL[ch] || "#999";
export const aaColor = (ch) => GRP_COL[AA_GRP[ch]] || "#555";
