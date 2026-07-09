"""Build the #53 results page: the TED/CATH fold-granularity fix, with links to the
corrected interactive viewer and the aligned Clean-5 gallery.

Renders two Plotly figures (recall fragment-vs-superfamily; atlas-wide impact),
writes their plot-ready data bundles under ``<figroot>/<name>/data.json``, and
writes the assembled ``index.html`` into ``<app>``.
"""
from __future__ import annotations

import argparse
import json
import os

EDITORIAL_8 = ['#C4650D', '#4E728A', '#2E6E4E', '#988453', '#B9605B', '#7495AB', '#84713A', '#31362E']
PLOTLY_CONFIG = {"responsive": True, "displayModeBar": "hover", "displaylogo": False,
                 "toImageButtonOptions": {"format": "png", "filename": "figure", "scale": 2}}


def apply_theme(fig, *, height: int = 400):
    fig.update_layout(
        height=height, margin=dict(t=40, r=24, b=24, l=24, autoexpand=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1D272A', family="'Suisse Intl',-apple-system,Arial,sans-serif", size=13),
        title=None, colorway=EDITORIAL_8,
        legend=dict(orientation='h', xref='container', x=0, xanchor='left',
                    yref='container', y=0, yanchor='bottom', bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12), title=dict(side='top')),
        hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='#B4B4B4',
                        font=dict(family='ui-monospace,Menlo,monospace', size=12, color='#1D272A')),
        modebar=dict(orientation='h', bgcolor='rgba(0,0,0,0)', color='#1D272A', activecolor='#C4650D',
                     remove=['lasso2d', 'select2d', 'autoScale2d']),
        uniformtext=dict(minsize=10, mode='hide'))
    fig.update_traces(textposition='none', selector=dict(type='bar'))
    fig.update_xaxes(gridcolor='#B4B4B4', zerolinecolor='#B4B4B4', automargin=True, ticks='outside',
                     tickfont=dict(size=12), title_font=dict(size=13), autotickangles=[0, 30])
    fig.update_yaxes(gridcolor='#B4B4B4', zerolinecolor='#B4B4B4', automargin=True, ticks='outside',
                     tickfont=dict(size=12), title_font=dict(size=13))
    for t in fig.data:
        if hasattr(t, 'cliponaxis'):
            t.cliponaxis = True
    return fig


def build_html(figroot: str) -> str:
    """Render figures, write their data bundles under ``figroot``, return the page HTML."""
    import plotly.graph_objects as go
    from plotly.io import to_html

    # ---- Figure 1: fragment vs superfamily recall (5 gallery features) ----
    feats = ["10255", "27629", "2044", "9968", "28943"]
    fold_names = ["Trigger factor<br>C-terminal", "Cytochrome<br>c-like", "cPLA2<br>catalytic",
                  "Met-tRNA syn.<br>domain 2", "Integrase<br>catalytic core"]
    frag_recall = [0.85, 0.76, 0.52, 0.50, 0.36]   # #40 recall on the mis-joined 3-level fragment
    sf_recall = [0.908, 0.752, 0.656, 0.734, 0.578]  # corrected dominant-superfamily recall (reproduces #52)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="Mis-joined 3-level fragment", x=fold_names, y=frag_recall,
                          marker_color='#4E728A',
                          customdata=feats, hovertemplate="feature %{customdata}<br>fragment recall %{y:.2f}<extra></extra>"))
    fig1.add_trace(go.Bar(name="Corrected superfamily", x=fold_names, y=sf_recall, marker_color='#C4650D',
                          customdata=feats, hovertemplate="feature %{customdata}<br>superfamily recall %{y:.2f}<extra></extra>"))
    fig1.add_hline(y=0.5, line=dict(color='#31362E', width=1, dash='dot'),
                   annotation_text="Tier-A gate (0.5)", annotation_position="top left",
                   annotation_font=dict(size=11, color='#31362E'))
    fig1.update_layout(barmode='group')
    fig1.update_yaxes(title="Recall on fold members", range=[0, 1.0])
    fig1.update_xaxes(title="Gallery feature (fold)")
    apply_theme(fig1, height=420)
    fig1_html = to_html(fig1, full_html=False, include_plotlyjs='cdn', config=PLOTLY_CONFIG)

    # ---- Figure 2: atlas-wide impact per dictionary ----
    dicts = ["OG2 · k64", "BCR · k64", "BCR · k16"]
    rows_removed = [1137, 731, 294]
    feats_changed = [763, 526, 198]
    top_corrected = [121, 78, 36]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Partial rows removed", x=dicts, y=rows_removed, marker_color='#C4650D'))
    fig2.add_trace(go.Bar(name="Features corrected", x=dicts, y=feats_changed, marker_color='#4E728A'))
    fig2.add_trace(go.Bar(name="Top-fold relabeled", x=dicts, y=top_corrected, marker_color='#2E6E4E'))
    fig2.update_layout(barmode='group')
    fig2.update_yaxes(title="Count")
    fig2.update_xaxes(title="Dictionary (SAE)")
    apply_theme(fig2, height=400)
    fig2_html = to_html(fig2, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    # save figure bundles (plot-ready data)
    for name, data in [("recall_fragment_vs_superfamily",
                        {"feats": feats, "fold_names": fold_names, "frag_recall": frag_recall, "sf_recall": sf_recall}),
                       ("atlas_impact",
                        {"dicts": dicts, "rows_removed": rows_removed, "feats_changed": feats_changed, "top_corrected": top_corrected})]:
        d = os.path.join(figroot, name)
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/data.json", "w") as fh:
            json.dump(data, fh, indent=1)

    SRC_MAIN = "experiments/issue-53/src/rebuild_atlas.py;experiments/issue-53/src/canon.py;experiments/issue-53/src/align_render.py"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>A raw-string TED fold join understated the viewer's fold-detector recall; the corrected superfamily unit puts all five showcase detectors in Tier-A</title>
</head><body>
<header class="report-header">
  <h1>A raw-string TED fold join understated the viewer's fold-detector recall; at the corrected superfamily unit all five showcase detectors are Tier-A</h1>
</header>

<section class="section" data-sources="{SRC_MAIN}">
  <h2>Question</h2>
  <p>The canonical EDEN feature viewer joins each SAE feature to its structural fold on TED's raw
  label strings. TED labels each domain at whatever CATH-style level its classifier is confident
  about, so the same fold appears under several keys, the full 4-level homologous superfamily
  (<code>1.10.760.10</code>), a 3-level topology fragment (<code>1.10.760</code>, a tiny
  partial-resolution subset), and homogeneous multi-domain comma strings
  (<code>1.10.760,1.10.760</code>). We asked:</p>
  <ol type="a">
    <li><strong>Is the viewer's fold layer on the buggy raw-string join?</strong> Confirm the atlas keys folds on the mixed-granularity strings.</li>
    <li><strong>Does the superfamily-unit fix reproduce the diagnosed numbers?</strong> The five showcase features must land at the recalls found by the earlier diagnosis (0.91, 0.75, 0.66, 0.73, 0.58).</li>
    <li><strong>What is the atlas-wide impact, without damage?</strong> Collapse the fragments across all three dictionaries without merging genuinely distinct superfamilies or losing annotations.</li>
  </ol>
</section>

<section class="section" data-sources="{SRC_MAIN}">
  <h2>Results</h2>
  <div class="key-finding"><ul>
    <li><strong>The bug is a raw-string TED join.</strong> Every fold split into a full-superfamily row plus tiny partial-resolution fragments; the atlas scored features against a fragment, understating recall and inflating a fragile specificity ratio. CATH (CATH-S95 phmmer, already superfamily-level, no commas) was unaffected.</li>
    <li><strong>The fix reproduces the diagnosis exactly.</strong> Collapsing each feature's fragments into its dominant 4-level superfamily gives the five showcase recalls 0.908, 0.752, 0.656, 0.734, 0.578, all at or above the 0.5 Tier-A gate; feature 28943 rises from 0.36 on the fragment to 0.58.</li>
    <li><strong>Atlas-wide, 2,162 fragment rows removed, zero damage.</strong> Across the three dictionaries 1,487 features are corrected and 2,162 partial-resolution rows collapsed; a safety scan confirms no removed fragment had absorbed another annotation, and distinct superfamilies sharing a topology stay separate.</li>
    <li><strong>Divergent members now visibly superimpose.</strong> The gallery's member strips are structurally aligned (TM-align correspondence, then a Kabsch fit on the feature-active residues) into a shared fixed camera, so one fold and one activation pattern read across members down to ~20% sequence identity.</li>
  </ul></div>

  <div class="stat-grid">
    <div class="stat-card"><div class="stat-label">Showcase detectors, Tier-A</div><div class="stat-value">5 / 5</div></div>
    <div class="stat-card"><div class="stat-label">28943 recall: fragment → superfamily</div><div class="stat-value">0.36 → 0.58</div></div>
    <div class="stat-card"><div class="stat-label">Fragment rows removed (3 dicts)</div><div class="stat-value">2,162</div></div>
    <div class="stat-card"><div class="stat-label">Features corrected</div><div class="stat-value">1,487</div></div>
    <div class="stat-card"><div class="stat-label">Absorbed-annotation losses</div><div class="stat-value">0</div></div>
    <div class="stat-card"><div class="stat-label">Re-inference / store reads</div><div class="stat-value">none</div></div>
  </div>

  <section class="figure" data-sources="experiments/issue-53/figures/recall_fragment_vs_superfamily/data.json;experiments/issue-53/src/rebuild_atlas.py">
    <h3 class="figure-title">Figure 1: Recall on the mis-joined fragment vs the corrected superfamily</h3>
    <div class="plot">{fig1_html}</div>
    <div class="figure-explanation">
      <p class="figure-description">Per showcase feature: TED span-recall computed on the 3-level topology fragment the atlas mis-joined on (slate) and on the corrected dominant 4-level superfamily (ember), with the 0.5 Tier-A gate marked. Recalls are #40's authoritative span-recall; the superfamily values are the numbers deployed to the viewer.</p>
      <p class="figure-findings">All five clear the gate at the corrected unit. The correction is not uniformly an increase: (a) 28943 rises sharply, 0.36 → 0.58, moving it from a spurious Tier-B flag to Tier-A; (b) 9968 and 2044 rise (0.50 → 0.73, 0.52 → 0.66) as the feature's real superfamily members re-enter the denominator; (c) 27629 barely moves (0.76 → 0.75) because its tiny fragment happened to have high recall, but its fragile specificity ratio (the 1.05× artifact) is what the fix repairs, not its recall. The fragment recall was never wrong arithmetically; it was computed against the wrong, tiny span set.</p>
    </div>
  </section>

  <section class="figure" data-sources="experiments/issue-53/figures/atlas_impact/data.json;experiments/issue-53/src/rebuild_atlas.py">
    <h3 class="figure-title">Figure 2: Atlas-wide impact by dictionary</h3>
    <div class="plot">{fig2_html}</div>
    <div class="figure-explanation">
      <p class="figure-description">Per dictionary (three SAEs): partial-resolution TED rows removed (ember), features whose fold layer changed (slate), and features whose displayed top annotation was relabeled off an inflated fragment onto the real superfamily or protein family (forest).</p>
      <p class="figure-findings">The fix touches 1,487 of the ~9,000 fold-bearing features across the three dictionaries and removes 2,162 fragment rows (OG2 1,137; BCR-k64 731; BCR-k16 294). The 235 top-annotation relabelings are the visible payoff: an inflated fragment (often a spurious 999× enrichment from a tiny out-of-fold baseline) was the atlas's displayed top fold and is replaced by the fold's real superfamily or its Pfam family. A pre-deploy safety scan found zero cases where a removed fragment had absorbed another annotation through the viewer's near-duplicate dedup, so no CATH or Pfam annotation is lost.</p>
    </div>
  </section>

  <section class="figure" data-sources="experiments/issue-53/src/align_render.py;experiments/issue-53/gallery_build/src/build_gallery.py">
    <h3 class="figure-title">Figure 3: Structurally aligned divergent-member strip (feature 27629, cytochrome c-like)</h3>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">
      <div style="outline:2px solid #C4650D;outline-offset:2px"><img src="gallery/struct_27629.png" style="width:100%" alt="home reference"><p class="caption">home reference<br>Q749D0</p></div>
      <div><img src="gallery/struct_27629_m0.png" style="width:100%" alt="aligned member"><p class="caption">D5SWR4</p></div>
      <div><img src="gallery/struct_27629_m1.png" style="width:100%" alt="aligned member"><p class="caption">Q5ZRI3</p></div>
      <div><img src="gallery/struct_27629_m2.png" style="width:100%" alt="aligned member"><p class="caption">P37197</p></div>
      <div><img src="gallery/struct_27629_m3.png" style="width:100%" alt="aligned member"><p class="caption">Q82VZ6</p></div>
    </div>
    <div class="figure-explanation">
      <p class="figure-description">The home reference (outlined) and four sequence-divergent members of the cytochrome-c-like fold, each AlphaFold domain painted by feature 27629's activation (shared viridis scale). Every member is superimposed onto the reference by full-domain TM-align (residue correspondence) then a Kabsch fit on the feature-active residues, and drawn in one fixed camera centered on the active motif.</p>
      <p class="figure-findings">Once aligned, the same helical cytochrome fold sits in the same orientation across all five panels with the activation on the same region, at 19–48% pairwise identity. Active-motif Cα RMSD to the reference is 1.9 Å for the closest member (P37197, TM 0.82) and 5.8–6.9 Å for the most divergent (TM 0.29–0.32), an honest superposition given the sequence divergence; no member needed the whole-domain fallback. This is the structure-beyond-sequence claim made comparable member by member. The full aligned gallery covers all five showcase features.</p>
    </div>
  </section>

  <div class="callout"><strong>Interactive artifacts.</strong>
  <a href="viewer/index.html">Open the corrected feature viewer</a> (all three dictionaries; each feature's fold layer now at the superfamily unit) ·
  <a href="gallery/index.html">Open the Clean-5 showcase gallery</a> (aligned member strips, corrected recall, specificity framing).</div>
</section>

<section class="section">
  <h2>Method</h2>
  <ul>
    <li><strong>Canonical unit.</strong> Each raw TED code is mapped to its 3-level topology; a partial-resolution code (3-level, or homogeneous multi-domain comma) is absorbed into the dominant 4-level homologous superfamily under that topology (the code with the most covered spans). Distinct 4-level superfamilies that share a topology stay separate.</li>
    <li><strong>Metrics source.</strong> Recall, span counts, precision-fold, and q come from experiment #40's reduced TED span-enrichment tables (per dictionary); display enrichment from #42's struct_folds. No model re-inference and no read of the 740 GB activation store, the fix is a re-key plus recompute over already-computed per-code metrics.</li>
    <li><strong>CATH layer.</strong> The "CATH" track is CATH-S95 phmmer (not Gene3D HMMs) and is already emitted at the 4-level superfamily with no comma codes, so it needed no change.</li>
    <li><strong>Structure alignment.</strong> tmtools TM-align (full domain) for member↔reference residue correspondence, then a masked Kabsch fit on the feature-active residues (activation &gt; 0); whole-domain fallback when fewer than 6 active residue pairs (none triggered). Coordinates and per-residue activation reused from the earlier diagnosis's evidence bundles.</li>
    <li><strong>Two enrichment metrics per annotation, embedded in the data.</strong> Each detected entry carries both, clearly labeled, as fields in the feature JSON: the span-level gate number (span_fold = cover-rate ÷ matched-negative-span rate, from #38 for pfam/genomic and #40 for ted/cath, with cover_rate and matched_bg_rate) and the position-level <strong>lift</strong> = PPV ÷ prior, where PPV = P(in annotation | feature fires) = firings-in-annotation ÷ total firings (bounded [0,1]) and prior = P(in annotation) = annotation positions ÷ total positions (with ppv and prior fields). Lift replaces the earlier position "fold": because PPV is bounded, a value that used to hit the 999× display cap is now a legible precision. Significance (FDR-q) still comes from the circular-shift null. The two metrics genuinely differ (2044's phospholipase A2 fold reads 1041× span vs 46× lift), so they are never conflated; annotation types with no per-position rate (GO/keyword) show "n/a".</li>
    <li><strong>Verification.</strong> The five showcase recalls reproduced on the live canonical files; residual-fragment scan = 0 across all three dictionaries; served viewer bytes checked to load (not blank).</li>
    <li><strong>Scope.</strong> Structural (TED/CATH) fold-unit fix only. The base panel, annotation, and both-strand layers are unchanged; the per-exemplar genome-track domain spans remain the raw TED calls (faithful per-position annotations, not the feature's fold association).</li>
  </ul>
</section>

<section class="section">
  <h2>Discussion</h2>
  <ul>
    <li><strong>Question answered.</strong> The viewer's fold layer was on the buggy raw-string join; canonicalizing to the dominant superfamily reproduces the diagnosed recalls exactly and puts all five showcase detectors in Tier-A.</li>
    <li><strong>What the fix repairs.</strong> Recall (fragments understated the members detected) and a fragile specificity number (tiny fragments had inflated enrichment from a near-empty out-of-fold baseline). It does not move the structure-beyond-sequence headline, which was already computed on the correctly aggregated fold.</li>
    <li><strong>Why it is safe.</strong> Restricting absorption to true partial-resolution codes keeps distinct homologous superfamilies that merely share a topology number apart (e.g. under the Rossmann topology 3.40.50); a safety scan confirmed no dropped fragment carried an absorbed annotation.</li>
    <li><strong>Limitation.</strong> For the most sequence-divergent members (TM 0.29–0.32), the active-motif superposition is ~6 Å; the fold is genuinely divergent there, so the overlay is as tight as the structures allow, not a tight crystallographic match.</li>
  </ul>
</section>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", required=True, help="destination dir; index.html is written here")
    ap.add_argument("--figroot", required=True,
                    help="dir for the plot-ready figure data bundles (<figroot>/<name>/data.json)")
    a = ap.parse_args(argv)
    html = build_html(a.figroot)
    os.makedirs(a.app, exist_ok=True)
    out = os.path.join(a.app, "index.html")
    with open(out, "w") as fh:
        fh.write(html)
    print("wrote", out, len(html), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
