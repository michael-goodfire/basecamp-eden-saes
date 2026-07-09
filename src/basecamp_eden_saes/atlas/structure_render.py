"""Structurally align a feature's divergent-member domains and render them in one
fixed camera, so the shared fold + the activation-painted motif line up by eye.

Per feature (from #52 evidence JSON, which carries Cα coords + per-residue
activation for the home representative and 4 sequence-divergent members):
  1. Full-domain TM-align (tmtools) of each member onto the representative to get
     the residue correspondence (and a whole-domain transform fallback).
  2. Masked Kabsch fit on ONLY the feature-active residues (activation>0), using
     the corresponding member/representative residue pairs from step 1, to tightly
     overlay the active motif. Fall back to the whole-domain TM-align transform
     when too few active residue pairs to fit stably (<MIN_ACTIVE_PAIRS).
  3. Apply the transform to the member's Cα, render all panels in one fixed camera
     centered on the representative's active-motif centroid, Cα worm coloured by
     activation on a shared viridis scale.

Outputs struct_<feat>.png (reference / representative) and struct_<feat>_m0..3.png
(aligned members) into the output dir, plus an align_report.json with the
alignment diagnostics used for the spot-check.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402,F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
from tmtools import tm_align  # noqa: E402

# Canonical on-cluster default (overridable via CLI).
DEFAULT_EVID = (
    "/mnt/data/shared/silico/experiments/exp_01kx1vb40mfhevvsy3k7sp85ck/"
    "worktree/experiments/issue-52/results/evidence"
)
CLEAN5 = [10255, 27629, 2044, 9968, 28943]
MIN_ACTIVE_PAIRS = 6
ELEV, AZIM = 18.0, -60.0


def load_struct(rec):
    ca = rec["ca"]  # [resnum, x, y, z, plddt, aa]
    coords = np.array([[r[1], r[2], r[3]] for r in ca], dtype=np.float64)
    seq = "".join((r[5] if isinstance(r[5], str) and len(r[5]) == 1 else "X") for r in ca)
    act_map = rec.get("activation", {})
    act = np.array([float(act_map.get(str(r[0]), 0.0)) for r in ca], dtype=np.float64)
    return coords, seq, act


def kabsch(P, Q):
    """Rotation R + translation t mapping P onto Q (x' = R@x + t), minimizing RMSD."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def correspondence(res):
    """Aligned (member_idx, rep_idx) pairs from the gapped TM-align alignment.
    seqxA = chain1 (member), seqyA = chain2 (representative)."""
    pairs = []
    mi = ri = 0
    for a, b in zip(res.seqxA, res.seqyA):
        if a != "-" and b != "-":
            pairs.append((mi, ri))
        if a != "-":
            mi += 1
        if b != "-":
            ri += 1
    return pairs


def align_member(m_coords, m_seq, m_act, r_coords, r_seq, r_act):
    """Return (transformed_member_coords, diag dict)."""
    res = tm_align(m_coords, r_coords, m_seq, r_seq)
    u, t_tm = np.array(res.u), np.array(res.t)
    xp_tm = (u @ m_coords.T).T + t_tm  # whole-domain superposition
    pairs = correspondence(res)
    # active-residue masked Kabsch: member residue active AND has a rep partner
    act_pairs = [(mi, ri) for (mi, ri) in pairs if m_act[mi] > 0]
    method = "masked_kabsch_active"
    if len(act_pairs) >= MIN_ACTIVE_PAIRS:
        P = np.array([m_coords[mi] for mi, _ in act_pairs])
        Q = np.array([r_coords[ri] for _, ri in act_pairs])
        R, t = kabsch(P, Q)
        xp = (R @ m_coords.T).T + t
        # active-region RMSD after masked fit (transform includes translation)
        Pt = (R @ P.T).T + t
        act_rmsd = float(np.sqrt(((Pt - Q) ** 2).sum(1).mean()))
    else:
        method = "fallback_whole_domain"
        xp = xp_tm
        if pairs:
            Pt = np.array([xp_tm[mi] for mi, _ in pairs])
            Q = np.array([r_coords[ri] for _, ri in pairs])
            act_rmsd = float(np.sqrt(((Pt - Q) ** 2).sum(1).mean()))
        else:
            act_rmsd = float("nan")
    # whole-domain RMSD (all aligned pairs) after the chosen transform
    if pairs:
        allP = np.array([xp[mi] for mi, _ in pairs])
        allQ = np.array([r_coords[ri] for _, ri in pairs])
        dom_rmsd = float(np.sqrt(((allP - allQ) ** 2).sum(1).mean()))
    else:
        dom_rmsd = float("nan")
    diag = {"method": method, "tm_score": float(res.tm_norm_chain2),
            "n_active_pairs": len(act_pairs), "active_rmsd": round(act_rmsd, 2),
            "domain_rmsd": round(dom_rmsd, 2)}
    return xp, diag


def render(ax, coords, act, vmax, lims, title):
    """Cα worm coloured by activation on a shared viridis scale, fixed camera.

    Spatially cropped to the fold domain around the active motif: only residues
    within the camera sphere are drawn, so the sprawling extra domains of large
    multi-domain members don't clutter the panel. Backbone drawn as segments
    between consecutive in-crop residues.
    """
    cmap = plt.get_cmap("viridis")
    (xc, yc, zc), R = lims
    center = np.array([xc, yc, zc])
    a = np.clip(act / vmax if vmax > 0 else act, 0, 1)
    inside = np.linalg.norm(coords - center, axis=1) < R * 1.02
    segs, seg_c = [], []
    for i in range(len(coords) - 1):
        if inside[i] and inside[i + 1]:
            segs.append([coords[i], coords[i + 1]])
            seg_c.append(cmap((a[i] + a[i + 1]) / 2))
    if segs:
        ax.add_collection3d(Line3DCollection(np.array(segs), colors=seg_c, linewidths=3.4))
    m = (act > 0) & inside
    if m.any():
        ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                   c=cmap(a[m]), s=16, edgecolors="none", depthshade=True)
    ax.set_xlim(xc - R, xc + R); ax.set_ylim(yc - R, yc + R); ax.set_zlim(zc - R, zc + R)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, color="#1D272A", pad=0)


def process_feature(feat, outdir, evid):
    with open(f"{evid}/{feat}.json") as fh:
        ev = json.load(fh)
    rep = ev["representative"]
    r_coords, r_seq, r_act = load_struct(rep)
    members = ev["shared_members"]

    # camera: centred on the representative's active-motif centroid, framed to
    # cover the active region + local fold context. Same for every panel.
    ractive = r_coords[r_act > 0]
    center = ractive.mean(0) if len(ractive) else r_coords.mean(0)
    base = ractive if len(ractive) else r_coords
    # 85th percentile radius (robust to a few far active residues) + margin
    rad = float(np.percentile(np.linalg.norm(base - center, axis=1), 85))
    R = float(max(rad * 1.2, 18.0))
    lims = (center, R)

    # shared viridis scale across all panels of this feature
    vmax = max([r_act.max()] + [max((float(v) for v in m["activation"].values()), default=0.0)
                                for m in members])
    vmax = float(vmax) if vmax > 0 else 1.0

    diags = []
    # reference (representative) panel
    fig = plt.figure(figsize=(2.6, 2.6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    render(ax, r_coords, r_act, vmax, lims, f"reference · {rep['uniprot']}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    fig.savefig(f"{outdir}/struct_{feat}.png", transparent=True); plt.close(fig)

    for i, m in enumerate(members):
        m_coords, m_seq, m_act = load_struct(m)
        xp, diag = align_member(m_coords, m_seq, m_act, r_coords, r_seq, r_act)
        diag["uniprot"] = m["uniprot"]
        diags.append(diag)
        fig = plt.figure(figsize=(2.6, 2.6), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        render(ax, xp, m_act, vmax, lims, f"{m['uniprot']} · pLDDT {m['plddt']:.0f}")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
        fig.savefig(f"{outdir}/struct_{feat}_m{i}.png", transparent=True); plt.close(fig)
    return {"feature": feat, "vmax": round(vmax, 3), "camera_radius": round(R, 1),
            "n_active_rep": int((r_act > 0).sum()), "members": diags}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", required=True, help="output dir for the PNGs + align_report.json")
    ap.add_argument("--evid", default=DEFAULT_EVID, help="#52 evidence JSON dir")
    ap.add_argument("feats", nargs="*", type=int, help="feature ids (default: CLEAN5)")
    a = ap.parse_args(argv)

    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)
    report = {}
    feats = a.feats or CLEAN5
    for feat in feats:
        r = process_feature(feat, outdir, a.evid)
        report[feat] = r
        fb = sum(1 for d in r["members"] if d["method"] == "fallback_whole_domain")
        print(f"feature {feat}: vmax={r['vmax']} R={r['camera_radius']} "
              f"active_rep={r['n_active_rep']} fallback_members={fb}")
        for d in r["members"]:
            print(f"    {d['uniprot']}: {d['method']} tm={d['tm_score']:.2f} "
                  f"active_pairs={d['n_active_pairs']} active_rmsd={d['active_rmsd']} dom_rmsd={d['domain_rmsd']}")
    with open(f"{outdir}/align_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
