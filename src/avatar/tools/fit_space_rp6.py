"""Uniform similarity PC_bind = s * X + t for the per-axis decode (rp6, 2026-10-27): body and garment seams in one frame.

With avatar_model.DECODES["rp6"] the body, its garments and the head share one frame (research/GARMENT-OFFSETS.json), so the
joint estimates of fit_space.py (MaleBody / FemaleBody seams at the reliable joints: finger segments, knees, ankles, elbows,
hands) and of fit_space_garment.py (elbows of every male top, knees of every male trouser) are pooled into one least-squares
fit, re-fitted without the worst 10 % (garments differ in cut). The per-axis (diagonal) fit of the same joints is reported as a
diagnostic: a uniform fit is only right when the three scales agree. The two subsets are fitted on their own as well.
Writes key `uniform_rp6` into research/SPACE-FIT.json; the c253 keys are kept for the reproduction path.
usage: fit_space_rp6.py [--dry]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
import bone_map
import fit_space as FSB
import fit_space_garment as FSG
import pcanim
from catalog import catalog, decoded

MODULE = Path(__file__).resolve().parents[1]
KEY = "uniform_rp6"


def body_joints() -> tuple:
    """Seam joints of MaleBody / FemaleBody at the reliable joints, selected exactly as fit_space.py does."""
    xb = bone_map.xbox_bone_names()
    mp = bone_map.mapping()
    z, f, m, bind, W = bone_map.pc_rig()
    par = pcanim.parents(m)
    seams = {}
    for src in FSB.SOURCES:
        me = AM.decode(decoded(src))["meshes"][0]
        P = AM.positions(me)
        dom = me["bones"][np.arange(len(P)), me["weights"].argmax(1)]
        T = me["triangles"]
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        for a, b in E:
            ba, bb = int(dom[a]), int(dom[b])
            if ba == bb:
                continue
            ja, jb = mp[ba], mp[bb]
            pj = [j for (p, q), j in FSB.PAIR_JOINT if {p, q} == {xb[ba], xb[bb]}]
            if pj:
                seams.setdefault((src, xb.index(pj[0]) if pj[0] in xb else pj[0]), []).append((P[a] + P[b]) / 2)
                continue
            if par[jb] == ja:
                child = bb
            elif par[ja] == jb:
                child = ba
            else:
                continue
            seams.setdefault((src, child), []).append((P[a] + P[b]) / 2)
    X, Y, lab = [], [], []
    for (src, child), pts in seams.items():
        if len(pts) < 6 or not (FSB.RELIABLE(xb[child]) or any(xb[child] == j for _, j in FSB.PAIR_JOINT)):
            continue
        X.append(np.mean(pts, 0))
        Y.append(W[mp[child], 3, :3])
        lab.append(dict(subset="body", source=int(src), joint=xb[child], seam_edges=int(len(pts))))
    return X, Y, lab


def garment_joints() -> tuple:
    """Elbow / knee seam joints of every male top and trouser, selected exactly as fit_space_garment.py does."""
    xb = bone_map.xbox_bone_names()
    z, f, m, bind, W = bone_map.pc_rig()
    pcn = [n["name"] for n in m["nodes"]]
    X, Y, lab = [], [], []
    for c in catalog():
        if c["kind"] != "model" or c["type"] not in (0x8, 0x208, 0x10) or ((c["variant"] >> 16) & 0xFF) != 1:
            continue
        mm = AM.decode(decoded(c["index"]))
        for me in mm["meshes"]:
            s = FSG.seams(me, xb)
            for pair, joint in FSG.SPEC:
                k = frozenset(pair)
                if k in s:
                    X.append(s[k])
                    Y.append(W[pcn.index(joint), 3, :3])
                    lab.append(dict(subset="garment", source=int(c["index"]), name=c["name"], joint=joint))
    return X, Y, lab


def uniform(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Least-squares Y = s X + t (one scale, no rotation: both rigs face +Z with +X to the avatar's left)."""
    A = np.zeros((3 * len(X), 4))
    for k in range(3):
        A[k::3, 0] = X[:, k]
        A[k::3, 1 + k] = 1
    sol, *_ = np.linalg.lstsq(A, Y.reshape(-1), rcond=None)
    return float(sol[0]), sol[1:]


def diagonal(X: np.ndarray, Y: np.ndarray) -> dict:
    """Per-axis Y_k = s_k X_k + t_k (diagnostic: the scale ratios say whether one uniform scale is right)."""
    sd, td = np.zeros(3), np.zeros(3)
    for k in range(3):
        M = np.stack([X[:, k], np.ones(len(X))], 1)
        (sd[k], td[k]), *_ = np.linalg.lstsq(M, Y[:, k], rcond=None)
    r = np.linalg.norm(Y - (X * sd + td), axis=1)
    return dict(scale=sd.tolist(), translation=td.tolist(), y_over_x=float(sd[1] / sd[0]), z_over_x=float(sd[2] / sd[0]),
                residual_median=float(np.median(r)), residual_max=float(r.max()))


def robust(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Uniform fit, then a re-fit without the worst 10 % of the joints (as fit_space_garment.py); returns (record, kept mask)."""
    s, t = uniform(X, Y)
    r = np.linalg.norm(Y - (s * X + t), axis=1)
    keep = r <= np.percentile(r, 90)
    s2, t2 = uniform(X[keep], Y[keep])
    r2 = np.linalg.norm(Y - (s2 * X + t2), axis=1)
    rec = dict(scale=s2, translation=t2.tolist(), joints=int(len(X)), used=int(keep.sum()), residual_median=float(np.median(r2[keep])),
               residual_max_used=float(r2[keep].max()), residual_max_all=float(r2.max()),
               first_pass=dict(scale=s, translation=t.tolist(), residual_median=float(np.median(r)), residual_max=float(r.max())))
    return rec, keep, r2


def main() -> None:
    """Fit, print and (unless --dry) write research/SPACE-FIT.json[uniform_rp6]."""
    if AM.DECODE_NAME != "rp6":
        raise SystemExit("fit_space_rp6 needs the rp6 decode profile")
    Xb, Yb, lb = body_joints()
    Xg, Yg, lg = garment_joints()
    X, Y, lab = np.array(Xb + Xg), np.array(Yb + Yg), lb + lg
    pooled, keep, r = robust(X, Y)
    body_only, _, _ = robust(np.array(Xb), np.array(Yb))
    garment_only, _, _ = robust(np.array(Xg), np.array(Yg))
    rows = [dict(l, xbox=X[i].round(5).tolist(), pc=Y[i].round(5).tolist(), residual=float(r[i]), used=bool(keep[i])) for i, l in enumerate(lab)]
    out = dict(pooled, decode=dict(profile=AM.DECODE_NAME, factors=AM.K.tolist()),
               subsets=dict(body=body_only, garments=garment_only),
               diagonal=dict(pooled=diagonal(X[keep], Y[keep]), body=diagonal(np.array(Xb), np.array(Yb))),
               method="body seams at reliable joints (fit_space.py selection) + male top elbows / trouser knees (fit_space_garment.py selection), "
                      "one frame under the per-axis decode; uniform least squares, re-fit without the worst 10 %",
               rows=rows)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if "--dry" in sys.argv:
        return
    p = MODULE / "research" / "SPACE-FIT.json"
    d = json.loads(p.read_text())
    d[KEY] = out
    p.write_text(json.dumps(d, indent=1) + "\n")
    print("wrote", p, KEY)


if __name__ == "__main__":
    main()
