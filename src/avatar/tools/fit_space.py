"""Fit the uniform transform PC_bind = s * Xbox + t from joint positions estimated at skinning seams.

Historical (c253 decode): this was Codex's first fit; it runs under avatar_model profile "c253" and writes only the key
`superseded_body_frame` of research/SPACE-FIT.json (the other keys are kept). The production fit is tools/fit_space_rp6.py.

For each edge of the decoded body/clothing meshes whose two vertices are dominated by a parent bone and its child
bone (per the authoritative Xbox->PC bone table), the edge midpoint lies on the seam around the child joint. The seam
centroid estimates the child joint position in Xbox avatar space. Least squares then gives s and t (no rotation:
both rigs face +Z with +X to the avatar's left, verified by limb positions). Output: research/SPACE-FIT.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
import bone_map
import pcanim
from catalog import decoded

MODULE = Path(__file__).resolve().parents[1]
SOURCES = [0, 1]  # MaleBody, FemaleBody (skin covers every limb seam)


def RELIABLE(name):
    """Seams that wrap a joint of a skinned chain (finger segments 02/03); torso, toe, collar and helper seams do not
    sit on the joint centre and were observed to bias the fit (first run kept in research/SPACE-FIT-all-seams.json)."""
    return name.endswith(("_02", "_03"))


# skin-bone pairs whose shared seam wraps a named joint (helpers carry the skin of thigh/calf/upper/lower arm)
PAIR_JOINT = []
for S, L in (("LF", "Left"), ("RT", "Right")):
    PAIR_JOINT += [((f"{S}_SC_H", f"{S}_SC_K"), f"{S}_K"), ((f"{S}_SC_K", f"{S}_A"), f"{S}_A"),
                   ((f"{S}_SC_S_SKIN", f"{S}_E_SKIN"), f"{L}Elbow"), ((f"{S}_SC_S", f"{S}_E_SKIN"), f"{L}Elbow"),
                   ((f"{S}_SC_E", f"{L}Hand"), f"{L}Hand")]


def main():
    AM.set_decode("c253")   # the fit this script documents was made with the factor-2 decode (research/FRAME-NOTES.md)
    xb = bone_map.xbox_bone_names()
    mp = bone_map.mapping()
    z, f, m, bind, W = bone_map.pc_rig()
    pcn = [n["name"] for n in m["nodes"]]
    par = pcanim.parents(m)
    inv = {j: i for i, j in enumerate(mp)}
    seams = {}
    for src in SOURCES:
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
            pj = [j for (p, q), j in PAIR_JOINT if {p, q} == {xb[ba], xb[bb]}]
            if pj:
                seams.setdefault((src, xb.index(pj[0]) if pj[0] in xb else pj[0]), []).append((P[a] + P[b]) / 2)
                continue
            # child joint of a parent/child pair
            if par[jb] == ja:
                child = bb
            elif par[ja] == jb:
                child = ba
            else:
                continue
            seams.setdefault((src, child), []).append((P[a] + P[b]) / 2)
    X, Y, names = [], [], []
    for (src, child), pts in seams.items():
        if len(pts) < 6 or not (RELIABLE(xb[child]) or any(xb[child] == j for _, j in PAIR_JOINT)):
            continue
        X.append(np.mean(pts, 0))
        Y.append(W[mp[child], 3, :3])
        names.append((src, xb[child], len(pts)))
    X, Y = np.array(X), np.array(Y)
    # uniform model: Y = s X + t
    A = np.zeros((3 * len(X), 4))
    for k in range(3):
        A[k::3, 0] = X[:, k]
        A[k::3, 1 + k] = 1
    sol, *_ = np.linalg.lstsq(A, Y.reshape(-1), rcond=None)
    s, t = sol[0], sol[1:]
    res_u = np.linalg.norm(Y - (s * X + t), axis=1)
    # diagonal model: Y_k = s_k X_k + t_k per axis
    sd, td = np.zeros(3), np.zeros(3)
    for k in range(3):
        M = np.stack([X[:, k], np.ones(len(X))], 1)
        (sd[k], td[k]), *_ = np.linalg.lstsq(M, Y[:, k], rcond=None)
    res = np.linalg.norm(Y - (X * sd + td), axis=1)
    rows = [dict(source=n[0], joint=n[1], seam_edges=n[2], xbox=X[i].round(5).tolist(), pc=Y[i].round(5).tolist(), residual_uniform=float(res_u[i]),
                 residual_diagonal=float(res[i])) for i, n in enumerate(names)]
    out = dict(uniform=dict(scale=float(s), translation=t.tolist(), residual_median=float(np.median(res_u)), residual_max=float(res_u.max())),
               diagonal=dict(scale=sd.tolist(), translation=td.tolist(), residual_median=float(np.median(res)), residual_max=float(res.max())),
               joints=len(X), rows=rows,
               note="fits to MaleBody seams; anisotropic because the body's upper half is ~0.155 higher than the garment/head frame (research/FRAME-NOTES.md)")
    p = MODULE / "research" / "SPACE-FIT.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["superseded_body_frame"] = out          # the other keys (uniform_garment_frame, uniform_rp6) are kept
    p.write_text(json.dumps(d, indent=1) + "\n")
    print("uniform s", s, "t", t, "median", np.median(res_u), "max", res_u.max())
    print("diagonal s", sd, "t", td, "n", len(X), "median res", np.median(res), "max", res.max())
    for r in sorted(rows, key=lambda r: -r["residual_diagonal"])[:12]:
        print(r["source"], r["joint"], r["seam_edges"], round(r["residual_diagonal"], 4))


if __name__ == "__main__":
    main()
