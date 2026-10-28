"""Uniform similarity PC_bind = s * X + t from GARMENT-frame joints (supersedes the body-frame diagonal fit).

Historical (c253 decode): the fit of the C253 / rp4 / rp5 bakes (s = 2.1679). It runs under avatar_model profile "c253" and
writes only the key `uniform_garment_frame` of research/SPACE-FIT.json (the other keys are kept), which the c253 reproduction
path (tests/test_control_rp4.py) reads. The production fit is tools/fit_space_rp6.py.

Joints: elbows from every top that has the (SC_E, SC_S) seam, knees from every trouser that has the (SC_H, SC_K)
seam, for the male garments (the PC rig is one skeleton). Rotation is not fitted (both face +Z, +X left).
Output: research/SPACE-FIT.json key `uniform_garment_frame` (earlier fits preserved under `superseded_*`)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
import bone_map
from catalog import catalog, decoded

MODULE = Path(__file__).resolve().parents[1]
SPEC = []
for S, L in (("LF", "Left"), ("RT", "Right")):
    SPEC += [((f"{S}_SC_E", f"{S}_SC_S"), f"{L}Elbow"), ((f"{S}_SC_H", f"{S}_SC_K"), f"{S}_K")]


def seams(me, xb):
    P = AM.positions(me)
    dom = me["bones"][np.arange(len(P)), me["weights"].argmax(1)]
    T = me["triangles"]
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    out = {}
    for a, b in E:
        ba, bb = int(dom[a]), int(dom[b])
        if ba != bb:
            out.setdefault(frozenset((xb[ba], xb[bb])), []).append((P[a] + P[b]) / 2)
    return {k: np.mean(v, 0) for k, v in out.items() if len(v) >= 4}


def main():
    AM.set_decode("c253")   # the garment-frame fit exists only under the factor-2 decode (research/FRAME-NOTES.md)
    xb = bone_map.xbox_bone_names()
    z, f, m, bind, W = bone_map.pc_rig()
    pcn = [n["name"] for n in m["nodes"]]
    X, Y, lab = [], [], []
    for c in catalog():
        if c["kind"] != "model" or c["type"] not in (0x8, 0x208, 0x10) or ((c["variant"] >> 16) & 0xFF) != 1:
            continue
        mm = AM.decode(decoded(c["index"]))
        for me in mm["meshes"]:
            s = seams(me, xb)
            for pair, joint in SPEC:
                k = frozenset(pair)
                if k in s:
                    X.append(s[k]); Y.append(W[pcn.index(joint), 3, :3]); lab.append((c["index"], c["name"], joint))
    X, Y = np.array(X), np.array(Y)
    A = np.zeros((3 * len(X), 4))
    for k in range(3):
        A[k::3, 0] = X[:, k]
        A[k::3, 1 + k] = 1
    sol, *_ = np.linalg.lstsq(A, Y.reshape(-1), rcond=None)
    s, t = float(sol[0]), sol[1:]
    r = np.linalg.norm(Y - (s * X + t), axis=1)
    # robust re-fit without the worst 10% (garments differ slightly in cut)
    keep = r <= np.percentile(r, 90)
    sol2, *_ = np.linalg.lstsq(A.reshape(len(X), 3, 4)[keep].reshape(-1, 4), Y[keep].reshape(-1), rcond=None)
    s2, t2 = float(sol2[0]), sol2[1:]
    r2 = np.linalg.norm(Y - (s2 * X + t2), axis=1)
    p = MODULE / "research" / "SPACE-FIT.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["uniform_garment_frame"] = dict(scale=s2, translation=t2.tolist(), joints=int(len(X)), used=int(keep.sum()),
                                      residual_median=float(np.median(r2[keep])), residual_max_used=float(r2[keep].max()),
                                      residual_max_all=float(r2.max()), first_pass=dict(scale=s, translation=t.tolist(), residual_median=float(np.median(r)),
                                                                                         residual_max=float(r.max())),
                                      garments=sorted({f"{a} {b}" for a, b, _ in lab}))
    p.write_text(json.dumps(d, indent=1) + "\n")       # the other keys (superseded_body_frame, uniform_rp6) are kept
    print(json.dumps({k: v for k, v in d["uniform_garment_frame"].items() if k != "garments"}, indent=1), len(d["uniform_garment_frame"]["garments"]))


if __name__ == "__main__":
    main()
