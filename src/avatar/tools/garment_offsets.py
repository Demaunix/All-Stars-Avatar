"""Measure per-garment rigid offsets against the matching body (MaleBody / FemaleBody) from shared skinning seams.

For every decoded skinned garment, edges whose endpoints are dominated by two different bones form seams; seams
that also exist on the body give paired joint estimates. The median delta (garment - body) is reported per
garment with its spread. Output research/GARMENT-OFFSETS.json for the active decode profile (rp6: everything within
~2 cm). The factor-2 decode's result (tops -0.15, gloves -0.20: the artefact that led to the per-axis decode) is kept as
research/GARMENT-OFFSETS-c253-decode.json. Read-only, offline."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
import bone_map
from catalog import catalog, decoded

MODULE = Path(__file__).resolve().parents[1]
XB = bone_map.xbox_bone_names()


def seams(me):
    P = AM.positions(me)
    dom = me["bones"][np.arange(len(P)), me["weights"].argmax(1)]
    T = me["triangles"]
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    out = {}
    for a, b in E:
        ba, bb = int(dom[a]), int(dom[b])
        if ba != bb:
            out.setdefault(tuple(sorted((XB[ba], XB[bb]))), []).append((P[a] + P[b]) / 2)
    return {k: np.mean(v, 0) for k, v in out.items() if len(v) >= 4}


def main():
    bodies = {1: seams(AM.decode(decoded(0))["meshes"][0]), 2: seams(AM.decode(decoded(1))["meshes"][0])}
    rows = []
    for c in catalog():
        if c["kind"] != "model" or c["index"] in (0, 1, 2):
            continue
        m = AM.decode(decoded(c["index"]))
        if m["empty"]:
            continue
        gender = (c["variant"] >> 16) & 0xFF  # 1 male, 2 female, 3 both (second byte of the TOC variant word)
        for g in ([1, 2] if gender == 3 else [gender]):
            if g not in bodies:
                continue
            ds = []
            for me in m["meshes"]:
                s = seams(me)
                for k in set(s) & set(bodies[g]):
                    ds.append(s[k] - bodies[g][k])
            if len(ds) < 2:
                continue
            ds = np.array(ds)
            med = np.median(ds, 0)
            rows.append(dict(index=c["index"], name=c["name"], type=hex(c["type"]), body="male" if g == 1 else "female", seams=len(ds),
                             median_delta=med.round(4).tolist(), spread=np.median(np.abs(ds - med), 0).round(4).tolist()))
    by_type = {}
    for r in rows:
        by_type.setdefault((r["type"], r["body"]), []).append(r["median_delta"][1])
    summary = {f"{k[0]}/{k[1]}": dict(n=len(v), dy_median=float(np.median(v)), dy_min=float(np.min(v)), dy_max=float(np.max(v)),
                                       dy_hist=Counter(round(x, 2) for x in v).most_common(6)) for k, v in sorted(by_type.items())}
    (MODULE / "research" / "GARMENT-OFFSETS.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
