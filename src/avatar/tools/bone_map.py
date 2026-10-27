"""Xbox avatar bone index -> PC `Avatar` rig node mapping.

Authoritative source: the original Xbox 360 All-Stars image (xbimg, SHA-256 188c4946...) holds a 71-entry table of
bone-name pointers at 0x826A6328. Function 0x8260B090 bounds the avatar bone index (< 0x47), takes the name from
this table, hashes it and returns the node of the racer's model tree with that name hash, i.e. the game maps
XAvatar bone i to the `Avatar.zif` node named table[i]. This module reads that table, resolves the names against
the PC rig, and cross-checks the result with geometry (vertices of bone i must lie near PC joint table[i] after the
uniform PC/Xbox scale is fitted). Output: research/BONE-MAP.json.

Owned copy (ASR-Five-Racers, 2026-10-21): the 71 names are read from research/BONE-MAP.json (recorded from the image by
this module's main()), so a bake no longer needs the Xbox image dump under <codex-archive>. Whenever that dump is present the
names are still cross-checked against it.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
import pcanim
from catalog import catalog, decoded
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]
TABLE_VA = 0x826A6328
CONSUMER_VA = 0x8260B090
NBONES = 71
NAMES_JSON = MODULE / "research" / "BONE-MAP.json"


def xbox_bone_names_from_image():
    """The 71 names read from the Xbox image table (needs xbimg's image dump and capstone)."""
    import xbimg
    names = []
    for i in range(NBONES):
        p = xbimg.u32(TABLE_VA + 4 * i)
        if not 0x820E3900 <= p < 0x820E3CC0:
            raise ValueError(f"table entry {i} points outside the name block: {p:#x}")
        names.append(xbimg.cstr(p))
    if len(set(names)) != NBONES:
        raise ValueError("duplicate names in bone table")
    return names


def image_available() -> bool:
    """True when the Xbox image dump (and capstone) can be used for the cross-check."""
    try:
        import xbimg
    except ImportError:
        return False
    return Path(xbimg.ROOT + "xbox-image.bin").exists()


@lru_cache(None)
def xbox_bone_names():
    """The 71 Xbox bone names in table order, from research/BONE-MAP.json; cross-checked against the image when present."""
    rows = sorted(json.loads(NAMES_JSON.read_text())["rows"], key=lambda r: r["xbox_index"])
    names = [r["name"] for r in rows]
    if [r["xbox_index"] for r in rows] != list(range(NBONES)) or len(set(names)) != NBONES:
        raise ValueError("research/BONE-MAP.json does not hold the 71-entry bone table")
    if image_available() and xbox_bone_names_from_image() != names:
        raise ValueError("research/BONE-MAP.json names differ from the Xbox image table")
    return names


def pc_rig():
    z = Zif(MODULE / "work/pc-avatar/Racers_Avatar.zif", MODULE / "work/pc-avatar/Racers_Avatar.zig")
    f = z.forests[0]
    m = next(mm for mm in f["models"] if mm["root"] == "Avatar")
    bind = pcanim.bind_locals(z, f, m)
    W = pcanim.world(m, *bind)
    return z, f, m, bind, W


def mapping():
    """xbox index -> PC node index (for the rig named `root`)."""
    xb = xbox_bone_names()
    _, _, m, _, _ = pc_rig()
    pcn = [n["name"] for n in m["nodes"]]
    out = []
    for i, n in enumerate(xb):
        if pcn.count(n) != 1:
            raise ValueError(f"bone {n} not uniquely present in PC rig")
        out.append(pcn.index(n))
    return out


def seg_dist(P, a, b):
    ab = b - a
    L = float(ab @ ab)
    if L < 1e-12:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip((P - a) @ ab / L, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def main():
    import xbimg
    xb = xbox_bone_names_from_image()
    z, f, m, bind, W = pc_rig()
    pcn = [n["name"] for n in m["nodes"]]
    par = pcanim.parents(m)
    mp = mapping()
    pts = {}
    for c in catalog():
        if c["kind"] != "model":
            continue
        for me in AM.decode(decoded(c["index"]))["meshes"]:
            P = AM.positions(me)
            dom = me["bones"][np.arange(len(P)), me["weights"].argmax(1)]
            for b in np.unique(dom):
                pts.setdefault(int(b), []).append(P[dom == b])
    pts = {b: np.concatenate(v) for b, v in pts.items()}
    kids = {i: [k for k in range(len(par)) if par[k] == i] for i in range(len(par))}
    # fit uniform scale s (PC = s * Xbox) by minimizing median vertex-to-own-bone-segment distance
    best = None
    for s in np.arange(1.80, 2.30, 0.005):
        J = W[:, 3, :3] / s
        tot = []
        for b, P in pts.items():
            j = mp[b]
            ch = [k for k in kids[j] if pcn[k] in xb and not pcn[k].startswith(("LF_SC", "RT_SC", "SC_")) and "SKIN" not in pcn[k] and "PROP" not in pcn[k] and "SPECIAL" not in pcn[k]]
            end = J[ch].mean(0) if ch else J[j]
            tot.append(np.median(seg_dist(P, J[j], end)))
        val = float(np.mean(tot))
        if best is None or val < best[0]:
            best = (val, float(s))
    s = best[1]
    J = W[:, 3, :3] / s
    rows = []
    rank_ok = 0
    for b in range(NBONES):
        j = mp[b]
        r = dict(xbox_index=b, name=xb[b], pc_node=j, pc_parent=pcn[par[j]] if par[j] >= 0 else None, vertices=int(len(pts.get(b, []))))
        if b in pts:
            P = pts[b]
            d_all = np.array([np.median(np.linalg.norm(P - J[k], axis=1)) for k in range(1, len(pcn))])
            r["median_dist_to_joint"] = float(np.median(np.linalg.norm(P - J[j], axis=1)))
            r["rank_of_mapped_joint_by_distance"] = int(np.sum(d_all < r["median_dist_to_joint"]) + 1)
            rank_ok += r["rank_of_mapped_joint_by_distance"] <= 5
        rows.append(r)
    out = dict(source=dict(image=xbimg.ROOT + "xbox-image.bin", image_sha256=xbimg.SHA, table_va=hex(TABLE_VA), consumer_va=hex(CONSUMER_VA),
                           note="0x8260B090: bounds index < 0x47, hashes table[index] name, returns tree node with the same name hash"),
               scale_pc_over_xbox=s, fit_mean_median_segment_distance=best[0],
               bones_with_vertices=len(pts), mapped_joint_within_5_nearest=rank_ok, rows=rows)
    (MODULE / "research" / "BONE-MAP.json").write_text(json.dumps(out, indent=1) + "\n")
    print("scale", s, "fit", best[0], "within-5-nearest", rank_ok, "/", len(pts))
    for r in rows:
        print(r["xbox_index"], r["name"].ljust(16), r.get("vertices"), r.get("rank_of_mapped_joint_by_distance"), round(r.get("median_dist_to_joint", -1), 4))


if __name__ == "__main__":
    main()
