"""Face shape overrides for the head mesh (nose 0x80000, chin 0x100000, ears 0x200000; M10 step 4, 2026-10-21).

Same block layout as the hiding templates (avatar_hiding / research/HIDING-TEMPLATES.md) with no hidden triangles and the
head as target: header word 3 = gender << 16 | target TOC index (0x30002 = both, MaleBodyHead), index-bytes word = the head's
total index bytes (2 meshes x 1269 x 2 = 5076), vertex-bytes word = 648 vertices x 52-byte stride rounded to 128 (the head
carries 12 texcoord halves, so its stride is 28 + 24 = 52, not the body's 40). Each override record replaces one head vertex
(both meshes indexed as one buffer: 0..323 mesh 0, 324..647 mesh 1): position, normal, skin, colour; texcoords kept.
The blocks with zero overrides are the defaults (Small Chin 801, Angled Nose 807, Round Ears 823).

Frame (rp6, 2026-10-27): with the factor-2 decode the override positions sat at a constant per-region offset from the stock head
(noses about (0, -0.021, -0.051), ears (0, -0.024, -0.018)) and `apply` subtracted that mean ("shape only"). With the per-axis
decode (avatar_model.DECODES["rp6"]) the offsets vanish (nose (0, -0.001, +0.002), ears (0, -0.002, 0.0005), chins
(0, -0.002, 0.004); research/xbox-layering-analysis/minfit/verifyA.py), so the overrides are applied as stored and the
remaining mean offset is only recorded.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_model as AM
from catalog import catalog, raw_block

HEAD_TARGET = 0x30002
HEAD_STRIDE = 52
HEAD_INDEX = 2
DEFAULTS = {0x100000: 801, 0x80000: 807, 0x200000: 823}


class FaceShapeError(Exception):
    pass


def decode(data: bytes) -> dict:
    """Decode a head override block: vertex indices (0..647), positions, normals, weights, bones, colour."""
    n, ibytes, one, target = struct.unpack_from("<4I", data, 0)
    if n != 0 or target != HEAD_TARGET:
        raise FaceShapeError(f"not a head override block (hidden {n}, target {target:#x})")
    p = 32
    nv, vbytes, one2, target2 = struct.unpack_from("<4I", data, p)
    obits = struct.unpack_from("<I", data, p + 28)[0]
    step, mx, my, mz = struct.unpack_from("<4f", data, p + 32)
    if nv == 0:      # the default shapes carry no records (offset_bits is 0 there)
        return dict(vertices=np.zeros(0, int), positions=np.zeros((0, 3)), normal=np.zeros((0, 3)), weights=np.zeros((0, 4), int),
                    bones=np.zeros((0, 4), int), color=np.zeros((0, 4), int), index_bytes=ibytes, vertex_bytes=vbytes)
    if target2 != HEAD_TARGET or abs(step - AM.STEP) > 1e-12 or not 0 < obits <= 32:
        raise FaceShapeError("bad override header")
    b = AM.Bits(data)
    s = (p + 48) * 8
    widths = [b.val(s + 6 * k, 6) for k in range(3)]
    nb, nbits, wb, wbits, ib, ibits, cb, cbits = [b.val(s + 18 + 32 * k, 32) for k in range(8)]
    rec = obits + sum(widths) + nbits + wbits + ibits + cbits
    vstart = s + 274
    if (vstart + rec * nv + 7) // 8 != len(data):
        raise FaceShapeError("override records do not consume the block")
    st = vstart + rec * np.arange(nv)
    off = b.column(st, 0, obits)
    o = obits
    Q = np.stack([b.column(st, o + sum(widths[:k]), widths[k]) for k in range(3)], 1).astype(np.float64)
    o += sum(widths)
    normal_raw = (b.column(st, o, nbits) + nb) & 0xFFFFFFFF
    o += nbits
    weights = AM.bytes4(b.column(st, o, wbits) + wb)
    o += wbits
    bones = AM.bytes4(b.column(st, o, ibits) + ib)
    o += ibits
    color = AM.bytes4(b.column(st, o, cbits) + cb)
    if (off % HEAD_STRIDE).any():
        raise FaceShapeError("override offset not on a head vertex boundary")
    return dict(vertices=(off // HEAD_STRIDE).astype(int), positions=np.array([mx, my, mz]) + Q * step * AM.K, quantized=Q, min=(mx, my, mz),
                normal=AM.hend3n(normal_raw), weights=weights, bones=bones, color=color, index_bytes=ibytes, vertex_bytes=vbytes)


def apply(head_meshes: list, indices: list) -> tuple:
    """Return copies of the two head meshes with the overrides of `indices` applied as stored, and a record (per block: the mean
    offset of the overridden vertices against the stock head; per-region average below 4 mm with the per-axis decode, single
    blocks up to 7 mm, e.g. Defined Nose, Large Chin)."""
    names = {c["index"]: c["name"] for c in catalog()}
    counts = [len(m["quantized"]) for m in head_meshes]
    if sum(counts) != 648 or len(head_meshes) != 2:
        raise FaceShapeError("expected the two 324-vertex head meshes")
    out = []
    for m in head_meshes:
        c = dict(m)
        c["pos_override"] = AM.positions(m).copy()
        for k in ("normal", "normal_raw", "weights", "bones", "color"):
            c[k] = m[k].copy()
        out.append(c)
    rec = []
    for idx in indices:
        if idx is None or idx in DEFAULTS.values():
            rec.append(dict(index=idx, name=names.get(idx), applied=False, reason="default shape (no overrides)"))
            continue
        t = decode(raw_block(idx))
        stock = np.concatenate([AM.positions(m) for m in head_meshes])
        delta = t["positions"] - stock[t["vertices"]]
        mean = delta.mean(0)
        pos = t["positions"]
        for j, v in enumerate(t["vertices"]):
            mi, vi = (0, v) if v < counts[0] else (1, v - counts[0])
            out[mi]["pos_override"][vi] = pos[j]
            out[mi]["normal"][vi] = t["normal"][j]
        rec.append(dict(index=idx, name=names.get(idx), applied=True, overrides=int(len(t["vertices"])),
                        mean_offset=[round(float(x), 4) for x in mean], residual_median=round(float(np.median(np.linalg.norm(delta - mean, axis=1))), 4)))
    return out, rec


if __name__ == "__main__":
    for i in (803, 810, 794, 821):
        t = decode(raw_block(i))
        print(i, len(t["vertices"]), t["vertices"][:5], np.round(t["positions"][:2], 3))
